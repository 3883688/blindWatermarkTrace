from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .config import V4Config
from .payload import (
    permute_codeword_bits,
    phase_for_tile,
    phase_permutation,
)


CELL_SIZE = 16
GRID_SIZE = 8
TILE_SIZE = 128
BIT_COUNT = 64
LUMINANCE_CENTER = 128.0
DCT_TILE_BATCH = 8


@dataclass(frozen=True, slots=True)
class TileScores:
    tile_x: int
    tile_y: int
    phase: int
    logical_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.tile_x) is not int or type(self.tile_y) is not int:
            raise TypeError("tile coordinates must be integers")
        if self.tile_x < 0 or self.tile_y < 0:
            raise ValueError("tile coordinates must be nonnegative")
        if type(self.phase) is not int:
            raise TypeError("phase must be an integer")
        if self.phase not in range(4):
            raise ValueError("phase must be between 0 and 3")
        if type(self.logical_scores) is not tuple:
            raise TypeError("scores must be a tuple")
        if len(self.logical_scores) != BIT_COUNT:
            raise ValueError("scores must contain exactly 64 values")
        if any(
            type(score) is not float or not np.isfinite(score)
            for score in self.logical_scores
        ):
            raise ValueError("scores must contain only finite floats")


def _dct_basis(size: int) -> np.ndarray:
    return _cached_dct_basis(size).view()


@lru_cache(maxsize=1)
def _cached_dct_basis(size: int) -> np.ndarray:
    if type(size) is not int:
        raise TypeError("DCT basis size must be an integer")
    if size != CELL_SIZE:
        raise ValueError("DCT basis size must be exactly 16")

    frequencies = np.arange(size, dtype=np.float64)[:, None]
    samples = np.arange(size, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * (samples + 0.5) * frequencies / size)
    basis[0] *= np.sqrt(1.0 / size)
    basis[1:] *= np.sqrt(2.0 / size)
    basis.flags.writeable = False
    return basis


def _forward_dct_blocks(blocks: np.ndarray) -> np.ndarray:
    values = _validated_blocks(blocks)
    basis = _dct_basis(CELL_SIZE)
    with np.errstate(over="ignore", invalid="ignore"):
        result = basis @ values @ basis.T
    return _validated_output(result)


def _inverse_dct_blocks(blocks: np.ndarray) -> np.ndarray:
    values = _validated_blocks(blocks)
    basis = _dct_basis(CELL_SIZE)
    with np.errstate(over="ignore", invalid="ignore"):
        result = basis.T @ values @ basis
    return _validated_output(result)


def embed_tile_bits(
    luminance_tile: np.ndarray,
    bits: tuple[int, ...],
    config: V4Config,
) -> np.ndarray:
    tile = _validated_tile(luminance_tile)
    _validate_bits(bits)
    _validate_config(config)

    centered_blocks = _tile_to_blocks(tile - LUMINANCE_CENTER)
    coefficients = _forward_dct_blocks(centered_blocks)[None, ...]
    physical_bits = np.asarray(bits, dtype=np.float64)[None, ...]
    _embed_coefficients(coefficients, physical_bits, config)
    restored = _blocks_to_tile(_inverse_dct_blocks(coefficients[0]))
    return _validated_output(restored + LUMINANCE_CENTER)


def extract_tile_scores(
    luminance_tile: np.ndarray,
    config: V4Config,
) -> tuple[float, ...]:
    tile = _validated_tile(luminance_tile)
    _validate_config(config)

    centered_blocks = _tile_to_blocks(tile - LUMINANCE_CENTER)
    coefficients = _forward_dct_blocks(centered_blocks)[None, ...]
    scores = _extract_coefficient_scores(coefficients, config)[0]
    return tuple(scores.tolist())


def embed_codeword(
    image: Image.Image,
    codeword: bytes,
    config: V4Config = V4Config(),
) -> Image.Image:
    _validate_image(image)
    if type(codeword) is not bytes:
        raise TypeError("codeword must be bytes")
    if len(codeword) != 8:
        raise ValueError("codeword must contain exactly 8 bytes")
    _validate_config(config)
    tiles = _eligible_tiles(image, config)

    source = np.asarray(image)
    output = source.copy()
    for batch_start in range(0, len(tiles), DCT_TILE_BATCH):
        batch = tiles[batch_start : batch_start + DCT_TILE_BATCH]
        tile_count = len(batch)
        rgb_strip = np.concatenate(
            tuple(
                source[
                    tile_y * TILE_SIZE : (tile_y + 1) * TILE_SIZE,
                    tile_x * TILE_SIZE : (tile_x + 1) * TILE_SIZE,
                    :3,
                ]
                for tile_x, tile_y, _ in batch
            ),
            axis=0,
        )
        ycrcb_strip = cv2.cvtColor(rgb_strip, cv2.COLOR_RGB2YCrCb)
        luminance_tiles = ycrcb_strip[..., 0].reshape(
            tile_count,
            TILE_SIZE,
            TILE_SIZE,
        )
        centered_blocks = np.stack(
            tuple(
                _tile_to_blocks(tile.astype(np.float64) - LUMINANCE_CENTER)
                for tile in luminance_tiles
            ),
            axis=0,
        )
        coefficients = _forward_dct_blocks(
            centered_blocks.reshape(
                tile_count * BIT_COUNT,
                CELL_SIZE,
                CELL_SIZE,
            )
        ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)
        physical_bits = np.asarray(
            [permute_codeword_bits(codeword, phase) for _, _, phase in batch],
            dtype=np.float64,
        )
        _embed_coefficients(coefficients, physical_bits, config)
        restored_blocks = _inverse_dct_blocks(
            coefficients.reshape(
                tile_count * BIT_COUNT,
                CELL_SIZE,
                CELL_SIZE,
            )
        ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)

        for tile_index in range(tile_count):
            top = tile_index * TILE_SIZE
            embedded_y = (
                _blocks_to_tile(restored_blocks[tile_index]) + LUMINANCE_CENTER
            )
            ycrcb_strip[top : top + TILE_SIZE, :, 0] = np.clip(
                np.rint(embedded_y),
                0,
                255,
            ).astype(np.uint8)

        converted_strip = cv2.cvtColor(ycrcb_strip, cv2.COLOR_YCrCb2RGB)
        for tile_index, (tile_x, tile_y, _) in enumerate(batch):
            source_top = tile_index * TILE_SIZE
            target_top = tile_y * TILE_SIZE
            target_left = tile_x * TILE_SIZE
            output[
                target_top : target_top + TILE_SIZE,
                target_left : target_left + TILE_SIZE,
                :3,
            ] = converted_strip[
                source_top : source_top + TILE_SIZE,
                :,
            ]

    return Image.fromarray(output)


def extract_image_tiles(
    image: Image.Image,
    config: V4Config = V4Config(),
) -> tuple[TileScores, ...]:
    _validate_image(image)
    _validate_config(config)
    tiles = _eligible_tiles(image, config)

    rgb = np.asarray(image)[..., :3]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    centered_blocks = _gather_centered_blocks(ycrcb[..., 0], tiles)
    tile_count = len(tiles)
    coefficients = _forward_dct_blocks(
        centered_blocks.reshape(tile_count * BIT_COUNT, CELL_SIZE, CELL_SIZE)
    ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)
    physical_score_batches = _extract_coefficient_scores(coefficients, config)
    records = []
    for tile_index, (tile_x, tile_y, phase) in enumerate(tiles):
        logical_scores = physical_score_batches[tile_index][
            np.asarray(phase_permutation(phase), dtype=np.intp)
        ]
        records.append(
            TileScores(
                tile_x=tile_x,
                tile_y=tile_y,
                phase=phase,
                logical_scores=tuple(logical_scores.tolist()),
            )
        )
    return tuple(records)


def _embed_coefficients(
    coefficients: np.ndarray,
    physical_bits: np.ndarray,
    config: V4Config,
) -> None:
    pairs = np.asarray(config.coefficient_pairs, dtype=np.intp)
    first_rows = pairs[:, 0, 0]
    first_columns = pairs[:, 0, 1]
    second_rows = pairs[:, 1, 0]
    second_columns = pairs[:, 1, 1]
    signs = (2.0 * physical_bits - 1.0)[..., None]
    differences = (
        coefficients[:, :, first_rows, first_columns]
        - coefficients[:, :, second_rows, second_columns]
    )
    corrections = np.maximum(config.dct_margin - signs * differences, 0.0) / 2.0
    coefficients[:, :, first_rows, first_columns] += signs * corrections
    coefficients[:, :, second_rows, second_columns] -= signs * corrections


def _extract_coefficient_scores(
    coefficients: np.ndarray,
    config: V4Config,
) -> np.ndarray:
    pairs = np.asarray(config.coefficient_pairs, dtype=np.intp)
    differences = (
        coefficients[:, :, pairs[:, 0, 0], pairs[:, 0, 1]]
        - coefficients[:, :, pairs[:, 1, 0], pairs[:, 1, 1]]
    )
    return np.mean(differences, axis=2) / config.dct_margin


def _gather_centered_blocks(
    luminance: np.ndarray,
    tiles: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    return np.stack(
        tuple(
            _tile_to_blocks(
                luminance[
                    tile_y * TILE_SIZE : (tile_y + 1) * TILE_SIZE,
                    tile_x * TILE_SIZE : (tile_x + 1) * TILE_SIZE,
                ].astype(np.float64)
                - LUMINANCE_CENTER
            )
            for tile_x, tile_y, _ in tiles
        ),
        axis=0,
    )


def _validated_blocks(blocks: np.ndarray) -> np.ndarray:
    if not isinstance(blocks, np.ndarray):
        raise TypeError("DCT blocks must be a NumPy array")
    if blocks.ndim != 3 or blocks.shape[1:] != (CELL_SIZE, CELL_SIZE):
        raise ValueError("DCT blocks must have shape (N, 16, 16)")
    if blocks.dtype.kind not in "iuf":
        raise TypeError("DCT blocks must contain real numeric values")
    if not np.isfinite(blocks).all():
        raise ValueError("DCT blocks must contain only finite values")

    values = blocks.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("DCT blocks must be representable as finite float64 values")
    return values


def _validated_tile(luminance_tile: np.ndarray) -> np.ndarray:
    if type(luminance_tile) is not np.ndarray:
        raise TypeError("luminance tile must be a NumPy array")
    if luminance_tile.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError("luminance tile must have shape (128, 128)")
    if luminance_tile.dtype.kind not in "iuf":
        raise TypeError("luminance tile must contain real numeric values")
    if not np.isfinite(luminance_tile).all():
        raise ValueError("luminance tile must contain only finite values")

    tile = luminance_tile.astype(np.float64, copy=False)
    if not np.isfinite(tile).all():
        raise ValueError("luminance tile must be representable as finite float64 values")
    return tile


def _validate_bits(bits: tuple[int, ...]) -> None:
    if type(bits) is not tuple:
        raise TypeError("bits must be a tuple")
    if len(bits) != BIT_COUNT:
        raise ValueError("bits must contain exactly 64 values")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only integer zero or one values")


def _validate_config(config: V4Config) -> None:
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config instance")


def _validate_image(image: Image.Image) -> None:
    if type(image) is not Image.Image:
        raise TypeError("image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")


def _eligible_tiles(
    image: Image.Image,
    config: V4Config,
) -> tuple[tuple[int, int, int], ...]:
    columns = image.width // TILE_SIZE
    rows = image.height // TILE_SIZE
    tiles = tuple(
        (tile_x, tile_y, phase_for_tile(tile_x, tile_y))
        for tile_y in range(rows)
        for tile_x in range(columns)
    )
    if len(tiles) < config.minimum_tiles:
        raise ValueError("image does not contain the minimum number of complete tiles")
    if len({phase for _, _, phase in tiles}) < config.minimum_phases:
        raise ValueError("image does not contain enough distinct tile phases")
    return tiles


def _tile_to_blocks(tile: np.ndarray) -> np.ndarray:
    return (
        tile.reshape(GRID_SIZE, CELL_SIZE, GRID_SIZE, CELL_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(BIT_COUNT, CELL_SIZE, CELL_SIZE)
    )


def _blocks_to_tile(blocks: np.ndarray) -> np.ndarray:
    return (
        blocks.reshape(GRID_SIZE, GRID_SIZE, CELL_SIZE, CELL_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(TILE_SIZE, TILE_SIZE)
    )


def _validated_output(result: np.ndarray) -> np.ndarray:
    if not np.isfinite(result).all():
        raise ValueError("DCT transform must produce only finite output values")
    return result


__all__ = (
    "TileScores",
    "embed_codeword",
    "embed_tile_bits",
    "extract_image_tiles",
    "extract_tile_scores",
)
