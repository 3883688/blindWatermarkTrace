import ast
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytest
from PIL import Image

import watermark_v4
import watermark_v4.dct as dct_module
from watermark_v4 import V4Config
from watermark_v4.dct import (
    TileScores,
    _cached_dct_basis,
    _dct_basis,
    _forward_dct_blocks,
    _inverse_dct_blocks,
    embed_codeword,
    embed_tile_bits,
    extract_image_tiles,
    extract_tile_scores,
)
from watermark_v4.observation import extract_observation
from watermark_v4.payload import bytes_to_bits, carrier_class_for_tile, phase_for_tile
from tests.commercial_quality_metrics import quality_metrics


CELL_SIZE = 16
TILE_SIZE = 128
BIT_COUNT = 64
IMAGE_ROOT = Path("img")
IMAGE_PATHS = (
    tuple(
        path
        for path in sorted(IMAGE_ROOT.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if IMAGE_ROOT.is_dir()
    else ()
)


def _tile_blocks(tile: np.ndarray) -> np.ndarray:
    return (
        tile.reshape(8, CELL_SIZE, 8, CELL_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(BIT_COUNT, CELL_SIZE, CELL_SIZE)
    )


def test_dct_basis_uses_shared_immutable_cached_storage_and_is_orthonormal() -> None:
    basis = _dct_basis(CELL_SIZE)
    second = _dct_basis(CELL_SIZE)

    assert basis is not second
    assert basis.base is second.base
    assert isinstance(basis.base, np.ndarray)
    assert not basis.flags.owndata
    assert basis.shape == (CELL_SIZE, CELL_SIZE)
    assert basis.dtype == np.float64
    assert not basis.flags.writeable
    assert not basis.base.flags.writeable
    np.testing.assert_allclose(
        basis @ basis.T,
        np.eye(CELL_SIZE),
        rtol=0.0,
        atol=1e-12,
    )


def test_dct_basis_callers_cannot_poison_cached_storage() -> None:
    basis = _dct_basis(CELL_SIZE)
    original = basis.copy()

    with pytest.raises(ValueError):
        basis.setflags(write=True)
    with pytest.raises(ValueError):
        basis[0, 0] = 999.0

    fresh = _dct_basis(CELL_SIZE)
    np.testing.assert_array_equal(fresh, original)
    assert fresh.base is basis.base


@pytest.mark.parametrize("size", (15, 17, 10**6))
def test_dct_basis_rejects_sizes_outside_fixed_cell_contract(size: int) -> None:
    with pytest.raises(ValueError, match="16"):
        _dct_basis(size)


def test_dct_basis_cache_holds_only_the_fixed_cell_basis() -> None:
    _dct_basis(CELL_SIZE)

    assert _cached_dct_basis.cache_info().maxsize == 1
    assert _cached_dct_basis.cache_info().currsize == 1


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_forward_dct_batch_matches_opencv_without_mutating_input(
    dtype: type[np.floating],
) -> None:
    blocks = np.random.default_rng(20260714).normal(
        loc=0.0,
        scale=64.0,
        size=(7, CELL_SIZE, CELL_SIZE),
    ).astype(dtype)
    original = blocks.copy()

    transformed = _forward_dct_blocks(blocks)
    expected = np.stack([cv2.dct(block.astype(np.float64)) for block in blocks])

    assert transformed.shape == blocks.shape
    assert transformed.dtype == np.dtype(np.float64)
    assert np.isfinite(transformed).all()
    assert np.max(np.abs(transformed - expected)) <= 1e-4
    np.testing.assert_array_equal(blocks, original)
    assert not np.shares_memory(transformed, blocks)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_inverse_dct_batch_matches_opencv_without_mutating_input(
    dtype: type[np.floating],
) -> None:
    coefficients = np.random.default_rng(20260715).normal(
        loc=0.0,
        scale=64.0,
        size=(7, CELL_SIZE, CELL_SIZE),
    ).astype(dtype)
    original = coefficients.copy()

    restored = _inverse_dct_blocks(coefficients)
    expected = np.stack(
        [cv2.idct(block.astype(np.float64)) for block in coefficients]
    )

    assert restored.shape == coefficients.shape
    assert restored.dtype == np.dtype(np.float64)
    assert np.isfinite(restored).all()
    assert np.max(np.abs(restored - expected)) <= 1e-4
    np.testing.assert_array_equal(coefficients, original)
    assert not np.shares_memory(restored, coefficients)


@pytest.mark.parametrize("seed", (20260714, 20260715, 20260716))
def test_float32_image_domain_batch_matches_float64_opencv_reference(
    seed: int,
) -> None:
    blocks = np.random.default_rng(seed).uniform(
        0.0,
        255.0,
        size=(64, CELL_SIZE, CELL_SIZE),
    ).astype(np.float32)

    transformed = _forward_dct_blocks(blocks)
    expected = np.stack([cv2.dct(block.astype(np.float64)) for block in blocks])

    assert transformed.dtype == np.dtype(np.float64)
    assert np.max(np.abs(transformed - expected)) <= 1e-4


@pytest.mark.parametrize("transform", (_forward_dct_blocks, _inverse_dct_blocks))
@pytest.mark.parametrize("dtype", (np.int16, np.float32, np.float64))
def test_dct_batches_convert_accepted_real_numeric_inputs_to_float64(
    transform: object,
    dtype: type[np.generic],
) -> None:
    blocks = np.ones((2, CELL_SIZE, CELL_SIZE), dtype=dtype)

    result = transform(blocks)  # type: ignore[operator]

    assert result.dtype == np.dtype(np.float64)
    assert np.isfinite(result).all()


def test_forward_inverse_round_trip_preserves_batch() -> None:
    blocks = np.random.default_rng(20260716).uniform(
        -128.0,
        127.0,
        size=(4, CELL_SIZE, CELL_SIZE),
    ).astype(np.float64)

    restored = _inverse_dct_blocks(_forward_dct_blocks(blocks))

    np.testing.assert_allclose(restored, blocks, rtol=0.0, atol=1e-10)


def test_forward_and_inverse_dct_use_compute_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class BackendSpy:
        def forward_dct(self, blocks: np.ndarray, basis: np.ndarray) -> np.ndarray:
            calls.append("forward")
            return basis @ blocks @ basis.T

        def inverse_dct(self, blocks: np.ndarray, basis: np.ndarray) -> np.ndarray:
            calls.append("inverse")
            return basis.T @ blocks @ basis

    monkeypatch.setattr(dct_module, "get_compute_backend", lambda: BackendSpy())
    blocks = np.arange(2 * 16 * 16, dtype=np.float64).reshape(2, 16, 16)

    restored = dct_module._inverse_dct_blocks(
        dct_module._forward_dct_blocks(blocks)
    )

    assert calls == ["forward", "inverse"]
    np.testing.assert_allclose(restored, blocks, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("transform", (_forward_dct_blocks, _inverse_dct_blocks))
@pytest.mark.parametrize(
    "malformed",
    (
        [[[0.0] * CELL_SIZE] * CELL_SIZE],
        np.zeros((CELL_SIZE, CELL_SIZE), dtype=np.float32),
        np.zeros((1, 8, CELL_SIZE), dtype=np.float32),
        np.zeros((1, CELL_SIZE, CELL_SIZE, 1), dtype=np.float32),
        np.zeros((1, CELL_SIZE, CELL_SIZE), dtype=np.bool_),
        np.full((1, CELL_SIZE, CELL_SIZE), "1.0"),
        np.zeros((1, CELL_SIZE, CELL_SIZE), dtype=np.complex64),
    ),
)
def test_dct_batch_rejects_malformed_types_and_shapes(
    transform: object,
    malformed: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="blocks"):
        transform(malformed)  # type: ignore[operator]


@pytest.mark.parametrize("transform", (_forward_dct_blocks, _inverse_dct_blocks))
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), float("-inf")))
def test_dct_batch_rejects_nonfinite_values(transform: object, nonfinite: float) -> None:
    blocks = np.zeros((2, CELL_SIZE, CELL_SIZE), dtype=np.float64)
    blocks[1, 3, 5] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        transform(blocks)  # type: ignore[operator]


@pytest.mark.parametrize("transform", (_forward_dct_blocks, _inverse_dct_blocks))
def test_dct_batch_rejects_nonfinite_output_without_emitting_warning(
    transform: object,
) -> None:
    blocks = np.full(
        (1, CELL_SIZE, CELL_SIZE),
        np.finfo(np.float64).max,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="finite output"):
        transform(blocks)  # type: ignore[operator]


@pytest.mark.parametrize(
    "bits",
    (
        (0,) * BIT_COUNT,
        (1,) * BIT_COUNT,
        tuple(index % 2 for index in range(BIT_COUNT)),
    ),
)
def test_embed_and_extract_recovers_zero_one_and_alternating_bits(
    bits: tuple[int, ...],
) -> None:
    config = V4Config()
    tile = np.random.default_rng(20260717).uniform(
        0.0,
        255.0,
        size=(TILE_SIZE, TILE_SIZE),
    ).astype(np.float32)
    original = tile.copy()

    embedded = embed_tile_bits(tile, bits, config)
    scores = extract_tile_scores(embedded, config)

    assert embedded.shape == tile.shape
    assert embedded.dtype == np.dtype(np.float64)
    assert np.isfinite(embedded).all()
    assert type(scores) is tuple
    assert len(scores) == BIT_COUNT
    assert all(type(score) is float and np.isfinite(score) for score in scores)
    assert tuple(int(score > 0.0) for score in scores) == bits
    np.testing.assert_array_equal(tile, original)
    assert not np.shares_memory(embedded, tile)


def test_embed_enforces_both_coefficient_pair_margins_with_row_major_mapping() -> None:
    config = V4Config()
    bits = tuple((index // 8 + index % 8) % 2 for index in range(BIT_COUNT))
    tile = np.random.default_rng(20260718).normal(
        128.0,
        45.0,
        size=(TILE_SIZE, TILE_SIZE),
    )

    embedded = embed_tile_bits(tile, bits, config)
    coefficients = _forward_dct_blocks(_tile_blocks(embedded))
    signs = np.where(np.asarray(bits) == 1, 1.0, -1.0)

    for first, second in config.coefficient_pairs:
        signed_differences = signs * (
            coefficients[:, first[0], first[1]]
            - coefficients[:, second[0], second[1]]
        )
        assert np.min(signed_differences) >= config.dct_margin - 1e-10


@pytest.mark.parametrize("operation", ("embed", "extract"))
def test_tile_carrier_centers_luminance_before_forward_dct(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[np.ndarray] = []
    original_forward = dct_module._forward_dct_blocks

    def recording_forward(blocks: np.ndarray) -> np.ndarray:
        captured.append(blocks.copy())
        return original_forward(blocks)

    monkeypatch.setattr(dct_module, "_forward_dct_blocks", recording_forward)
    tile = np.full((TILE_SIZE, TILE_SIZE), 128.0)
    if operation == "embed":
        embed_tile_bits(tile, (1,) * BIT_COUNT, V4Config())
    else:
        extract_tile_scores(tile, V4Config())

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], np.zeros_like(captured[0]))


def test_tile_embedding_adds_luminance_center_after_inverse_dct() -> None:
    embedded = embed_tile_bits(
        np.full((TILE_SIZE, TILE_SIZE), 128.0),
        (1,) * BIT_COUNT,
        V4Config(),
    )

    assert np.mean(embedded) == pytest.approx(128.0, abs=1e-12)
    centered_coefficients = _forward_dct_blocks(_tile_blocks(embedded - 128.0))
    assert np.max(np.abs(centered_coefficients[:, 0, 0])) <= 1e-10


def test_extract_scores_are_average_pair_difference_normalized_by_margin() -> None:
    config = V4Config()
    tile = np.random.default_rng(20260719).uniform(
        0.0,
        255.0,
        size=(TILE_SIZE, TILE_SIZE),
    )
    coefficients = _forward_dct_blocks(_tile_blocks(tile))
    expected_pair_differences = []
    for first, second in config.coefficient_pairs:
        expected_pair_differences.append(
            coefficients[:, first[0], first[1]]
            - coefficients[:, second[0], second[1]]
        )
    expected = np.mean(expected_pair_differences, axis=0) / config.dct_margin

    scores = extract_tile_scores(tile, config)

    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-12)


def _representative_tiles() -> tuple[np.ndarray, ...]:
    horizontal = np.linspace(0.0, 255.0, TILE_SIZE, dtype=np.float64)
    gradient = np.broadcast_to(horizontal, (TILE_SIZE, TILE_SIZE)).copy()
    return (
        np.full((TILE_SIZE, TILE_SIZE), 128.0, dtype=np.float64),
        gradient,
        np.random.default_rng(20260720).uniform(
            0.0,
            255.0,
            size=(TILE_SIZE, TILE_SIZE),
        ),
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8),
        np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8),
    )


@pytest.mark.parametrize("tile", _representative_tiles())
def test_pre_quantized_embedding_recovers_on_representative_content(
    tile: np.ndarray,
) -> None:
    config = V4Config()
    bits = tuple(index % 3 == 0 for index in range(BIT_COUNT))
    exact_bits = tuple(int(bit) for bit in bits)

    embedded = embed_tile_bits(tile, exact_bits, config)
    scores = extract_tile_scores(embedded, config)

    assert tuple(int(score > 0.0) for score in scores) == exact_bits


def test_tile_embedding_does_not_clip_image_domain_values() -> None:
    config = V4Config()

    dark = embed_tile_bits(
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float64),
        (1,) * BIT_COUNT,
        config,
    )
    bright = embed_tile_bits(
        np.full((TILE_SIZE, TILE_SIZE), 255.0),
        (0,) * BIT_COUNT,
        config,
    )

    assert np.min(dark) < 0.0
    assert np.max(bright) > 255.0


def test_tile_functions_accept_read_only_inputs() -> None:
    tile = np.full((TILE_SIZE, TILE_SIZE), 128.0)
    tile.flags.writeable = False
    bits = tuple(index % 2 for index in range(BIT_COUNT))

    embedded = embed_tile_bits(tile, bits, V4Config())
    scores = extract_tile_scores(tile, V4Config())

    assert embedded.shape == tile.shape
    assert len(scores) == BIT_COUNT


@pytest.mark.parametrize("function", (embed_tile_bits, extract_tile_scores))
@pytest.mark.parametrize(
    "malformed",
    (
        [[0.0] * TILE_SIZE] * TILE_SIZE,
        np.zeros((TILE_SIZE, TILE_SIZE, 1), dtype=np.float64),
        np.zeros((64, 64), dtype=np.float64),
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.bool_),
        np.full((TILE_SIZE, TILE_SIZE), "1.0"),
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.complex64),
        np.full((TILE_SIZE, TILE_SIZE), np.nan),
        np.full((TILE_SIZE, TILE_SIZE), np.inf),
    ),
)
def test_tile_functions_reject_malformed_tiles(
    function: object,
    malformed: object,
) -> None:
    config = V4Config()
    with pytest.raises((TypeError, ValueError), match="tile"):
        if function is embed_tile_bits:
            function(malformed, (0,) * BIT_COUNT, config)  # type: ignore[operator]
        else:
            function(malformed, config)  # type: ignore[operator]


@pytest.mark.parametrize(
    "bits",
    (
        [0] * BIT_COUNT,
        (0,) * (BIT_COUNT - 1),
        (0,) * (BIT_COUNT + 1),
        (0,) * (BIT_COUNT - 1) + (2,),
        (0,) * (BIT_COUNT - 1) + (-1,),
        (0,) * (BIT_COUNT - 1) + (True,),
        (0,) * (BIT_COUNT - 1) + (1.0,),
    ),
)
def test_embed_rejects_malformed_bits(bits: object) -> None:
    with pytest.raises((TypeError, ValueError), match="bits"):
        embed_tile_bits(
            np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float64),
            bits,  # type: ignore[arg-type]
            V4Config(),
        )


@pytest.mark.parametrize("config", (None, object()))
def test_tile_functions_require_exact_v4_config(config: object) -> None:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float64)

    with pytest.raises(TypeError, match="config"):
        embed_tile_bits(tile, (0,) * BIT_COUNT, config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        extract_tile_scores(tile, config)  # type: ignore[arg-type]


@pytest.mark.parametrize("function", (embed_tile_bits, extract_tile_scores))
def test_tile_carrier_entry_points_contain_no_python_loops(function: object) -> None:
    tree = ast.parse(inspect.getsource(function))  # type: ignore[arg-type]

    assert not any(
        isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
        for node in ast.walk(tree)
    )


CODEWORD = bytes.fromhex("00112233445566778899aabbccddeeff")


def _expected_tile_bits(record: TileScores) -> tuple[int, ...]:
    all_bits = bytes_to_bits(CODEWORD)
    start = carrier_class_for_tile(record.tile_x, record.tile_y) * BIT_COUNT
    return all_bits[start : start + BIT_COUNT]


def _rgb_image(kind: str, width: int = 256, height: int = 128) -> Image.Image:
    if kind == "constant":
        pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    elif kind == "gradient":
        horizontal = np.linspace(0, 255, width, dtype=np.uint8)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = horizontal
        pixels[..., 1] = np.flip(horizontal)
        pixels[..., 2] = 96
    else:
        pixels = np.random.default_rng(20260721).integers(
            0,
            256,
            size=(height, width, 3),
            dtype=np.uint8,
        )
    return Image.fromarray(pixels)


def _decoded_bits(scores: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(int(score > 0.0) for score in scores)


@pytest.mark.parametrize("size", ((256, 128), (128, 256)))
def test_embed_codeword_preserves_dimensions_and_input_image(size: tuple[int, int]) -> None:
    image = _rgb_image("gradient", *size)
    original = image.tobytes()

    embedded = embed_codeword(image, CODEWORD)

    assert type(embedded) is Image.Image
    assert embedded.mode == "RGB"
    assert embedded.size == size
    assert embedded.tobytes() != original
    assert image.tobytes() == original


def test_rgba_embedding_preserves_alpha_byte_for_byte() -> None:
    rgb = np.asarray(_rgb_image("noise")).copy()
    alpha = np.random.default_rng(20260722).integers(
        0,
        256,
        size=rgb.shape[:2],
        dtype=np.uint8,
    )
    rgba = Image.fromarray(np.dstack((rgb, alpha)))
    original = rgba.tobytes()

    embedded = embed_codeword(rgba, CODEWORD)

    assert embedded.mode == "RGBA"
    np.testing.assert_array_equal(np.asarray(embedded)[..., 3], alpha)
    assert rgba.tobytes() == original


def test_incomplete_right_and_bottom_edges_remain_byte_identical() -> None:
    image = _rgb_image("noise", 270, 150)
    original = np.asarray(image).copy()

    embedded = np.asarray(embed_codeword(image, CODEWORD))

    np.testing.assert_array_equal(embedded[:, 256:], original[:, 256:])
    np.testing.assert_array_equal(embedded[128:, :], original[128:, :])
    assert np.any(embedded[:128, :256] != original[:128, :256])


@pytest.mark.parametrize("kind", ("constant", "gradient", "noise"))
def test_intact_quantized_tiles_recover_exact_logical_codeword(kind: str) -> None:
    image = _rgb_image(kind)

    embedded = embed_codeword(image, CODEWORD)
    records = extract_image_tiles(embedded)

    assert type(records) is tuple
    assert len(records) == 2
    assert all(
        _decoded_bits(record.logical_scores) == _expected_tile_bits(record)
        for record in records
    )


def test_extract_image_tiles_returns_row_major_phase_aware_immutable_records() -> None:
    image = _rgb_image("gradient", 256, 256)
    records = extract_image_tiles(embed_codeword(image, CODEWORD))

    assert [(record.tile_x, record.tile_y, record.phase) for record in records] == [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 2),
        (1, 1, 3),
    ]
    assert all(isinstance(record, TileScores) for record in records)
    assert all(type(record.logical_scores) is tuple for record in records)
    assert all(
        record.phase == phase_for_tile(record.tile_x, record.tile_y)
        for record in records
    )
    with pytest.raises(FrozenInstanceError):
        records[0].phase = 2  # type: ignore[misc]
    assert not hasattr(records[0], "__dict__")


def test_tile_scores_rejects_malformed_fields() -> None:
    valid_scores = (0.0,) * BIT_COUNT
    with pytest.raises(TypeError, match="tile"):
        TileScores(True, 0, 0, valid_scores)
    with pytest.raises(ValueError, match="tile"):
        TileScores(-1, 0, 0, valid_scores)
    with pytest.raises((TypeError, ValueError), match="phase"):
        TileScores(0, 0, 4, valid_scores)
    with pytest.raises(TypeError, match="scores"):
        TileScores(0, 0, 0, [0.0] * BIT_COUNT)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scores"):
        TileScores(0, 0, 0, (0.0,) * 63)
    with pytest.raises(ValueError, match="finite"):
        TileScores(0, 0, 0, (0.0,) * 63 + (float("nan"),))


@pytest.mark.parametrize("mode", ("L", "P", "CMYK"))
def test_image_carrier_rejects_unsupported_modes(mode: str) -> None:
    image = Image.new(mode, (256, 128))

    with pytest.raises(ValueError, match="RGB"):
        embed_codeword(image, CODEWORD)
    with pytest.raises(ValueError, match="RGB"):
        extract_image_tiles(image)


@pytest.mark.parametrize("image", (None, np.zeros((128, 256, 3), dtype=np.uint8)))
def test_image_carrier_requires_exact_pil_image(image: object) -> None:
    with pytest.raises(TypeError, match="image"):
        embed_codeword(image, CODEWORD)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="image"):
        extract_image_tiles(image)  # type: ignore[arg-type]


@pytest.mark.parametrize("codeword", (bytes(15), bytes(17), bytearray(16), None))
def test_embed_codeword_rejects_malformed_codeword(codeword: object) -> None:
    with pytest.raises((TypeError, ValueError), match="codeword"):
        embed_codeword(_rgb_image("constant"), codeword)  # type: ignore[arg-type]


@pytest.mark.parametrize("config", (None, object()))
def test_image_carrier_requires_exact_v4_config(config: object) -> None:
    image = _rgb_image("constant")
    with pytest.raises(TypeError, match="config"):
        embed_codeword(image, CODEWORD, config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        extract_image_tiles(image, config)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", ((127, 128), (128, 128), (255, 128)))
def test_image_carrier_rejects_images_without_minimum_complete_tiles(
    size: tuple[int, int],
) -> None:
    image = Image.new("RGB", size, (128, 128, 128))

    with pytest.raises(ValueError, match="tiles"):
        embed_codeword(image, CODEWORD)
    with pytest.raises(ValueError, match="tiles"):
        extract_image_tiles(image)


def test_image_carrier_rejects_insufficient_distinct_phases() -> None:
    image = _rgb_image("constant", 256, 128)
    config = V4Config(minimum_phases=3)

    with pytest.raises(ValueError, match="phases"):
        embed_codeword(image, CODEWORD, config)
    with pytest.raises(ValueError, match="phases"):
        extract_image_tiles(image, config)


def test_package_exports_stable_dct_image_api() -> None:
    assert watermark_v4.TileScores is TileScores
    assert watermark_v4.embed_codeword is embed_codeword
    assert watermark_v4.extract_image_tiles is extract_image_tiles


def test_multi_tile_embed_uses_one_transform_batch_and_one_color_conversion_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"forward": 0, "inverse": 0, "rgb_to_ycrcb": 0, "ycrcb_to_rgb": 0}
    original_forward = dct_module._forward_dct_blocks
    original_inverse = dct_module._inverse_dct_blocks
    original_cvt_color = dct_module.cv2.cvtColor

    def counted_forward(blocks: np.ndarray) -> np.ndarray:
        counts["forward"] += 1
        return original_forward(blocks)

    def counted_inverse(blocks: np.ndarray) -> np.ndarray:
        counts["inverse"] += 1
        return original_inverse(blocks)

    def counted_cvt_color(array: np.ndarray, code: int) -> np.ndarray:
        if code == cv2.COLOR_RGB2YCrCb:
            counts["rgb_to_ycrcb"] += 1
        if code == cv2.COLOR_YCrCb2RGB:
            counts["ycrcb_to_rgb"] += 1
        return original_cvt_color(array, code)

    monkeypatch.setattr(dct_module, "_forward_dct_blocks", counted_forward)
    monkeypatch.setattr(dct_module, "_inverse_dct_blocks", counted_inverse)
    monkeypatch.setattr(dct_module.cv2, "cvtColor", counted_cvt_color)

    embed_codeword(_rgb_image("noise", 256, 256), CODEWORD)

    assert counts == {
        "forward": 1,
        "inverse": 1,
        "rgb_to_ycrcb": 1,
        "ycrcb_to_rgb": 1,
    }


def test_large_embed_codeword_uses_bounded_tile_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _rgb_image("gradient", 1152, 1152)
    conversion_rows = []
    original_cvt_color = dct_module.cv2.cvtColor

    def forbidden_full_gather(*args: object, **kwargs: object) -> object:
        raise AssertionError("embed_codeword must not gather every image tile")

    def tracked_cvt_color(values: np.ndarray, code: int) -> np.ndarray:
        conversion_rows.append(values.shape[0])
        return original_cvt_color(values, code)

    monkeypatch.setattr(dct_module, "_gather_centered_blocks", forbidden_full_gather)
    monkeypatch.setattr(dct_module.cv2, "cvtColor", tracked_cvt_color)

    embedded = embed_codeword(image, CODEWORD)

    assert embedded.size == image.size
    assert embedded.tobytes() != image.tobytes()
    assert len(conversion_rows) > 2
    assert max(conversion_rows) <= dct_module.DCT_TILE_BATCH * TILE_SIZE


def test_multi_tile_extract_uses_one_forward_batch_and_one_color_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"forward": 0, "inverse": 0, "rgb_to_ycrcb": 0, "ycrcb_to_rgb": 0}
    original_forward = dct_module._forward_dct_blocks
    original_inverse = dct_module._inverse_dct_blocks
    original_cvt_color = dct_module.cv2.cvtColor

    def counted_forward(blocks: np.ndarray) -> np.ndarray:
        counts["forward"] += 1
        return original_forward(blocks)

    def counted_inverse(blocks: np.ndarray) -> np.ndarray:
        counts["inverse"] += 1
        return original_inverse(blocks)

    def counted_cvt_color(array: np.ndarray, code: int) -> np.ndarray:
        if code == cv2.COLOR_RGB2YCrCb:
            counts["rgb_to_ycrcb"] += 1
        if code == cv2.COLOR_YCrCb2RGB:
            counts["ycrcb_to_rgb"] += 1
        return original_cvt_color(array, code)

    monkeypatch.setattr(dct_module, "_forward_dct_blocks", counted_forward)
    monkeypatch.setattr(dct_module, "_inverse_dct_blocks", counted_inverse)
    monkeypatch.setattr(dct_module.cv2, "cvtColor", counted_cvt_color)

    records = extract_image_tiles(_rgb_image("gradient", 256, 256))

    assert len(records) == 4
    assert counts == {
        "forward": 1,
        "inverse": 0,
        "rgb_to_ycrcb": 1,
        "ycrcb_to_rgb": 0,
    }


@pytest.mark.parametrize("margin", (4.0, 6.0, 8.0))
@pytest.mark.parametrize("image_path", IMAGE_PATHS)
def test_representative_dct_quality_and_default_margin_recovery_gate(
    image_path: Path,
    margin: float,
    record_property: object,
) -> None:
    with Image.open(image_path) as loaded:
        original = loaded.convert("RGB")
    config = V4Config(dct_margin=margin)

    embedded = embed_codeword(original, CODEWORD, config)
    metrics = quality_metrics(original, embedded)
    record_property(  # type: ignore[operator]
        f"{Path(image_path).name}-margin-{margin:g}",
        str(metrics),
    )

    assert np.isfinite(float(metrics["psnr"]))
    assert np.isfinite(float(metrics["ssim"]))
    if margin == 6.0:
        assert float(metrics["psnr"]) >= 38.0
        assert float(metrics["ssim"]) >= 0.95

        records = extract_image_tiles(embedded, config)
        observation = extract_observation(records)
        assert observation is not None
        exact_tile_fraction = np.mean(
            [
                _decoded_bits(record.logical_scores) == _expected_tile_bits(record)
                for record in records
            ]
        )

        assert observation.observed_codeword == CODEWORD
        assert exact_tile_fraction >= 0.90


def test_1024_synthetic_extraction_has_bounded_regression_runtime(
    record_property: object,
) -> None:
    horizontal = np.linspace(0, 255, 1024, dtype=np.uint8)
    pixels = np.empty((1024, 1024, 3), dtype=np.uint8)
    pixels[..., 0] = horizontal
    pixels[..., 1] = horizontal[:, None]
    pixels[..., 2] = 127
    image = Image.fromarray(pixels)
    _dct_basis(CELL_SIZE)

    started = perf_counter()
    records = extract_image_tiles(image)
    elapsed = perf_counter() - started
    record_property(  # type: ignore[operator]
        "1024-extraction-regression-seconds",
        f"{elapsed:.6f}",
    )

    assert len(records) == (1024 // TILE_SIZE) ** 2
    assert elapsed < 2.0  # Regression ceiling, not the commercial SLA.


def test_image_carrier_never_calls_scalar_opencv_dct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("scalar OpenCV DCT must not be called")

    monkeypatch.setattr(cv2, "dct", forbidden)
    monkeypatch.setattr(cv2, "idct", forbidden)
    image = _rgb_image("gradient", 256, 128)

    embedded = embed_codeword(image, CODEWORD)
    records = extract_image_tiles(embedded)

    assert len(records) == 2
    assert all(
        _decoded_bits(record.logical_scores) == _expected_tile_bits(record)
        for record in records
    )
