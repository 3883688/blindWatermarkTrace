from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from PIL import Image

from watermark_v4 import V4Config
from watermark_v4.dct import embed_codeword, extract_image_tiles
from watermark_v4.observation import extract_observation
from watermark_v4.payload import (
    carrier_class_for_tile,
    permute_codeword_bits,
)


CODEWORD = bytes.fromhex("00112233445566778899aabbccddeeff")


def _bits(value: bytes) -> tuple[int, ...]:
    return tuple((byte >> shift) & 1 for byte in value for shift in range(7, -1, -1))


def test_checkerboard_neighbors_select_opposite_carrier_classes() -> None:
    assert [[carrier_class_for_tile(x, y) for x in range(3)] for y in range(2)] == [
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_a_and_b_tiles_carry_separate_codeword_halves() -> None:
    for phase in range(4):
        a_physical = permute_codeword_bits(CODEWORD, phase, 0)
        b_physical = permute_codeword_bits(CODEWORD, phase, 1)
        assert sorted(a_physical) == sorted(_bits(CODEWORD[:8]))
        assert sorted(b_physical) == sorted(_bits(CODEWORD[8:]))
        assert a_physical != b_physical


def test_image_tiles_recover_their_class_half_and_both_pair_margins() -> None:
    pixels = np.random.default_rng(20260729).integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )
    image = Image.fromarray(pixels)
    embedded = embed_codeword(image, CODEWORD, V4Config())
    tiles = extract_image_tiles(embedded, V4Config())

    assert len(tiles) == 4
    for tile in tiles:
        expected = _bits(
            CODEWORD[:8] if carrier_class_for_tile(tile.tile_x, tile.tile_y) == 0 else CODEWORD[8:]
        )
        assert tuple(int(score > 0) for score in tile.logical_scores) == expected


def test_observation_is_immutable_and_requires_both_classes_and_phases() -> None:
    image = Image.fromarray(
        np.random.default_rng(8).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    )
    tiles = extract_image_tiles(embed_codeword(image, CODEWORD), V4Config())
    observation = extract_observation(
        tiles,
        minimum_tiles_per_class=1,
        minimum_phases=2,
    )
    assert observation is not None
    assert observation.observed_codeword == CODEWORD
    assert len(observation.byte_confidences) == 16
    assert observation.tile_counts == (2, 2)
    assert observation.phase_counts == (2, 2)
    assert type(observation.class_evidence) is tuple
    with pytest.raises(FrozenInstanceError):
        observation.elapsed_seconds = 0.0  # type: ignore[misc]

    only_a = tuple(
        tile for tile in tiles if carrier_class_for_tile(tile.tile_x, tile.tile_y) == 0
    )
    assert extract_observation(only_a) is None
    assert extract_observation(
        tiles,
        minimum_phases=5,
    ) is None
    assert extract_observation(
        tiles,
        coverages=(0.5,) * len(tiles),
        minimum_coverage=0.70,
    ) is None
    assert extract_observation(
        tiles,
        coverages=(1.0, 1.0, 0.0, 0.0),
        minimum_coverage=0.70,
        minimum_phases=3,
    ) is None
