from __future__ import annotations

import numpy as np
import pytest

from trace_app.v4.region_protection import (
    ProtectedRegion,
    detect_protected_regions,
    reinforced_tiles,
)


def test_protected_region_validates_kind_and_bounds() -> None:
    region = ProtectedRegion("face", 10, 20, 80, 100, 0.9)

    assert region.kind == "face"
    with pytest.raises(ValueError, match="kind"):
        ProtectedRegion("person", 10, 20, 80, 100, 0.9)
    with pytest.raises(ValueError, match="bounds"):
        ProtectedRegion("face", 10, 20, 10, 100, 0.9)


def test_reinforced_tiles_selects_only_complete_intersecting_tiles() -> None:
    regions = (
        ProtectedRegion("face", 120, 120, 150, 150, 0.9),
        ProtectedRegion("foot_shoe", 300, 300, 390, 390, 0.8),
    )

    assert reinforced_tiles(
        regions,
        image_width=400,
        image_height=400,
        tile_size=128,
    ) == frozenset({(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)})


def test_reinforced_tiles_ignores_incomplete_edge_only_region() -> None:
    regions = (ProtectedRegion("face", 390, 390, 400, 400, 0.9),)

    assert reinforced_tiles(
        regions,
        image_width=400,
        image_height=400,
        tile_size=128,
    ) == frozenset()


def test_builtin_region_detectors_accept_rgb_image() -> None:
    image = np.full((256, 256, 3), 127, dtype=np.uint8)

    assert detect_protected_regions(image) == ()


def test_region_detector_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        detect_protected_regions(np.zeros((256, 256), dtype=np.uint8))
