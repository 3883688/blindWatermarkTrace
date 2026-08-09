from __future__ import annotations

import numpy as np
import pytest

import trace_app.v4.region_protection as region_protection
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


def test_region_detector_accepts_rgb_image(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    monkeypatch.setattr(region_protection, "_detect_with_pose_models", lambda image: ())

    assert detect_protected_regions(image) == ()


def test_person_model_boxes_are_rescaled_to_source_image() -> None:
    class Input:
        name = "images"

    class Session:
        def get_inputs(self) -> list[Input]:
            return [Input()]

        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert inputs["images"].shape == (1, 3, 416, 416)
            return [
                np.asarray([[[41.6, 20.8, 208.0, 374.4, 0.9]]], dtype=np.float32),
                np.asarray([[0]], dtype=np.int64),
            ]

    people = region_protection._detect_people_onnx(
        np.zeros((1000, 500, 3), dtype=np.uint8), Session()
    )

    assert people[0][0] == pytest.approx((100, 50, 500, 900))
    assert people[0][1] == pytest.approx(0.9)


def test_wholebody_keypoints_create_face_and_foot_regions() -> None:
    points = np.zeros((133, 2), dtype=np.float32)
    scores = np.zeros(133, dtype=np.float32)
    points[23:91] = np.asarray([120, 90])
    points[23] = (100, 70)
    points[24] = (140, 110)
    scores[23:91] = 0.8
    points[[15, 17, 18, 19]] = ((80, 350), (70, 370), (90, 380), (110, 375))
    points[[16, 20, 21, 22]] = ((200, 350), (190, 370), (210, 380), (230, 375))
    scores[[15, 17, 18, 19, 16, 20, 21, 22]] = 0.7

    regions = region_protection._regions_from_keypoints(
        points, scores, (40, 20, 280, 400), 0.9, 320, 420
    )

    assert [region.kind for region in regions] == ["face", "foot_shoe", "foot_shoe"]
    assert [region.confidence for region in regions] == pytest.approx([0.8, 0.7, 0.7])
    assert regions[0].left < 100 and regions[0].right > 140


def test_model_failure_uses_builtin_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (ProtectedRegion("face", 1, 2, 3, 4, 0.5),)
    monkeypatch.setattr(
        region_protection,
        "_detect_with_pose_models",
        lambda image: (_ for _ in ()).throw(RuntimeError("broken model")),
    )
    monkeypatch.setattr(
        region_protection, "_detect_with_builtin_detectors", lambda image: expected
    )

    assert detect_protected_regions(np.zeros((8, 8, 3), dtype=np.uint8)) == expected


def test_region_detector_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        detect_protected_regions(np.zeros((256, 256), dtype=np.uint8))
