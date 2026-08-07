from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

import watermark_v4.features as features_module
from watermark_v4.features import (
    FEATURE_INDEX_MAX_DESCRIPTORS,
    FEATURE_INDEX_SCHEMA_VERSION,
    FeatureIndex,
    FeatureMatch,
    extract_feature_index,
    load_feature_index,
    match_feature_indexes,
    rank_feature_candidates,
    save_feature_index,
)
from watermark_v4 import V4Config


def _feature_image(size: tuple[int, int] = (900, 600), seed: int = 17) -> Image.Image:
    width, height = size
    rng = np.random.default_rng(seed)
    pixels = rng.integers(30, 225, size=(height, width, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (0, 0), sigmaX=1.0)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for index in range(24):
        x = 20 + (index * 83) % (width - 90)
        y = 20 + (index * 47) % (height - 70)
        draw.rectangle((x, y, x + 52, y + 31), outline="white", width=3)
        draw.line((x, y + 31, x + 52, y), fill="black", width=2)
    return image


def test_extract_feature_index_has_versioned_bounded_original_coordinate_data() -> None:
    image = _feature_image()

    index = extract_feature_index(image)

    assert type(index) is FeatureIndex
    assert index.schema_version == FEATURE_INDEX_SCHEMA_VERSION == 4
    assert index.opencv_version == cv2.__version__
    assert (index.image_width, index.image_height) == image.size
    assert index.analysis_scale == pytest.approx(640 / 900)
    assert index.keypoints.dtype == np.float32
    assert index.keypoints.ndim == 2 and index.keypoints.shape[1] == 2
    assert index.descriptors.dtype == np.uint8
    assert index.descriptors.shape == (len(index.keypoints), 32)
    assert 12 <= len(index.descriptors) <= FEATURE_INDEX_MAX_DESCRIPTORS
    assert np.all(index.keypoints[:, 0] >= 0.0)
    assert np.all(index.keypoints[:, 0] < image.width)
    assert np.all(index.keypoints[:, 1] >= 0.0)
    assert np.all(index.keypoints[:, 1] < image.height)
    assert index.thumbnail.shape == (32, 32)
    assert index.thumbnail.dtype == np.uint8
    assert not index.keypoints.flags.writeable
    assert not index.descriptors.flags.writeable
    assert not index.thumbnail.flags.writeable


def test_feature_index_is_frozen_slotted_and_arrays_cannot_be_poisoned() -> None:
    index = extract_feature_index(_feature_image((640, 480)))

    assert not hasattr(index, "__dict__")
    with pytest.raises(FrozenInstanceError):
        index.image_width = 10  # type: ignore[misc]
    with pytest.raises(ValueError):
        index.descriptors[0, 0] ^= 1


def test_feature_index_round_trips_without_pickle(tmp_path: Path) -> None:
    index = extract_feature_index(_feature_image())
    path = tmp_path / "record-v4.npz"

    save_feature_index(path, index)
    restored = load_feature_index(path)

    assert type(restored) is FeatureIndex
    assert restored.schema_version == index.schema_version
    assert restored.opencv_version == index.opencv_version
    assert restored.image_width == index.image_width
    assert restored.image_height == index.image_height
    assert restored.analysis_scale == index.analysis_scale
    np.testing.assert_array_equal(restored.keypoints, index.keypoints)
    np.testing.assert_array_equal(restored.descriptors, index.descriptors)
    np.testing.assert_array_equal(restored.thumbnail, index.thumbnail)


@pytest.mark.parametrize("kind", ("missing", "malformed", "wrong-schema"))
def test_load_feature_index_rejects_invalid_indexes(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "bad.npz"
    if kind == "malformed":
        path.write_bytes(b"not an npz")
    elif kind == "wrong-schema":
        np.savez_compressed(
            path,
            schema_version=np.asarray(3, dtype=np.int64),
            opencv_version=np.asarray(cv2.__version__),
            image_width=np.asarray(640, dtype=np.int64),
            image_height=np.asarray(480, dtype=np.int64),
            analysis_scale=np.asarray(1.0, dtype=np.float64),
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 32), dtype=np.uint8),
            thumbnail=np.zeros((32, 32), dtype=np.uint8),
        )

    assert load_feature_index(path) is None


def test_reusing_extracted_query_index_does_not_repeat_opencv_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"color": 0, "orb": 0}
    original_color = features_module.cv2.cvtColor
    original_create = features_module.cv2.ORB_create

    def counted_color(*args: object, **kwargs: object) -> np.ndarray:
        calls["color"] += 1
        return original_color(*args, **kwargs)

    class CountedOrb:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def detectAndCompute(self, *args: object, **kwargs: object) -> object:
            calls["orb"] += 1
            return self.wrapped.detectAndCompute(*args, **kwargs)  # type: ignore[attr-defined]

    def counted_create(*args: object, **kwargs: object) -> CountedOrb:
        return CountedOrb(original_create(*args, **kwargs))

    monkeypatch.setattr(features_module.cv2, "cvtColor", counted_color)
    monkeypatch.setattr(features_module.cv2, "ORB_create", counted_create)

    query = extract_feature_index(_feature_image())
    candidates = [
        extract_feature_index(_feature_image(seed=seed))
        for seed in (31, 32, 33)
    ]

    assert len(query.descriptors) > 0
    assert all(len(candidate.descriptors) > 0 for candidate in candidates)
    assert calls == {"color": 4, "orb": 4}
    for candidate in candidates:
        _ = (query.descriptors, candidate.descriptors)
    assert calls == {"color": 4, "orb": 4}


def test_orb_ransac_recovers_query_to_registered_original_coordinates() -> None:
    target_image = _feature_image()
    query_image = target_image.crop((100, 80, 800, 520)).resize(
        (840, 528),
        Image.Resampling.BICUBIC,
    )
    query = extract_feature_index(query_image)
    target = extract_feature_index(target_image)

    match = match_feature_indexes(query, target)

    assert type(match) is FeatureMatch
    assert match.good_matches >= 18
    assert match.inliers >= 18
    assert match.inlier_ratio >= 0.32
    assert match.median_reprojection_error <= 5.0
    points = np.asarray([[[0.0, 0.0]], [[840.0, 528.0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, match.query_to_target).reshape(-1, 2)
    np.testing.assert_allclose(mapped[0], (100.0, 80.0), atol=8.0)
    np.testing.assert_allclose(mapped[1], (800.0, 520.0), atol=8.0)
    assert not match.query_to_target.flags.writeable


def test_spatially_balanced_index_recovers_thirty_percent_center_crop() -> None:
    target_image = _feature_image((1280, 960), seed=707)
    query_image = target_image.crop((448, 336, 832, 624))

    match = match_feature_indexes(
        extract_feature_index(query_image),
        extract_feature_index(target_image),
    )

    assert type(match) is FeatureMatch
    assert match.good_matches >= 18
    assert match.inliers >= 18
    points = np.asarray([[[0.0, 0.0]], [[384.0, 288.0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, match.query_to_target).reshape(-1, 2)
    np.testing.assert_allclose(mapped[0], (448.0, 336.0), atol=10.0)
    np.testing.assert_allclose(mapped[1], (832.0, 624.0), atol=10.0)


def test_orb_ransac_rejects_unrelated_and_blank_images() -> None:
    target = extract_feature_index(_feature_image(seed=17))
    unrelated = extract_feature_index(_feature_image(seed=991))
    blank = extract_feature_index(Image.new("RGB", (640, 480), "white"))

    assert match_feature_indexes(unrelated, target) is None
    assert match_feature_indexes(blank, target) is None


def test_orb_ransac_rejects_singular_homography(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = extract_feature_index(_feature_image())
    target = extract_feature_index(_feature_image())
    monkeypatch.setattr(
        features_module.cv2,
        "estimateAffinePartial2D",
        lambda *args, **kwargs: (
            np.zeros((2, 3), dtype=np.float64),
            np.ones((len(query.descriptors), 1), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        features_module.cv2,
        "findHomography",
        lambda *args, **kwargs: (
            np.zeros((3, 3), dtype=np.float64),
            np.ones((len(query.descriptors), 1), dtype=np.uint8),
        ),
    )

    assert match_feature_indexes(query, target) is None


def test_candidate_ranking_is_bounded_to_two_feature_hits_and_one_recent_reserve() -> None:
    target_image = _feature_image()
    query = extract_feature_index(
        target_image.crop((90, 70, 810, 530)).resize(
            (720, 460),
            Image.Resampling.BICUBIC,
        )
    )
    candidates = (
        ("target", extract_feature_index(target_image)),
        ("related", extract_feature_index(target_image.rotate(2.0))),
        ("unrelated", extract_feature_index(_feature_image(seed=400))),
        ("recent", extract_feature_index(_feature_image(seed=401))),
    )

    ranked = rank_feature_candidates(
        query,
        candidates,
        recent_record_ids=("recent",),
        config=V4Config(),
    )

    assert len(ranked) == 3
    assert {item.record_id for item in ranked[:2]} == {"target", "related"}
    assert ranked[2].record_id == "recent"
    assert ranked[2].reserved is True
    assert all(item.record_id != "unrelated" for item in ranked)


def test_candidate_ranking_searches_the_full_stored_candidate_index() -> None:
    rng = np.random.default_rng(20260714)
    query_descriptors = rng.integers(0, 256, size=(256, 32), dtype=np.uint8)

    def index(descriptors: np.ndarray) -> FeatureIndex:
        count = len(descriptors)
        return FeatureIndex(
            schema_version=FEATURE_INDEX_SCHEMA_VERSION,
            opencv_version=cv2.__version__,
            image_width=640,
            image_height=480,
            analysis_scale=1.0,
            keypoints=np.column_stack(
                (
                    np.arange(count, dtype=np.float32) % 640,
                    np.arange(count, dtype=np.float32) % 480,
                )
            ).astype(np.float32),
            descriptors=np.asarray(descriptors, dtype=np.uint8),
            thumbnail=np.zeros((32, 32), dtype=np.uint8),
        )

    unrelated = tuple(
        (
            f"decoy-{number}",
            index(rng.integers(0, 256, size=(512, 32), dtype=np.uint8)),
        )
        for number in range(3)
    )
    target_descriptors = np.vstack(
        (
            rng.integers(0, 256, size=(256, 32), dtype=np.uint8),
            query_descriptors,
        )
    )

    ranked = rank_feature_candidates(
        index(query_descriptors),
        (*unrelated, ("z-target", index(target_descriptors))),
        recent_record_ids=("decoy-2",),
        config=V4Config(),
    )

    assert "z-target" in {item.record_id for item in ranked}
