from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import V4Config


FEATURE_INDEX_SCHEMA_VERSION = 4
FEATURE_INDEX_MAX_SIDE = 640
FEATURE_INDEX_MAX_DESCRIPTORS = 3072
FEATURE_COARSE_DESCRIPTORS = 256
FEATURE_DETECTION_MAX_DESCRIPTORS = 4096
FEATURE_DESCRIPTOR_BYTES = 32
FEATURE_THUMBNAIL_SIZE = 32
FEATURE_INDEX_MAX_FILE_BYTES = 8 * 1024 * 1024
FEATURE_SELECTION_GRID = 8


@dataclass(frozen=True, slots=True)
class FeatureIndex:
    schema_version: int
    opencv_version: str
    image_width: int
    image_height: int
    analysis_scale: float
    keypoints: np.ndarray
    descriptors: np.ndarray
    thumbnail: np.ndarray

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("feature schema version must be an integer")
        if self.schema_version != FEATURE_INDEX_SCHEMA_VERSION:
            raise ValueError("feature schema version is incompatible")
        if type(self.opencv_version) is not str or not self.opencv_version:
            raise TypeError("OpenCV version must be a nonempty string")
        if type(self.image_width) is not int or type(self.image_height) is not int:
            raise TypeError("feature image dimensions must be integers")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("feature image dimensions must be positive")
        if type(self.analysis_scale) is not float or not np.isfinite(self.analysis_scale):
            raise TypeError("feature analysis scale must be a finite float")
        if not 0.0 < self.analysis_scale <= 1.0:
            raise ValueError("feature analysis scale must be between zero and one")

        keypoints = np.asarray(self.keypoints)
        descriptors = np.asarray(self.descriptors)
        thumbnail = np.asarray(self.thumbnail)
        if keypoints.dtype != np.float32 or keypoints.ndim != 2 or keypoints.shape[1:] != (2,):
            raise ValueError("feature keypoints must have shape (n, 2) float32")
        if (
            descriptors.dtype != np.uint8
            or descriptors.ndim != 2
            or descriptors.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        ):
            raise ValueError("feature descriptors must have shape (n, 32) uint8")
        if len(keypoints) != len(descriptors) or len(keypoints) > FEATURE_INDEX_MAX_DESCRIPTORS:
            raise ValueError("feature keypoints and descriptors have incompatible counts")
        if thumbnail.dtype != np.uint8 or thumbnail.shape != (
            FEATURE_THUMBNAIL_SIZE,
            FEATURE_THUMBNAIL_SIZE,
        ):
            raise ValueError("feature thumbnail must have shape (32, 32) uint8")
        if not np.isfinite(keypoints).all():
            raise ValueError("feature keypoints must be finite")
        if len(keypoints) and (
            np.any(keypoints[:, 0] < 0.0)
            or np.any(keypoints[:, 0] >= self.image_width)
            or np.any(keypoints[:, 1] < 0.0)
            or np.any(keypoints[:, 1] >= self.image_height)
        ):
            raise ValueError("feature keypoints must be within original image bounds")

        object.__setattr__(self, "keypoints", _readonly_copy(keypoints, np.float32))
        object.__setattr__(self, "descriptors", _readonly_copy(descriptors, np.uint8))
        object.__setattr__(self, "thumbnail", _readonly_copy(thumbnail, np.uint8))


@dataclass(frozen=True, slots=True)
class FeatureMatch:
    query_to_target: np.ndarray
    good_matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.query_to_target)
        if matrix.shape != (3, 3) or matrix.dtype.kind not in "f":
            raise ValueError("feature homography must be a 3x3 floating matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("feature homography must be finite")
        for name, value in (("good matches", self.good_matches), ("inliers", self.inliers)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.inliers > self.good_matches:
            raise ValueError("feature inliers cannot exceed good matches")
        for name, value in (
            ("inlier ratio", self.inlier_ratio),
            ("reprojection error", self.median_reprojection_error),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.inlier_ratio > 1.0:
            raise ValueError("inlier ratio must not exceed one")
        object.__setattr__(self, "query_to_target", _readonly_copy(matrix, np.float64))


@dataclass(frozen=True, slots=True)
class RankedFeatureCandidate:
    record_id: str
    index: FeatureIndex
    match_count: int
    match_quality: float
    thumbnail_distance: float
    reserved: bool


def extract_feature_index(image: Image.Image) -> FeatureIndex:
    if type(image) is not Image.Image:
        raise TypeError("feature image must be an exact PIL Image")
    rgb = image.convert("RGB")
    scale = min(1.0, FEATURE_INDEX_MAX_SIDE / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
    grayscale = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(
        nfeatures=FEATURE_DETECTION_MAX_DESCRIPTORS,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=7,
    )
    cv_keypoints, cv_descriptors = orb.detectAndCompute(grayscale, None)
    if cv_descriptors is None or not cv_keypoints:
        keypoints = np.empty((0, 2), dtype=np.float32)
        descriptors = np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    else:
        selected = _spatially_balanced_keypoint_indices(
            cv_keypoints,
            grayscale.shape[1],
            grayscale.shape[0],
        )
        count = min(len(selected), len(cv_descriptors), FEATURE_INDEX_MAX_DESCRIPTORS)
        selected = selected[:count]
        keypoints = np.asarray(
            [cv_keypoints[index].pt for index in selected],
            dtype=np.float32,
        )
        keypoints /= np.float32(scale)
        keypoints[:, 0] = np.minimum(keypoints[:, 0], image.width - 1e-4)
        keypoints[:, 1] = np.minimum(keypoints[:, 1], image.height - 1e-4)
        descriptors = np.ascontiguousarray(cv_descriptors[selected], dtype=np.uint8)
    thumbnail = cv2.resize(
        grayscale,
        (FEATURE_THUMBNAIL_SIZE, FEATURE_THUMBNAIL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return FeatureIndex(
        schema_version=FEATURE_INDEX_SCHEMA_VERSION,
        opencv_version=cv2.__version__,
        image_width=image.width,
        image_height=image.height,
        analysis_scale=float(scale),
        keypoints=keypoints,
        descriptors=descriptors,
        thumbnail=np.asarray(thumbnail, dtype=np.uint8),
    )


def save_feature_index(path: Path, index: FeatureIndex) -> None:
    if not isinstance(path, Path):
        raise TypeError("feature index path must be a Path")
    if type(index) is not FeatureIndex:
        raise TypeError("feature index must be an exact FeatureIndex")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(index.schema_version, dtype=np.int64),
        opencv_version=np.asarray(index.opencv_version),
        image_width=np.asarray(index.image_width, dtype=np.int64),
        image_height=np.asarray(index.image_height, dtype=np.int64),
        analysis_scale=np.asarray(index.analysis_scale, dtype=np.float64),
        keypoints=index.keypoints,
        descriptors=index.descriptors,
        thumbnail=index.thumbnail,
    )


def load_feature_index(path: Path) -> FeatureIndex | None:
    if not isinstance(path, Path):
        raise TypeError("feature index path must be a Path")
    try:
        if not path.is_file() or path.stat().st_size > FEATURE_INDEX_MAX_FILE_BYTES:
            return None
        with np.load(path, allow_pickle=False) as payload:
            return FeatureIndex(
                schema_version=int(payload["schema_version"].item()),
                opencv_version=str(payload["opencv_version"].item()),
                image_width=int(payload["image_width"].item()),
                image_height=int(payload["image_height"].item()),
                analysis_scale=float(payload["analysis_scale"].item()),
                keypoints=np.asarray(payload["keypoints"]),
                descriptors=np.asarray(payload["descriptors"]),
                thumbnail=np.asarray(payload["thumbnail"]),
            )
    except (OSError, KeyError, TypeError, ValueError, EOFError):
        return None


def match_feature_indexes(
    query: FeatureIndex,
    target: FeatureIndex,
) -> FeatureMatch | None:
    _validate_feature_pair(query, target)
    good = _good_descriptor_matches(query.descriptors, target.descriptors)
    if len(good) < 18:
        return None
    query_points = np.asarray(
        [query.keypoints[item.queryIdx] for item in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    target_points = np.asarray(
        [target.keypoints[item.trainIdx] for item in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    affine, affine_mask = cv2.estimateAffinePartial2D(
        query_points.reshape(-1, 2),
        target_points.reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is not None and affine_mask is not None:
        affine_matrix = np.vstack(
            (np.asarray(affine, dtype=np.float64), np.asarray([0.0, 0.0, 1.0]))
        )
        affine_match = _feature_match_from_geometry(
            affine_matrix,
            affine_mask,
            good,
            query_points,
            target_points,
            query,
            target,
        )
        if affine_match is not None:
            return affine_match

    homography, mask = cv2.findHomography(
        query_points,
        target_points,
        cv2.RANSAC,
        5.0,
    )
    return _feature_match_from_geometry(
        homography,
        mask,
        good,
        query_points,
        target_points,
        query,
        target,
    )


def match_feature_indexes_constrained(
    query: FeatureIndex,
    target: FeatureIndex,
    *,
    rotation_degrees: float,
    scale: float,
    tile_size: int,
    tile_offset: tuple[int, int] | None = None,
) -> FeatureMatch | None:
    _validate_feature_pair(query, target)
    if type(rotation_degrees) not in (int, float) or not np.isfinite(rotation_degrees):
        raise TypeError("constrained rotation must be finite")
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("constrained scale must be finite and positive")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("constrained tile size must be a positive integer")
    if tile_offset is not None and (
        type(tile_offset) is not tuple
        or len(tile_offset) != 2
        or any(type(value) is not int for value in tile_offset)
    ):
        raise ValueError("tile offset must be an integer pair")

    good = _good_descriptor_matches(query.descriptors, target.descriptors)
    if len(good) < 3:
        return None
    radians = math.radians(float(rotation_degrees))
    inverse_scale = 1.0 / float(scale)
    linear = inverse_scale * np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [query.keypoints[item.queryIdx] for item in good],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [target.keypoints[item.trainIdx] for item in good],
        dtype=np.float64,
    )
    translations = target_points - query_points @ linear.T
    seed_index = max(
        range(len(translations)),
        key=lambda index: int(
            np.count_nonzero(
                np.linalg.norm(translations - translations[index], axis=1) <= 6.0
            )
        ),
    )
    seed = translations[seed_index]
    inlier_mask = np.linalg.norm(translations - seed, axis=1) <= 6.0
    translation = np.median(translations[inlier_mask], axis=0)
    inlier_mask = np.linalg.norm(translations - translation, axis=1) <= 5.0
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 3:
        return None
    translation = np.median(translations[inlier_mask], axis=0)
    if tile_offset is not None:
        translation = np.asarray(
            [
                offset + round((value - offset) / tile_size) * tile_size
                for value, offset in zip(translation, tile_offset)
            ],
            dtype=np.float64,
        )
    matrix = np.asarray(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not _plausible_homography(matrix, query, target):
        return None
    projected = query_points @ linear.T + translation
    errors = np.linalg.norm(projected - target_points, axis=1)
    inlier_mask = errors <= 5.0
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 3:
        return None
    median_error = float(np.median(errors[inlier_mask]))
    return FeatureMatch(
        query_to_target=matrix,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(inliers / len(good)),
        median_reprojection_error=median_error,
    )


def _feature_match_from_geometry(
    geometry: np.ndarray | None,
    mask: np.ndarray | None,
    good: list[cv2.DMatch],
    query_points: np.ndarray,
    target_points: np.ndarray,
    query: FeatureIndex,
    target: FeatureIndex,
) -> FeatureMatch | None:
    if geometry is None or mask is None:
        return None
    matrix = np.asarray(geometry, dtype=np.float64)
    if not _plausible_homography(matrix, query, target):
        return None
    inlier_mask = np.asarray(mask).ravel()[: len(good)].astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / len(good)
    if inliers < 18 or (ratio < 0.32 and inliers < 256):
        return None
    projected = cv2.perspectiveTransform(query_points, matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points.reshape(-1, 2), axis=1)
    inlier_errors = errors[inlier_mask]
    median_error = float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
    if not np.isfinite(median_error) or median_error > 5.0:
        return None
    matrix /= matrix[2, 2]
    return FeatureMatch(
        query_to_target=matrix,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(ratio),
        median_reprojection_error=median_error,
    )


def rank_feature_candidates(
    query: FeatureIndex,
    candidates: tuple[tuple[str, FeatureIndex], ...],
    *,
    recent_record_ids: tuple[str, ...] = (),
    config: V4Config,
) -> tuple[RankedFeatureCandidate, ...]:
    if type(query) is not FeatureIndex:
        raise TypeError("query feature index must be an exact FeatureIndex")
    if type(candidates) is not tuple:
        raise TypeError("feature candidates must be a tuple")
    if type(recent_record_ids) is not tuple or any(
        type(value) is not str or not value for value in recent_record_ids
    ):
        raise ValueError("recent record IDs must be nonempty strings in a tuple")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")

    ranked = []
    seen_ids: set[str] = set()
    for record_id, index in candidates:
        if type(record_id) is not str or not record_id or record_id in seen_ids:
            raise ValueError("candidate record IDs must be unique nonempty strings")
        if type(index) is not FeatureIndex:
            raise TypeError("candidate indexes must be exact FeatureIndex instances")
        seen_ids.add(record_id)
        good = _good_descriptor_matches(
            query.descriptors[:FEATURE_COARSE_DESCRIPTORS],
            index.descriptors,
        )
        match_count = len(good)
        match_quality = match_count / max(
            1,
            min(len(query.descriptors), len(index.descriptors)),
        )
        thumbnail_distance = float(
            np.mean(
                np.abs(
                    query.thumbnail.astype(np.float32)
                    - index.thumbnail.astype(np.float32)
                )
            )
            / 255.0
        )
        ranked.append(
            RankedFeatureCandidate(
                record_id=record_id,
                index=index,
                match_count=match_count,
                match_quality=float(match_quality),
                thumbnail_distance=thumbnail_distance,
                reserved=False,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.match_count,
            -item.match_quality,
            item.thumbnail_distance,
            item.record_id,
        )
    )
    selected = ranked[: min(2, config.candidate_limit)]
    selected_ids = {item.record_id for item in selected}
    by_id = {item.record_id: item for item in ranked}
    for record_id in recent_record_ids:
        if len(selected) >= config.candidate_limit:
            break
        if record_id in selected_ids or record_id not in by_id:
            continue
        item = by_id[record_id]
        selected.append(
            RankedFeatureCandidate(
                record_id=item.record_id,
                index=item.index,
                match_count=item.match_count,
                match_quality=item.match_quality,
                thumbnail_distance=item.thumbnail_distance,
                reserved=True,
            )
        )
        selected_ids.add(record_id)
        break
    return tuple(selected)


def _good_descriptor_matches(
    query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
) -> list[cv2.DMatch]:
    if len(query_descriptors) < 2 or len(target_descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query_descriptors, target_descriptors, k=2)
    return [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]


def _validate_feature_pair(query: FeatureIndex, target: FeatureIndex) -> None:
    if type(query) is not FeatureIndex or type(target) is not FeatureIndex:
        raise TypeError("feature matching requires exact FeatureIndex instances")


def _plausible_homography(
    matrix: np.ndarray,
    query: FeatureIndex,
    target: FeatureIndex,
) -> bool:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if abs(float(matrix[2, 2])) < 1e-9:
        return False
    normalized = matrix / matrix[2, 2]
    determinant = float(np.linalg.det(normalized))
    if not np.isfinite(determinant) or abs(determinant) < 1e-8:
        return False
    if float(np.linalg.cond(normalized)) > 1e8:
        return False
    corners = np.asarray(
        [
            [[0.0, 0.0]],
            [[float(query.image_width), 0.0]],
            [[float(query.image_width), float(query.image_height)]],
            [[0.0, float(query.image_height)]],
        ],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, normalized).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False
    area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    target_area = float(target.image_width * target.image_height)
    if not 0.02 * target_area <= area <= 4.0 * target_area:
        return False
    if (
        np.any(projected[:, 0] < -target.image_width)
        or np.any(projected[:, 0] > 2 * target.image_width)
        or np.any(projected[:, 1] < -target.image_height)
        or np.any(projected[:, 1] > 2 * target.image_height)
    ):
        return False
    return True


def _readonly_copy(values: np.ndarray, dtype: np.dtype[object] | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _spatially_balanced_keypoint_indices(
    keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint],
    width: int,
    height: int,
) -> list[int]:
    available = min(len(keypoints), FEATURE_INDEX_MAX_DESCRIPTORS * 4)
    response_order = sorted(
        range(available),
        key=lambda index: (-float(keypoints[index].response), index),
    )
    quota = max(
        1,
        FEATURE_INDEX_MAX_DESCRIPTORS // (FEATURE_SELECTION_GRID**2),
    )
    counts = np.zeros((FEATURE_SELECTION_GRID, FEATURE_SELECTION_GRID), dtype=np.int16)
    octave_counts = np.zeros(
        (FEATURE_SELECTION_GRID, FEATURE_SELECTION_GRID, 8),
        dtype=np.int16,
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    for index in response_order:
        x, y = keypoints[index].pt
        column = min(FEATURE_SELECTION_GRID - 1, int(x * FEATURE_SELECTION_GRID / width))
        row = min(FEATURE_SELECTION_GRID - 1, int(y * FEATURE_SELECTION_GRID / height))
        octave = min(7, max(0, int(keypoints[index].octave)))
        if counts[row, column] >= quota or octave_counts[row, column, octave] >= 6:
            continue
        counts[row, column] += 1
        octave_counts[row, column, octave] += 1
        selected.append(index)
        selected_set.add(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            return selected
    for index in response_order:
        if index in selected_set:
            continue
        x, y = keypoints[index].pt
        column = min(FEATURE_SELECTION_GRID - 1, int(x * FEATURE_SELECTION_GRID / width))
        row = min(FEATURE_SELECTION_GRID - 1, int(y * FEATURE_SELECTION_GRID / height))
        if counts[row, column] >= quota:
            continue
        counts[row, column] += 1
        selected.append(index)
        selected_set.add(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            return selected
    for index in response_order:
        if index in selected_set:
            continue
        selected.append(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            break
    return selected


__all__ = (
    "FEATURE_INDEX_SCHEMA_VERSION",
    "FeatureIndex",
    "FeatureMatch",
    "extract_feature_index",
    "load_feature_index",
    "match_feature_indexes",
    "match_feature_indexes_constrained",
    "rank_feature_candidates",
    "save_feature_index",
)
