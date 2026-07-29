"""Bounded ORB/RANSAC confirmation with a strict LightGlue fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import UUID

import cv2
import numpy as np

from trace_app.v4.deadlines import Deadline


@dataclass(frozen=True, slots=True)
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray
    image_width: int
    image_height: int
    kind: str

    def __post_init__(self) -> None:
        points = np.array(self.points, copy=True, order="C")
        descriptors = np.array(self.descriptors, copy=True, order="C")
        if points.dtype != np.dtype("float32") or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("feature points must be float32 with shape (n, 2)")
        if self.kind == "orb":
            expected_dtype, expected_width = np.dtype("uint8"), 32
        elif self.kind == "superpoint":
            expected_dtype, expected_width = np.dtype("float32"), 256
        else:
            raise ValueError("feature kind must be orb or superpoint")
        if (
            descriptors.dtype != expected_dtype
            or descriptors.ndim != 2
            or descriptors.shape != (points.shape[0], expected_width)
        ):
            raise ValueError(f"invalid {self.kind} descriptors")
        if not np.isfinite(points).all() or (
            descriptors.dtype.kind == "f" and not np.isfinite(descriptors).all()
        ):
            raise ValueError("features must contain finite values")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("feature image dimensions must be positive")
        points.setflags(write=False)
        descriptors.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "descriptors", descriptors)


@dataclass(frozen=True, slots=True)
class GeometryCandidate:
    source_group_id: UUID
    orb: FeatureSet
    superpoint: FeatureSet | None = None
    allow_lightglue: bool = False

    def __post_init__(self) -> None:
        if self.orb.kind != "orb":
            raise ValueError("candidate ORB features are required")
        if self.superpoint is not None and self.superpoint.kind != "superpoint":
            raise ValueError("candidate SuperPoint features are invalid")
        if self.superpoint is not None and (
            self.superpoint.image_width != self.orb.image_width
            or self.superpoint.image_height != self.orb.image_height
        ):
            raise ValueError("candidate feature dimensions must match")


@dataclass(frozen=True, slots=True)
class GeometryEvidence:
    homography: np.ndarray
    method: str
    inliers: int
    ratio: float
    reprojection_error: float


@dataclass(frozen=True, slots=True)
class ConfirmedGroup:
    source_group_id: UUID
    homography: np.ndarray
    method: str
    inliers: int
    ratio: float
    reprojection_error: float


Matcher = Callable[[FeatureSet, FeatureSet], GeometryEvidence | None]


def validate_homography(
    value: np.ndarray,
    *,
    query_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype.kind != "f" or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite floating 3x3 matrix")
    matrix = np.asarray(matrix, dtype=np.float64)
    if abs(matrix[2, 2]) < 1e-12 or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("homography is singular")
    matrix /= matrix[2, 2]
    query_width, query_height = query_size
    target_width, target_height = target_size
    if min(query_width, query_height, target_width, target_height) <= 0:
        raise ValueError("homography dimensions must be positive")
    corners = np.asarray(
        [[[0.0, 0.0], [query_width, 0.0], [query_width, query_height], [0.0, query_height]]],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(corners, matrix)[0]
    if not np.isfinite(projected).all():
        raise ValueError("homography projects non-finite corners")
    x, y = projected[:, 0], projected[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    area_ratio = area / float(query_width * query_height)
    if not 0.01 <= area_ratio <= 100.0:
        raise ValueError("homography has implausible projected area")
    if np.max(np.abs(x)) > target_width * 10 or np.max(np.abs(y)) > target_height * 10:
        raise ValueError("homography projects outside plausible bounds")
    matrix.setflags(write=False)
    return matrix


def match_orb_ransac(query: FeatureSet, target: FeatureSet) -> GeometryEvidence | None:
    if query.kind != "orb" or target.kind != "orb":
        raise ValueError("ORB matcher requires ORB features")
    if min(len(query.descriptors), len(target.descriptors)) < 18:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        query.descriptors, target.descriptors, k=2
    )
    good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.75 * second.distance]
    if len(good) < 18:
        return None
    query_points = np.asarray([query.points[item.queryIdx] for item in good], dtype=np.float32).reshape(-1, 1, 2)
    target_points = np.asarray([target.points[item.trainIdx] for item in good], dtype=np.float32).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(query_points, target_points, cv2.RANSAC, 5.0)
    if matrix is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / len(good)
    if inliers < 18 or ratio < 0.32:
        return None
    projected = cv2.perspectiveTransform(query_points, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points.reshape(-1, 2), axis=1)
    error = float(np.median(errors[inlier_mask]))
    if not np.isfinite(error) or error > 5.0:
        return None
    return GeometryEvidence(np.asarray(matrix, dtype=np.float64), "orb_ransac", inliers, ratio, error)


def _confirmed(
    candidate: GeometryCandidate,
    evidence: GeometryEvidence,
    query: FeatureSet,
    *,
    expected_method: str,
) -> ConfirmedGroup | None:
    if (
        evidence.method != expected_method
        or evidence.inliers < 4
        or not np.isfinite(evidence.ratio)
        or not np.isfinite(evidence.reprojection_error)
        or not 0.0 <= evidence.ratio <= 1.0
        or evidence.reprojection_error < 0
    ):
        return None
    try:
        matrix = validate_homography(
            evidence.homography,
            query_size=(query.image_width, query.image_height),
            target_size=(candidate.orb.image_width, candidate.orb.image_height),
        )
    except ValueError:
        return None
    return ConfirmedGroup(
        candidate.source_group_id,
        matrix,
        evidence.method,
        evidence.inliers,
        evidence.ratio,
        evidence.reprojection_error,
    )


def confirm_source_group(
    query_orb: FeatureSet,
    candidates: tuple[GeometryCandidate, ...],
    deadline: Deadline,
    *,
    query_superpoint: FeatureSet | None = None,
    orb_matcher: Matcher = match_orb_ransac,
    lightglue_matcher: Matcher | None = None,
    max_lightglue_candidates: int = 3,
    low_texture_threshold: int = 64,
) -> ConfirmedGroup | None:
    if query_orb.kind != "orb":
        raise ValueError("query ORB features are required")
    if query_superpoint is not None and query_superpoint.kind != "superpoint":
        raise ValueError("query SuperPoint features are invalid")
    if max_lightglue_candidates < 0 or low_texture_threshold < 1:
        raise ValueError("invalid LightGlue bounds")
    lightglue_attempts = 0
    for candidate in candidates:
        deadline.check("geometry_candidate")
        evidence = orb_matcher(query_orb, candidate.orb)
        deadline.check("geometry_candidate")
        if evidence is not None:
            confirmed = _confirmed(
                candidate, evidence, query_orb, expected_method="orb_ransac"
            )
            if confirmed is not None:
                return confirmed
        low_texture = min(len(query_orb.descriptors), len(candidate.orb.descriptors)) < low_texture_threshold
        eligible = candidate.allow_lightglue or low_texture
        if (
            eligible
            and lightglue_matcher is not None
            and query_superpoint is not None
            and candidate.superpoint is not None
            and lightglue_attempts < max_lightglue_candidates
        ):
            deadline.check("lightglue_fallback")
            lightglue_attempts += 1
            evidence = lightglue_matcher(query_superpoint, candidate.superpoint)
            deadline.check("lightglue_fallback")
            if evidence is not None:
                confirmed = _confirmed(
                    candidate, evidence, query_superpoint, expected_method="lightglue"
                )
                if confirmed is not None:
                    return confirmed
    return None


__all__ = (
    "ConfirmedGroup",
    "FeatureSet",
    "GeometryCandidate",
    "GeometryEvidence",
    "confirm_source_group",
    "match_orb_ransac",
    "validate_homography",
)
