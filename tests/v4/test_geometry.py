from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from trace_app.v4.deadlines import Deadline, DeadlineExceeded
from trace_app.v4.geometry import (
    FeatureSet,
    GeometryCandidate,
    GeometryEvidence,
    confirm_source_group,
    validate_homography,
)


def _features(count: int, *, kind: str = "orb") -> FeatureSet:
    descriptor_width = 32 if kind == "orb" else 256
    descriptor_dtype = np.uint8 if kind == "orb" else np.float32
    return FeatureSet(
        points=np.column_stack((np.arange(count), np.arange(count))).astype(np.float32),
        descriptors=np.zeros((count, descriptor_width), dtype=descriptor_dtype),
        image_width=640,
        image_height=480,
        kind=kind,
    )


def _evidence(method: str = "orb_ransac") -> GeometryEvidence:
    return GeometryEvidence(
        homography=np.eye(3, dtype=np.float64),
        method=method,
        inliers=24,
        ratio=0.75,
        reprojection_error=1.2,
    )


def test_orb_is_attempted_first_and_returns_source_group_only() -> None:
    calls: list[str] = []
    candidate = GeometryCandidate(UUID(int=3), _features(30), _features(30, kind="superpoint"), allow_lightglue=True)

    confirmed = confirm_source_group(
        _features(30),
        (candidate,),
        Deadline.after(10),
        query_superpoint=_features(30, kind="superpoint"),
        orb_matcher=lambda query, target: calls.append("orb") or _evidence(),
        lightglue_matcher=lambda query, target: calls.append("lightglue") or _evidence("lightglue"),
    )

    assert confirmed is not None
    assert confirmed.source_group_id == UUID(int=3)
    assert confirmed.method == "orb_ransac"
    assert not hasattr(confirmed, "record_id")
    assert calls == ["orb"]


def test_feature_set_freezes_an_owned_copy_without_mutating_caller() -> None:
    points = np.zeros((2, 2), dtype=np.float32)
    descriptors = np.zeros((2, 32), dtype=np.uint8)

    features = FeatureSet(points, descriptors, 10, 10, "orb")

    assert points.flags.writeable and descriptors.flags.writeable
    assert not features.points.flags.writeable and not features.descriptors.flags.writeable


def test_lightglue_is_bounded_and_only_used_for_difficult_candidates() -> None:
    calls: list[tuple[str, int]] = []
    candidates = tuple(
        GeometryCandidate(
            UUID(int=value),
            _features(20),
            _features(20, kind="superpoint"),
            allow_lightglue=value != 1,
        )
        for value in range(1, 6)
    )

    confirmed = confirm_source_group(
        _features(20),
        candidates,
        Deadline.after(10),
        query_superpoint=_features(20, kind="superpoint"),
        orb_matcher=lambda query, target: None,
        lightglue_matcher=lambda query, target: calls.append(("lightglue", len(target.descriptors))) or None,
        max_lightglue_candidates=2,
    )

    assert confirmed is None
    assert len(calls) == 2


def test_low_texture_candidate_can_use_lightglue_without_explicit_hint() -> None:
    candidate = GeometryCandidate(
        UUID(int=7), _features(8), _features(8, kind="superpoint"), allow_lightglue=False
    )

    confirmed = confirm_source_group(
        _features(8),
        (candidate,),
        Deadline.after(10),
        query_superpoint=_features(8, kind="superpoint"),
        orb_matcher=lambda query, target: None,
        lightglue_matcher=lambda query, target: _evidence("lightglue"),
        low_texture_threshold=16,
    )

    assert confirmed is not None and confirmed.method == "lightglue"


def test_invalid_homography_is_rejected() -> None:
    for matrix in (
        np.zeros((3, 3), dtype=np.float64),
        np.full((3, 3), np.nan, dtype=np.float64),
        np.diag([1e-9, 1e-9, 1.0]).astype(np.float64),
    ):
        with pytest.raises(ValueError):
            validate_homography(matrix, query_size=(640, 480), target_size=(640, 480))


def test_non_finite_or_wrong_method_evidence_cannot_confirm() -> None:
    candidate = GeometryCandidate(UUID(int=8), _features(30))
    for evidence in (
        GeometryEvidence(np.eye(3), "lightglue", 24, 0.8, 1.0),
        GeometryEvidence(np.eye(3), "orb_ransac", 24, 0.8, float("nan")),
    ):
        confirmed = confirm_source_group(
            _features(30),
            (candidate,),
            Deadline.after(10),
            orb_matcher=lambda query, target, value=evidence: value,
        )
        assert confirmed is None


def test_deadline_is_checked_between_every_candidate() -> None:
    now = [0.0]
    deadline = Deadline.after(1.0, clock=lambda: now[0])

    def matcher(query, target):
        now[0] = 2.0
        return None

    with pytest.raises(DeadlineExceeded, match="geometry_candidate"):
        confirm_source_group(
            _features(30),
            (GeometryCandidate(UUID(int=1), _features(30)), GeometryCandidate(UUID(int=2), _features(30))),
            deadline,
            orb_matcher=matcher,
        )
