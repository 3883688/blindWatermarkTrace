from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from .config import V4Config
from .dct import extract_tile_scores
from .features import (
    FeatureIndex,
    extract_feature_index,
    match_feature_indexes,
    match_feature_indexes_constrained,
    rank_feature_candidates,
)
from .payload import decode_candidate_codeword, phase_for_tile, phase_permutation
from .sync import detect_pilot


@dataclass(frozen=True, slots=True)
class V4Candidate:
    record_id: str
    trace_id: str
    auth_tag: bytes
    feature_index: FeatureIndex

    def __post_init__(self) -> None:
        for name, value in (("record ID", self.record_id), ("trace ID", self.trace_id)):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty canonical string")
        if type(self.auth_tag) is not bytes or len(self.auth_tag) != 4:
            raise ValueError("candidate auth tag must contain exactly 4 bytes")
        if type(self.feature_index) is not FeatureIndex:
            raise TypeError("candidate feature index must be an exact FeatureIndex")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    record_id: str
    trace_id: str
    tile_count: int
    phase_count: int
    minimum_coverage: float
    corrected_symbols: int
    erasure_count: int
    bit_errors: int
    mean_abs_score: float

    def __post_init__(self) -> None:
        for name, value in (("record ID", self.record_id), ("trace ID", self.trace_id)):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name, value in (
            ("tile count", self.tile_count),
            ("phase count", self.phase_count),
            ("corrected symbols", self.corrected_symbols),
            ("erasure count", self.erasure_count),
            ("bit errors", self.bit_errors),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (
            ("minimum coverage", self.minimum_coverage),
            ("mean absolute score", self.mean_abs_score),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.minimum_coverage > 1.0:
            raise ValueError("minimum coverage must not exceed one")


@dataclass(frozen=True, slots=True)
class V4Detection:
    record_id: str
    trace_id: str
    codec: str
    geometry_method: str
    orb_inliers: int
    orb_ratio: float
    candidate_count: int
    tile_count: int
    phase_count: int
    corrected_symbols: int
    erasure_count: int
    bit_errors: int
    mean_abs_score: float
    sync_confidence: float | None
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("detection record ID must be a nonempty string")
        if type(self.trace_id) is not str or not self.trace_id:
            raise ValueError("detection trace ID must be a nonempty string")
        if type(self.codec) is not str or not self.codec:
            raise ValueError("detection codec must be a nonempty string")
        if self.geometry_method not in ("fft_orb_ransac", "orb_ransac"):
            raise ValueError("detection geometry method is invalid")
        for name, value in (
            ("ORB inliers", self.orb_inliers),
            ("candidate count", self.candidate_count),
            ("tile count", self.tile_count),
            ("phase count", self.phase_count),
            ("corrected symbols", self.corrected_symbols),
            ("erasure count", self.erasure_count),
            ("bit errors", self.bit_errors),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (
            ("ORB ratio", self.orb_ratio),
            ("mean absolute score", self.mean_abs_score),
            ("elapsed seconds", self.elapsed_seconds),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.orb_ratio > 1.0:
            raise ValueError("ORB ratio must not exceed one")
        if self.sync_confidence is not None and (
            type(self.sync_confidence) is not float
            or not np.isfinite(self.sync_confidence)
            or not 0.0 <= self.sync_confidence <= 1.0
        ):
            raise ValueError("sync confidence must be None or a float between zero and one")


def decode_aligned_candidate(
    image: Image.Image,
    query_to_target: np.ndarray,
    candidate: V4Candidate,
    config: V4Config,
    *,
    deadline: float | None = None,
) -> CandidateEvidence | None:
    _validate_image(image)
    matrix = _validated_homography(query_to_target)
    if type(candidate) is not V4Candidate:
        raise TypeError("candidate must be an exact V4Candidate")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")
    _validate_deadline(deadline)
    _check_deadline(deadline)

    target_width = candidate.feature_index.image_width
    target_height = candidate.feature_index.image_height
    rgb = np.asarray(image)[..., :3]
    warped_rgb = cv2.warpPerspective(
        rgb,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    source_mask = np.full((image.height, image.width), 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        source_mask,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _check_deadline(deadline)
    luminance = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2YCrCb)[..., 0]

    logical_batches: list[np.ndarray] = []
    coverages: list[float] = []
    phases: set[int] = set()
    tile_size = config.tile_size
    for tile_y in range(target_height // tile_size):
        for tile_x in range(target_width // tile_size):
            _check_deadline(deadline)
            top = tile_y * tile_size
            left = tile_x * tile_size
            coverage = float(
                np.mean(
                    valid_mask[
                        top : top + tile_size,
                        left : left + tile_size,
                    ]
                    > 0
                )
            )
            if coverage < config.minimum_coverage:
                continue
            physical = np.asarray(
                extract_tile_scores(
                    luminance[
                        top : top + tile_size,
                        left : left + tile_size,
                    ],
                    config,
                ),
                dtype=np.float64,
            )
            phase = phase_for_tile(tile_x, tile_y)
            logical = physical[np.asarray(phase_permutation(phase), dtype=np.intp)]
            robust_energy = float(np.median(np.abs(logical)))
            if not np.isfinite(robust_energy) or robust_energy <= 1e-9:
                continue
            logical_batches.append(logical / robust_energy)
            coverages.append(coverage)
            phases.add(phase)

    if (
        len(logical_batches) < config.minimum_tiles
        or len(phases) < config.minimum_phases
    ):
        return None
    aggregate = np.mean(np.stack(logical_batches), axis=0)
    observed = _scores_to_bytes(aggregate)
    byte_confidences = tuple(
        float(
            minimum
            / (1.0 + minimum)
        )
        for minimum in (
            np.min(np.abs(aggregate[start : start + 8]))
            for start in range(0, 64, 8)
        )
    )
    decoded = decode_candidate_codeword(
        observed,
        candidate.auth_tag,
        byte_confidences,
    )
    if decoded is None:
        return None
    return CandidateEvidence(
        record_id=candidate.record_id,
        trace_id=candidate.trace_id,
        tile_count=len(logical_batches),
        phase_count=len(phases),
        minimum_coverage=float(min(coverages)),
        corrected_symbols=decoded.corrected_symbols,
        erasure_count=decoded.erasure_count,
        bit_errors=decoded.bit_errors,
        mean_abs_score=float(np.mean(np.abs(aggregate))),
    )


def detect_v4(
    image: Image.Image,
    candidates: tuple[V4Candidate, ...],
    config: V4Config,
    *,
    recent_record_ids: tuple[str, ...] = (),
    deadline: float | None = None,
) -> V4Detection | None:
    _validate_image(image)
    if type(candidates) is not tuple or any(
        type(candidate) is not V4Candidate for candidate in candidates
    ):
        raise TypeError("v4 candidates must be a tuple of exact V4Candidate instances")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")
    _validate_deadline(deadline)
    started = monotonic()
    hard_deadline = started + config.hard_timeout_seconds
    effective_deadline = hard_deadline if deadline is None else min(deadline, hard_deadline)
    _check_deadline(effective_deadline)
    if not candidates:
        return None

    query_index = extract_feature_index(image)
    _check_deadline(effective_deadline)
    sync = detect_pilot(image, config, deadline=effective_deadline)
    candidate_by_id = {candidate.record_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("v4 candidate record IDs must be unique")
    ranked = rank_feature_candidates(
        query_index,
        tuple(
            (candidate.record_id, candidate.feature_index)
            for candidate in candidates
        ),
        recent_record_ids=recent_record_ids,
        config=config,
    )

    authenticated = []
    for ranked_candidate in ranked[: config.candidate_limit]:
        _check_deadline(effective_deadline)
        candidate = candidate_by_id[ranked_candidate.record_id]
        feature_match = match_feature_indexes(
            query_index,
            candidate.feature_index,
        )
        evidence = None
        matches = [] if feature_match is None else [feature_match]
        if feature_match is not None:
            rounded_translation = feature_match.query_to_target.copy()
            rounded_translation[0, 2] = round(float(rounded_translation[0, 2]))
            rounded_translation[1, 2] = round(float(rounded_translation[1, 2]))
            matrices = [feature_match.query_to_target]
            if not np.array_equal(rounded_translation, feature_match.query_to_target):
                matrices.append(rounded_translation)
            for matrix in matrices:
                _check_deadline(effective_deadline)
                evidence = decode_aligned_candidate(
                    image,
                    matrix,
                    candidate,
                    config,
                    deadline=effective_deadline,
                )
                if evidence is not None:
                    break
        if evidence is None and sync is not None:
            constrained = match_feature_indexes_constrained(
                query_index,
                candidate.feature_index,
                rotation_degrees=sync.rotation_degrees,
                scale=sync.scale,
                tile_size=config.tile_size,
                tile_offset=(sync.offset_x, sync.offset_y)
                if sync.offset_x is not None and sync.offset_y is not None
                else None,
            )
            if constrained is not None:
                matches.append(constrained)
                _check_deadline(effective_deadline)
                evidence = decode_aligned_candidate(
                    image,
                    constrained.query_to_target,
                    candidate,
                    config,
                    deadline=effective_deadline,
                )
        if evidence is not None:
            authenticated.append((evidence, matches[-1]))
    if len(authenticated) != 1:
        return None

    evidence, feature_match = authenticated[0]
    return V4Detection(
        record_id=evidence.record_id,
        trace_id=evidence.trace_id,
        codec=config.codec,
        geometry_method="fft_orb_ransac" if sync is not None else "orb_ransac",
        orb_inliers=feature_match.inliers,
        orb_ratio=feature_match.inlier_ratio,
        candidate_count=len(ranked),
        tile_count=evidence.tile_count,
        phase_count=evidence.phase_count,
        corrected_symbols=evidence.corrected_symbols,
        erasure_count=evidence.erasure_count,
        bit_errors=evidence.bit_errors,
        mean_abs_score=evidence.mean_abs_score,
        sync_confidence=None if sync is None else sync.confidence,
        elapsed_seconds=float(monotonic() - started),
    )


def _scores_to_bytes(scores: np.ndarray) -> bytes:
    bits = (scores > 0.0).astype(np.uint8).reshape(8, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8))[None, :]
    return bytes(np.sum(bits * weights, axis=1, dtype=np.uint16).astype(np.uint8))


def _validated_homography(value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("query-to-target homography must be a NumPy array")
    if value.shape != (3, 3) or value.dtype.kind not in "f":
        raise ValueError("query-to-target homography must be a 3x3 floating matrix")
    matrix = value.astype(np.float64, copy=True)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ValueError("query-to-target homography must be finite and nonsingular")
    return matrix


def _validate_image(image: Image.Image) -> None:
    if type(image) is not Image.Image:
        raise TypeError("query image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("query image mode must be RGB or RGBA")


def _validate_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise TypeError("deadline must be a finite monotonic timestamp")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("v4 detection deadline expired")


__all__ = (
    "CandidateEvidence",
    "V4Candidate",
    "V4Detection",
    "decode_aligned_candidate",
    "detect_v4",
)
