from dataclasses import FrozenInstanceError
from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from PIL import Image

import watermark_v4.detector as detector_module
from tests.test_watermark_v4_features import _feature_image
from watermark_v4 import V4Config, embed_codeword
from watermark_v4.detector import (
    CandidateEvidence,
    V4Candidate,
    V4Detection,
    decode_aligned_candidate,
    detect_v4,
)
from watermark_v4.features import extract_feature_index, load_feature_index
from watermark_v4.payload import AuthContext, authentication_tag, encode_codeword
from watermark_v4.sync import embed_pilot


AUTH_KEY = b"detector-test-key-material-32-bytes-minimum"


def _tag(trace_id: str) -> bytes:
    return authentication_tag(
        AuthContext(
            "v4",
            "test-key",
            1,
            b"s" * 32,
            trace_id,
        ),
        AUTH_KEY,
    )


def _marked_candidate() -> tuple[Image.Image, V4Candidate]:
    trace_id = "TR-V4-DETECTOR-A"
    tag = _tag(trace_id)
    original = _feature_image((768, 640), seed=77)
    marked = embed_codeword(
        embed_pilot(original, V4Config()),
        encode_codeword(tag),
        V4Config(),
    )
    candidate = V4Candidate(
        record_id="record-a",
        trace_id=trace_id,
        auth_tag=tag,
        feature_index=extract_feature_index(marked),
    )
    return marked, candidate


def test_decode_aligned_candidate_recovers_authenticated_crop() -> None:
    marked, candidate = _marked_candidate()
    query = marked.crop((64, 64, 704, 576))
    query_to_target = np.asarray(
        [[1.0, 0.0, 64.0], [0.0, 1.0, 64.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    evidence = decode_aligned_candidate(
        query,
        query_to_target,
        candidate,
        V4Config(),
    )

    assert type(evidence) is CandidateEvidence
    assert evidence.record_id == candidate.record_id
    assert evidence.trace_id == candidate.trace_id
    assert evidence.tile_count >= 2
    assert evidence.phase_count >= 2
    assert evidence.minimum_coverage >= 0.70
    assert evidence.corrected_symbols >= 0
    assert evidence.erasure_count in range(5)
    assert evidence.bit_errors >= 0
    assert evidence.mean_abs_score > 0.0


def test_decode_aligned_candidate_rejects_wrong_candidate_tag() -> None:
    marked, candidate = _marked_candidate()
    wrong = V4Candidate(
        record_id="record-wrong",
        trace_id="TR-V4-WRONG",
        auth_tag=_tag("TR-V4-WRONG"),
        feature_index=candidate.feature_index,
    )
    query = marked.crop((64, 64, 704, 576))
    query_to_target = np.asarray(
        [[1.0, 0.0, 64.0], [0.0, 1.0, 64.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert (
        decode_aligned_candidate(query, query_to_target, wrong, V4Config())
        is None
    )


def test_decode_aligned_candidate_rejects_insufficient_tile_coverage() -> None:
    marked, candidate = _marked_candidate()
    query = marked.crop((30, 30, 180, 180))
    query_to_target = np.asarray(
        [[1.0, 0.0, 30.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert (
        decode_aligned_candidate(query, query_to_target, candidate, V4Config())
        is None
    )


def test_decode_aligned_candidate_honors_expired_deadline() -> None:
    marked, candidate = _marked_candidate()

    with pytest.raises(TimeoutError, match="deadline"):
        decode_aligned_candidate(
            marked,
            np.eye(3, dtype=np.float64),
            candidate,
            V4Config(),
            deadline=monotonic() - 1.0,
        )


def test_candidate_and_evidence_are_frozen_slotted_and_strict() -> None:
    _, candidate = _marked_candidate()

    assert not hasattr(candidate, "__dict__")
    with pytest.raises(FrozenInstanceError):
        candidate.record_id = "changed"  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError), match="auth"):
        V4Candidate("record", "trace", b"bad", candidate.feature_index)


def test_detect_v4_recovers_exactly_one_authenticated_candidate() -> None:
    marked, candidate = _marked_candidate()
    query = marked.crop((64, 64, 704, 576))
    decoys = tuple(
        V4Candidate(
            record_id=f"decoy-{seed}",
            trace_id=f"TR-DECOY-{seed}",
                auth_tag=_tag(f"TR-DECOY-{seed}"),
            feature_index=extract_feature_index(_feature_image((768, 640), seed=seed)),
        )
        for seed in (101, 102, 103)
    )

    result = detect_v4(
        query,
        (decoys[0], candidate, decoys[1], decoys[2]),
        V4Config(),
        recent_record_ids=("decoy-103",),
    )

    assert type(result) is V4Detection
    assert result.record_id == candidate.record_id
    assert result.trace_id == candidate.trace_id
    assert result.codec == V4Config().codec
    assert result.geometry_method in ("fft_orb_ransac", "orb_ransac")
    assert result.orb_inliers >= 18
    assert result.orb_ratio >= 0.32
    assert result.candidate_count <= 3
    assert result.elapsed_seconds >= 0.0


def test_detect_v4_extracts_query_features_and_fft_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marked, candidate = _marked_candidate()
    query = marked.crop((64, 64, 704, 576))
    candidates = (candidate,)
    calls = {"features": 0, "fft": 0}
    original_features = detector_module.extract_feature_index
    original_fft = detector_module.detect_pilot

    def counted_features(*args: object, **kwargs: object) -> object:
        calls["features"] += 1
        return original_features(*args, **kwargs)

    def counted_fft(*args: object, **kwargs: object) -> object:
        calls["fft"] += 1
        return original_fft(*args, **kwargs)

    monkeypatch.setattr(detector_module, "extract_feature_index", counted_features)
    monkeypatch.setattr(detector_module, "detect_pilot", counted_fft)

    result = detect_v4(query, candidates, V4Config())

    assert type(result) is V4Detection
    assert calls == {"features": 1, "fft": 1}


def test_detect_v4_never_checks_more_than_three_candidate_geometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _feature_image((640, 480), seed=500)
    candidates = tuple(
        V4Candidate(
            record_id=f"record-{seed}",
            trace_id=f"TR-{seed}",
                auth_tag=_tag(f"TR-{seed}"),
            feature_index=extract_feature_index(
                _feature_image((640, 480), seed=seed)
            ),
        )
        for seed in range(10, 16)
    )
    calls = 0
    original_match = detector_module.match_feature_indexes

    def counted_match(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_match(*args, **kwargs)

    monkeypatch.setattr(detector_module, "match_feature_indexes", counted_match)

    assert detect_v4(query, candidates, V4Config()) is None
    assert calls <= V4Config().candidate_limit == 3


def test_detect_v4_rejects_multiple_authenticated_candidates() -> None:
    marked, candidate = _marked_candidate()
    duplicate = V4Candidate(
        record_id="record-duplicate",
        trace_id=candidate.trace_id,
        auth_tag=candidate.auth_tag,
        feature_index=candidate.feature_index,
    )
    query = marked.crop((64, 64, 704, 576))

    assert detect_v4(query, (candidate, duplicate), V4Config()) is None


def test_detect_v4_tries_one_rounded_translation_refinement() -> None:
    tag = bytes.fromhex("b52e76fb10203040")
    original = _feature_image((512, 384), seed=808)
    marked = embed_codeword(
        embed_pilot(original, V4Config()),
        encode_codeword(tag),
        V4Config(),
    )
    candidate = V4Candidate(
        record_id="subpixel-record",
        trace_id="TR-SUBPIXEL-REGRESSION",
        auth_tag=tag,
        feature_index=extract_feature_index(marked),
    )
    query = marked.crop((64, 0, 512, 384))

    result = detect_v4(query, (candidate,), V4Config())

    assert type(result) is V4Detection
    assert result.record_id == candidate.record_id


def test_detect_v4_recovers_real_random_crop_at_half_scale() -> None:
    config = V4Config()
    tag = bytes.fromhex("a1b2c3d410203040")
    with Image.open("img/1.png") as loaded:
        source = loaded.convert("RGB")
    marked = embed_codeword(
        embed_pilot(source, config),
        encode_codeword(tag),
        config,
    )
    candidate = V4Candidate(
        record_id="real-crop-record",
        trace_id="TR-REAL-CROP-REGRESSION",
        auth_tag=tag,
        feature_index=extract_feature_index(marked),
    )
    scaled = marked.resize(
        (round(marked.width * 0.5), round(marked.height * 0.5)),
        Image.Resampling.BICUBIC,
    )
    query = scaled.crop((374, 189, 709, 414))

    result = detect_v4(query, (candidate,), config)

    assert type(result) is V4Detection
    assert result.record_id == candidate.record_id
    assert result.trace_id == candidate.trace_id
