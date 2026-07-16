from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from PIL import Image
from watermark_v4.detector import V4Candidate


Detector = Callable[[Image.Image], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class DetectionPipeline:
    detectors: tuple[Detector, ...]

    def __call__(self, image: Image.Image) -> dict[str, Any] | None:
        for detector in self.detectors:
            result = detector(image)
            if result is not None:
                return result
        return None


def v4_candidate_records(
    *,
    records: list[dict[str, Any]],
    data_dir: Path,
    version_v4: int,
    config_factory: Callable[[], Any],
    record_feature_index_path: Callable[[dict[str, Any], Path], Path | None],
    load_feature_index: Callable[[Path], Any | None],
) -> tuple[V4Candidate, ...]:
    config = config_factory()
    candidates = []
    for record in records:
        if record.get("robust_watermark_version") != version_v4:
            continue
        if record.get("robust_watermark_codec") != config.codec:
            continue
        record_id = str(record.get("id") or "").strip()
        trace_id = str(record.get("trace_id") or "").strip()
        auth_hex = str(record.get("robust_auth_code") or "").strip()
        if not record_id or not trace_id or not re.fullmatch(r"[0-9a-f]{8}", auth_hex):
            continue
        path = record_feature_index_path(record, data_dir)
        if path is None:
            continue
        feature_index = load_feature_index(path)
        if feature_index is None:
            continue
        candidates.append(
            V4Candidate(
                record_id=record_id,
                trace_id=trace_id,
                auth_tag=bytes.fromhex(auth_hex),
                feature_index=feature_index,
            )
        )
    return tuple(candidates)


def detect_v4_watermark(
    image: Image.Image,
    candidates: tuple[V4Candidate, ...] | None = None,
    *,
    records: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    generated_trace_ids: list[str],
    version_v4: int,
    config_factory: Callable[[], Any],
    candidate_records: Callable[[], tuple[V4Candidate, ...]],
    detect: Callable[..., Any],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
) -> dict[str, Any] | None:
    available = candidate_records() if candidates is None else candidates
    if not available:
        return None
    current_records = records() if callable(records) else records
    record_by_id = {
        str(record.get("id")): record
        for record in current_records
        if record.get("robust_watermark_version") == version_v4
    }
    recent_record_ids = tuple(
        candidate.record_id
        for trace_id in generated_trace_ids
        for candidate in available
        if candidate.trace_id == trace_id
    )
    result = detect(
        image.convert("RGB"),
        available,
        config_factory(),
        recent_record_ids=recent_record_ids,
    )
    if result is None:
        return None
    record = record_by_id.get(result.record_id)
    if record is None:
        return None
    return with_evidence_fields({
        "id": result.record_id,
        "trace_id": result.trace_id,
        "user_id": record.get("user_id"),
        "mode": "v4_authenticated_dct",
        "mode_label": "V4 认证水印",
        "created_at": record.get("created_at"),
        "confidence": max(80, min(99, 99 - result.bit_errors * 4)),
        "phash_match": False,
        "status": "V4 认证命中",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers"),
        "layer_scores": {},
        "code_recovery": {
            "method": result.geometry_method,
            "codec": result.codec,
            "candidate_count": result.candidate_count,
            "authenticated_tiles": result.tile_count,
            "phase_count": result.phase_count,
            "corrected_symbols": result.corrected_symbols,
            "erasure_count": result.erasure_count,
            "bit_errors": result.bit_errors,
            "mean_abs_score": round(result.mean_abs_score, 6),
            "orb_inliers": result.orb_inliers,
            "orb_ratio": round(result.orb_ratio, 6),
            "sync_confidence": result.sync_confidence,
            "elapsed_ms": round(result.elapsed_seconds * 1000.0, 3),
        },
    }, record)


def should_run_frequency_fallbacks(image: Image.Image) -> bool:
    width, height = image.size
    pixels = width * height
    if pixels <= 3_000_000:
        return True
    aspect = max(width, height) / max(1, min(width, height))
    return pixels <= 5_000_000 and aspect >= 2.2


def should_run_visual_match_fallback(
    image: Image.Image, *, records: list[dict[str, Any]]
) -> bool:
    if not any(record.get("robust_watermark") for record in records):
        return False
    width, height = image.size
    return width * height >= 40_000


def extract_watermark_from_image(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    v4_candidates: tuple[V4Candidate, ...],
    detect_v4_watermark: Callable[..., dict[str, Any] | None],
    extract_full_lsb: Callable[[Image.Image], dict[str, Any] | None],
    extract_block_lsb: Callable[[Image.Image], dict[str, Any] | None],
    is_registered_original_image: Callable[[Image.Image], bool],
    should_run_frequency_fallbacks: Callable[[Image.Image], bool],
    should_run_visual_match_fallback: Callable[[Image.Image], bool],
    detect_dot_matrix_trace: Detector,
    detect_aligned_authenticated_watermark: Callable[..., dict[str, Any] | None],
    detect_by_visual_match: Detector,
    detect_small_crop_trace: Detector,
    detect_watermark_code: Detector,
    detect_robust_watermark: Detector,
    detect_by_residual_match: Detector,
    detect_visible_copyright: Detector,
    record_detection_result: Callable[[bool], None],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
    mode_label: Callable[[str], str],
    layer_scores_for_image: Callable[[Image.Image, str], Any],
    not_found_error: Callable[[], Exception],
    watermark_layers: Any,
    aligned_authenticated_detection_enabled: bool,
    aligned_candidate_limit: int,
    watermark_detection_budget_seconds: float,
    dense_watermark_fallback_enabled: bool,
    visual_match_fallback_enabled: bool,
    visible_watermark_detection_enabled: bool,
) -> dict[str, Any]:
    if v4_candidates:
        if is_registered_original_image(image):
            record_detection_result(False)
            raise not_found_error()
        v4_match = detect_v4_watermark(image, v4_candidates)
        if v4_match:
            record_detection_result(True)
            return v4_match
        record_detection_result(False)
        raise not_found_error()

    payload = extract_full_lsb(image)
    if not payload:
        if is_registered_original_image(image):
            record_detection_result(False)
            raise not_found_error()
        if should_run_frequency_fallbacks(image):
            dot_matrix_match = detect_dot_matrix_trace(image)
            if dot_matrix_match:
                record_detection_result(True)
                return dot_matrix_match
            if aligned_authenticated_detection_enabled:
                aligned_match = detect_aligned_authenticated_watermark(
                    image,
                    candidate_limit=aligned_candidate_limit,
                    budget_seconds=watermark_detection_budget_seconds,
                )
                if aligned_match:
                    record_detection_result(True)
                    return aligned_match
            if dense_watermark_fallback_enabled:
                if should_run_visual_match_fallback(image):
                    visual_match = detect_by_visual_match(image)
                    if visual_match:
                        record_detection_result(True)
                        return visual_match
                small_crop_match = detect_small_crop_trace(image)
                if small_crop_match:
                    record_detection_result(True)
                    return small_crop_match
                code_match = detect_watermark_code(image)
                if code_match:
                    record_detection_result(True)
                    return code_match
                robust_match = detect_robust_watermark(image)
                if robust_match:
                    record_detection_result(True)
                    return robust_match
        if visual_match_fallback_enabled:
            visual_match = detect_by_visual_match(image)
            if visual_match:
                record_detection_result(True)
                return visual_match
        residual_match = detect_by_residual_match(image)
        if residual_match:
            record_detection_result(True)
            return residual_match
        if visible_watermark_detection_enabled:
            fallback = detect_visible_copyright(image)
            if fallback:
                record_detection_result(True)
                return fallback
        payload = extract_block_lsb(image)
    if not payload:
        record_detection_result(False)
        raise not_found_error()
    current_records = records() if callable(records) else records
    matched = next((item for item in current_records if item.get("trace_id") == payload.get("trace_id")), None)
    record_detection_result(True)
    return with_evidence_fields({
        "id": payload.get("id"),
        "trace_id": payload.get("trace_id"),
        "evidence_uuid": payload.get("evidence_uuid"),
        "evidence_uuid_head": payload.get("evidence_uuid_head"),
        "evidence_uuid_tail": payload.get("evidence_uuid_tail"),
        "user_id": payload.get("user_id"),
        "mode": payload.get("mode"),
        "mode_label": payload.get("mode_label", mode_label(payload.get("mode", "dct"))),
        "created_at": payload.get("created_at"),
        "confidence": 98 if matched else 92,
        "phash_match": bool(matched),
        "status": "匹配" if matched else "检测到水印",
        "extracted_at": now_text(),
        "watermark_layers": matched.get("watermark_layers", watermark_layers) if matched else watermark_layers,
        "layer_scores": layer_scores_for_image(image, payload.get("trace_id")) if payload.get("trace_id") else {},
    }, matched)
