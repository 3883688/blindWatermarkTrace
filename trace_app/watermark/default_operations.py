from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from candidate_feature_index import (
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
from PIL import Image
from watermark_auth import auth_code_from_trace
from watermark_v4 import (
    V4Config,
    authentication_tag as v4_authentication_tag,
    embed_codeword as embed_v4_codeword,
    embed_pilot as embed_v4_pilot,
    encode_codeword as encode_v4_codeword,
)
from watermark_v4.detector import detect_v4
from watermark_v4.features import (
    extract_feature_index as extract_v4_feature_index,
    load_feature_index as load_v4_feature_index,
    save_feature_index as save_v4_feature_index,
)

from trace_app.config import (
    CODE_WATERMARK_VERSION,
    DEFAULT_ROBUST_WATERMARK_STRENGTH,
    DEFAULT_WATERMARK_AUTH_KEY,
    DOT_MATRIX_VERSION,
    ROBUST_WATERMARK_CODEC_V2,
    ROBUST_WATERMARK_CODEC_V3,
    ROBUST_WATERMARK_VERSION_V1,
    ROBUST_WATERMARK_VERSION_V2,
    ROBUST_WATERMARK_VERSION_V3,
    ROBUST_WATERMARK_VERSION_V4,
    SMALL_TRACE_VERSION,
    WATERMARK_LAYERS,
    Settings,
)
from trace_app.database.repositories import Repository
from trace_app.imaging import feature_matching, fingerprints, io, visible_mark
from trace_app.runtime import Runtime
from trace_app.watermark import detection, dot_matrix, frequency, lsb, robust, small_crop
from trace_app.watermark.service import WatermarkOperations


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _evidence_uuid_fields(evidence_uuid: str) -> dict[str, str]:
    normalized = evidence_uuid.replace("-", "").upper()
    return {
        "evidence_uuid": normalized,
        "evidence_uuid_head": normalized[:4],
        "evidence_uuid_tail": normalized[-4:],
    }


def _with_evidence_fields(
    result: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    if record:
        for key in ("evidence_uuid", "evidence_uuid_head", "evidence_uuid_tail"):
            if record.get(key) and not result.get(key):
                result[key] = record[key]
    return result


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").lower() in {"1", "true", "yes", "on", "启用"}


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _normalize_mode(raw: str) -> str:
    text = (raw or "").lower()
    if "lsb" in text or "空间" in raw or "最快" in raw:
        return "lsb"
    if "全部" in raw or "hybrid" in text or "最强" in raw:
        return "hybrid"
    if "dwt" in text:
        return "dwt"
    if "fft" in text:
        return "fft"
    return "dct"


def _mode_label(mode: str) -> str:
    return {
        "lsb": "仅空间域",
        "dct": "DCT + 空间域",
        "dwt": "DWT + 空间域",
        "fft": "FFT + 空间域",
        "hybrid": "全部算法",
    }.get(mode, "DCT + 空间域")


def build_default_operations(
    *,
    settings: Settings,
    repository: Repository,
    runtime: Runtime,
    state_value: Callable[[str], Any],
    ensure_directories: Callable[[], None],
) -> WatermarkOperations:
    records = repository.read_records
    generated = lambda: list(runtime.generated_trace_ids)
    save_feature = lambda image, record_id: feature_matching.save_record_feature_index(
        image,
        record_id,
        settings.data_dir,
        extract_feature_descriptors_fn=extract_feature_descriptors,
        save_feature_descriptors_fn=save_feature_descriptors,
    )
    save_feature_v4 = (
        lambda image, record_id: feature_matching.save_record_feature_index_v4(
            image,
            record_id,
            settings.data_dir,
            extract_v4_feature_index_fn=extract_v4_feature_index,
            save_v4_feature_index_fn=save_v4_feature_index,
        )
    )
    visual_consistency = lambda image, record: feature_matching.record_visual_consistency(
        image, record, settings.upload_dir
    )

    def candidates(current: list[dict[str, Any]]) -> Any:
        return detection.v4_candidate_records(
            records=current,
            data_dir=settings.data_dir,
            version_v4=ROBUST_WATERMARK_VERSION_V4,
            config_factory=V4Config,
            record_feature_index_path=feature_matching.record_feature_index_path,
            load_feature_index=load_v4_feature_index,
        )

    def detect_v4_current(
        image: Image.Image, current_candidates: Any, current: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        return detection.detect_v4_watermark(
            image,
            current_candidates,
            records=current,
            generated_trace_ids=generated(),
            version_v4=ROBUST_WATERMARK_VERSION_V4,
            config_factory=V4Config,
            candidate_records=lambda: candidates(current),
            detect=detect_v4,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
        )

    def aligned(image: Image.Image, current: list[dict[str, Any]], **kwargs: Any):
        rank = lambda subject, values: feature_matching.rank_aligned_candidates(
            subject,
            values,
            upload_dir=settings.upload_dir,
            data_dir=settings.data_dir,
            generated_trace_ids=generated(),
            save_record_feature_index_fn=save_feature,
            extract_feature_descriptors_fn=extract_feature_descriptors,
            load_feature_descriptors_fn=load_feature_descriptors,
        )
        return robust.detect_aligned_authenticated_watermark(
            image,
            kwargs.get("candidate_limit", 8),
            kwargs.get("budget_seconds", 5.0),
            records=current,
            rank_candidates=rank,
            align_query=lambda subject, record: feature_matching.align_query_to_record(
                subject, record, settings.upload_dir
            ),
            decode_v1=robust.decode_aligned_robust_trace,
            decode_v2=robust.decode_aligned_robust_trace_v2,
            decode_v3=robust.decode_aligned_robust_trace_v3,
            normalize_version=lambda value: robust.normalize_robust_watermark_version(
                value,
                version_v1=ROBUST_WATERMARK_VERSION_V1,
                version_v2=ROBUST_WATERMARK_VERSION_V2,
                version_v3=ROBUST_WATERMARK_VERSION_V3,
                version_v4=ROBUST_WATERMARK_VERSION_V4,
            ),
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            version_v1=ROBUST_WATERMARK_VERSION_V1,
            version_v2=ROBUST_WATERMARK_VERSION_V2,
            version_v3=ROBUST_WATERMARK_VERSION_V3,
            codec_v2=ROBUST_WATERMARK_CODEC_V2,
            codec_v3=ROBUST_WATERMARK_CODEC_V3,
            watermark_layers=WATERMARK_LAYERS,
        )

    return WatermarkOperations(
        ensure_dirs=ensure_directories,
        evidence_uuid_fields=_evidence_uuid_fields,
        load_upload_image=lambda file: io.load_upload_image(file),
        normalize_mode=_normalize_mode,
        apply_visible_copyright=visible_mark.apply_visible_copyright,
        parse_bool=_parse_bool,
        clamp_float=_clamp_float,
        now_text=_now_text,
        mode_label=_mode_label,
        fidelity_to_strength=lambda value: 1.0
        - _clamp_float(value, 0.75, 0.0, 1.0) * 0.72,
        robust_strength_to_scale=lambda value: _clamp_float(
            value,
            _clamp_float(DEFAULT_ROBUST_WATERMARK_STRENGTH, 1.0, 0.0, 2.0),
            0.0,
            2.0,
        ),
        normalize_robust_watermark_version=lambda value: robust.normalize_robust_watermark_version(
            value,
            version_v1=ROBUST_WATERMARK_VERSION_V1,
            version_v2=ROBUST_WATERMARK_VERSION_V2,
            version_v3=ROBUST_WATERMARK_VERSION_V3,
            version_v4=ROBUST_WATERMARK_VERSION_V4,
        ),
        v4_config=V4Config,
        v4_authentication_tag=v4_authentication_tag,
        auth_code_from_trace=auth_code_from_trace,
        state_value=state_value,
        small_crop_strength_to_scale=small_crop.small_crop_strength_to_scale,
        normalize_small_crop_density=small_crop.normalize_small_crop_density,
        embed_v4_pilot=embed_v4_pilot,
        encode_v4_codeword=encode_v4_codeword,
        embed_v4_codeword=embed_v4_codeword,
        embed_robust_watermark=robust.embed_robust_watermark,
        embed_robust_watermark_v2=robust.embed_robust_watermark_v2,
        embed_robust_watermark_v3=robust.embed_robust_watermark_v3,
        apply_frequency_layers=frequency.apply_frequency_layers,
        apply_code_layer=small_crop.apply_code_layer,
        apply_small_crop_trace_layer=small_crop.apply_small_crop_trace_layer,
        apply_dot_matrix_trace_layer=lambda image, trace_id, strength: dot_matrix.apply_dot_matrix_trace_layer(
            image,
            trace_id,
            strength,
            clamp_float_fn=_clamp_float,
            watermark_payload_from_trace_fn=robust.watermark_payload_from_trace,
        ),
        embed_lsb=lsb.embed_lsb,
        save_thumbnail=io.save_thumbnail,
        save_record_feature_index=save_feature,
        save_record_feature_index_v4=save_feature_v4,
        path_sha256=fingerprints.path_sha256,
        image_content_sha256=fingerprints.image_content_sha256,
        layer_scores_for_image=frequency.layer_scores_for_image,
        matched_file_fingerprint=lambda content, current: fingerprints.matched_file_fingerprint(
            content,
            read_records=lambda: current,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            watermark_layers=WATERMARK_LAYERS,
        ),
        load_image_from_bytes=io.load_image_from_bytes,
        load_image_from_url=lambda url: io.load_image_from_url(url, settings.upload_dir),
        v4_candidate_records=candidates,
        detect_v4_watermark=detect_v4_current,
        extract_full_lsb=lsb.extract_full_lsb,
        extract_block_lsb=lsb.extract_block_lsb,
        is_registered_original_image=lambda image, current: feature_matching.is_registered_original_image(
            image, records=current, upload_dir=settings.upload_dir
        ),
        should_run_frequency_fallbacks=detection.should_run_frequency_fallbacks,
        should_run_visual_match_fallback=lambda image, current: detection.should_run_visual_match_fallback(
            image, records=current
        ),
        detect_dot_matrix_trace=lambda image, current: dot_matrix.detect_dot_matrix_trace(
            image,
            dot_matrix.dot_matrix_candidate_records(
                current,
                watermark_payload_from_trace_fn=robust.watermark_payload_from_trace,
            ),
            hamming_distance_fn=robust.hamming_distance,
            code_crc16_fn=robust.code_crc16,
            now_text_fn=_now_text,
            with_evidence_fields_fn=_with_evidence_fields,
        ),
        detect_aligned_authenticated_watermark=aligned,
        detect_by_visual_match=lambda image, current: feature_matching.detect_by_visual_match(
            image,
            records=current,
            upload_dir=settings.upload_dir,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            watermark_layers=WATERMARK_LAYERS,
        ),
        detect_small_crop_trace=lambda image, current: small_crop.detect_small_crop_trace(
            image,
            current,
            generated(),
            watermark_payload_from_trace=robust.watermark_payload_from_trace,
            record_visual_consistency=visual_consistency,
            recover_payload_from_code=robust.recover_payload_from_code,
            hamming_distance=robust.hamming_distance,
            code_crc16=robust.code_crc16,
            now_text=_now_text,
            with_evidence_fields=_with_evidence_fields,
        ),
        detect_watermark_code=lambda image, current: small_crop.detect_watermark_code(
            image,
            current,
            generated(),
            watermark_payload_from_trace=robust.watermark_payload_from_trace,
            record_visual_consistency=visual_consistency,
            recover_payload_from_code=robust.recover_payload_from_code,
            hamming_distance=robust.hamming_distance,
            code_crc16=robust.code_crc16,
            now_text=_now_text,
            with_evidence_fields=_with_evidence_fields,
        ),
        detect_robust_watermark=lambda image, current: robust.detect_robust_watermark(
            image,
            records=robust.legacy_robust_candidate_records(current),
            extract_code=lambda subject, values: robust.extract_robust_code(
                subject, records=values
            ),
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            layer_scores_for_image=frequency.layer_scores_for_image,
            watermark_layers=WATERMARK_LAYERS,
        ),
        detect_by_residual_match=lambda image, current: feature_matching.detect_by_residual_match(
            image
        ),
        detect_visible_copyright=lambda image, current: visible_mark.detect_visible_copyright(
            image,
            records=current,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
        ),
        with_evidence_fields=_with_evidence_fields,
        watermark_detection_pipeline=detection.extract_watermark_from_image,
        default_watermark_auth_key=DEFAULT_WATERMARK_AUTH_KEY,
        robust_version_v2=ROBUST_WATERMARK_VERSION_V2,
        robust_version_v3=ROBUST_WATERMARK_VERSION_V3,
        robust_version_v4=ROBUST_WATERMARK_VERSION_V4,
        robust_codec_v2=ROBUST_WATERMARK_CODEC_V2,
        robust_codec_v3=ROBUST_WATERMARK_CODEC_V3,
        small_trace_version=SMALL_TRACE_VERSION,
        dot_matrix_version=DOT_MATRIX_VERSION,
        code_watermark_version=CODE_WATERMARK_VERSION,
        watermark_layers=WATERMARK_LAYERS,
    )
