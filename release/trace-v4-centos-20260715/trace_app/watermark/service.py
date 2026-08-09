from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, UploadFile
from PIL import Image

from trace_app.config import Settings
from trace_app.database.repositories import Repository
from trace_app.runtime import Runtime


@dataclass(frozen=True, slots=True)
class WatermarkOperations:
    ensure_dirs: Callable[[], None]
    evidence_uuid_fields: Callable[[str], dict[str, str]]
    load_upload_image: Callable[[UploadFile], Any]
    normalize_mode: Callable[[str], str]
    apply_visible_copyright: Callable[..., Image.Image]
    parse_bool: Callable[[Any], bool]
    clamp_float: Callable[..., float]
    now_text: Callable[[], str]
    mode_label: Callable[[str], str]
    fidelity_to_strength: Callable[[str], float]
    robust_strength_to_scale: Callable[[Any], float]
    normalize_robust_watermark_version: Callable[[Any], int]
    v4_config: Callable[[], Any]
    v4_authentication_tag: Callable[[str, str], bytes]
    auth_code_from_trace: Callable[[str, str], bytes]
    state_value: Callable[[str], Any]
    small_crop_strength_to_scale: Callable[[Any], float]
    normalize_small_crop_density: Callable[[str], str]
    embed_v4_pilot: Callable[..., Image.Image]
    encode_v4_codeword: Callable[[bytes], Any]
    embed_v4_codeword: Callable[..., Image.Image]
    embed_robust_watermark: Callable[..., Image.Image]
    embed_robust_watermark_v2: Callable[..., Image.Image]
    embed_robust_watermark_v3: Callable[..., Image.Image]
    apply_frequency_layers: Callable[[Image.Image, str], Image.Image]
    apply_code_layer: Callable[..., Image.Image]
    apply_small_crop_trace_layer: Callable[..., Image.Image]
    apply_dot_matrix_trace_layer: Callable[..., Image.Image]
    embed_lsb: Callable[[Image.Image, dict[str, Any]], Image.Image]
    save_thumbnail: Callable[[Image.Image, Path], None]
    save_record_feature_index: Callable[[Image.Image, str], str]
    save_record_feature_index_v4: Callable[[Image.Image, str], str]
    path_md5: Callable[[Path], str]
    path_sha256: Callable[[Path], str]
    image_content_sha256: Callable[[Image.Image], str]
    layer_scores_for_image: Callable[[Image.Image, str], dict[str, float]]
    matched_file_fingerprint: Callable[
        [bytes, list[dict[str, Any]]], dict[str, Any] | None
    ]
    load_image_from_bytes: Callable[[bytes], Image.Image]
    load_image_from_url: Callable[[str], Image.Image]
    v4_candidate_records: Callable[[list[dict[str, Any]]], Any]
    detect_v4_watermark: Callable[
        [Image.Image, Any, list[dict[str, Any]]], dict[str, Any] | None
    ]
    extract_full_lsb: Callable[[Image.Image], dict[str, Any] | None]
    extract_block_lsb: Callable[[Image.Image], dict[str, Any] | None]
    is_registered_original_image: Callable[
        [Image.Image, list[dict[str, Any]]], bool
    ]
    should_run_frequency_fallbacks: Callable[[Image.Image], bool]
    should_run_visual_match_fallback: Callable[
        [Image.Image, list[dict[str, Any]]], bool
    ]
    detect_dot_matrix_trace: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_aligned_authenticated_watermark: Callable[..., dict[str, Any] | None]
    detect_by_visual_match: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_small_crop_trace: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_watermark_code: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_robust_watermark: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_by_residual_match: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_visible_copyright: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    with_evidence_fields: Callable[..., dict[str, Any]]
    watermark_detection_pipeline: Callable[..., dict[str, Any]]
    default_watermark_auth_key: str
    robust_version_v2: int
    robust_version_v3: int
    robust_version_v4: int
    robust_codec_v2: str
    robust_codec_v3: str
    small_trace_version: int
    dot_matrix_version: int
    code_watermark_version: int
    watermark_layers: dict[str, bool]


class WatermarkService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        runtime: Runtime,
        operations: WatermarkOperations | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.runtime = runtime
        self.operations = operations

    def _operations(self) -> WatermarkOperations:
        if self.operations is None:
            raise RuntimeError("Watermark operations are not configured")
        return self.operations

    def _remember_generated_trace(self, trace_id: str) -> None:
        self.runtime.generated_trace_ids.insert(0, trace_id)
        del self.runtime.generated_trace_ids[24:]

    async def embed(
        self,
        *,
        file: UploadFile,
        user_id: str,
        mode: str,
        copyright_enabled: str,
        copyright_text: str,
        copyright_opacity: str,
        copyright_complexity: str,
        copyright_irregular_enabled: str,
        copyright_prominent_corner_enabled: str,
        fidelity_level: str,
        robust_watermark_strength: str,
        robust_watermark_version: str,
        small_crop_trace_enabled: str,
        small_crop_trace_strength: str,
        small_crop_trace_density: str,
        dot_matrix_trace_enabled: str,
        dot_matrix_trace_strength: str,
    ) -> dict[str, Any]:
        op = self._operations()
        op.ensure_dirs()
        image_id = uuid.uuid4().hex
        trace_id = f"TR-{uuid.uuid4().hex[:16].upper()}"
        evidence_uuid = uuid.uuid4().hex.upper()
        evidence_fields = op.evidence_uuid_fields(evidence_uuid)
        safe_name = Path(file.filename or "image.png").name
        original_path = self.settings.original_dir / f"{image_id}-{safe_name}"
        output_path = self.settings.watermarked_dir / f"{image_id}-watermarked.png"
        thumbnail_path = self.settings.thumbnail_dir / f"{image_id}-thumb.png"

        image = await op.load_upload_image(file)
        image.save(original_path)
        normalized_mode = op.normalize_mode(mode)
        visible = op.apply_visible_copyright(
            image,
            op.parse_bool(copyright_enabled),
            copyright_text,
            op.clamp_float(copyright_opacity, 0.16, 0.02, 0.90),
            copyright_complexity,
            op.parse_bool(copyright_irregular_enabled),
            op.parse_bool(copyright_prominent_corner_enabled),
        )
        created_at = op.now_text()
        payload = {
            "id": image_id,
            "trace_id": trace_id,
            **evidence_fields,
            "user_id": user_id,
            "mode": normalized_mode,
            "mode_label": op.mode_label(normalized_mode),
            "created_at": created_at,
        }
        strength_scale = op.fidelity_to_strength(fidelity_level)
        robust_strength = op.robust_strength_to_scale(robust_watermark_strength)
        robust_version = op.normalize_robust_watermark_version(
            robust_watermark_version
        )
        robust_auth_code = None
        v4_config = op.v4_config()
        if robust_version == op.robust_version_v4:
            try:
                robust_auth_code = op.v4_authentication_tag(
                    trace_id, op.default_watermark_auth_key
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        elif robust_version == op.robust_version_v3:
            try:
                robust_auth_code = op.auth_code_from_trace(
                    trace_id, op.default_watermark_auth_key
                )
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        small_crop_enabled = (
            op.state_value("small_crop_trace_default_enabled")
            if str(small_crop_trace_enabled or "").strip() == ""
            else op.parse_bool(small_crop_trace_enabled)
        )
        small_crop_strength = op.small_crop_strength_to_scale(
            small_crop_trace_strength
        )
        small_crop_density = op.normalize_small_crop_density(
            small_crop_trace_density
        )
        dot_matrix_enabled = op.parse_bool(dot_matrix_trace_enabled)
        dot_matrix_strength = op.clamp_float(
            dot_matrix_trace_strength, 0.85, 0.0, 1.0
        )
        if robust_version == op.robust_version_v4:
            small_crop_enabled = False
            dot_matrix_enabled = False
            try:
                watermarked = op.embed_v4_codeword(
                    op.embed_v4_pilot(visible, v4_config),
                    op.encode_v4_codeword(robust_auth_code),
                    v4_config,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            if robust_version == op.robust_version_v3:
                robust = op.embed_robust_watermark_v3(
                    visible, robust_auth_code, robust_strength
                )
            elif robust_version == op.robust_version_v2:
                robust = op.embed_robust_watermark_v2(
                    visible, trace_id, robust_strength
                )
            else:
                robust = op.embed_robust_watermark(
                    visible, trace_id, robust_strength
                )
            frequency_marked = op.apply_frequency_layers(robust, trace_id)
            code_marked = op.apply_code_layer(
                frequency_marked, trace_id, strength_scale
            )
            small_crop_marked = (
                op.apply_small_crop_trace_layer(
                    code_marked,
                    trace_id,
                    small_crop_strength,
                    small_crop_density,
                    strength_scale,
                )
                if small_crop_enabled
                else code_marked
            )
            dot_matrix_marked = (
                op.apply_dot_matrix_trace_layer(
                    small_crop_marked, trace_id, dot_matrix_strength
                )
                if dot_matrix_enabled
                else small_crop_marked
            )
            watermarked = op.embed_lsb(dot_matrix_marked, payload)
        watermarked.save(output_path, format="PNG")
        op.save_thumbnail(watermarked, thumbnail_path)
        feature_index_path = (
            op.save_record_feature_index_v4(watermarked, image_id)
            if robust_version == op.robust_version_v4
            else op.save_record_feature_index(watermarked, image_id)
        )
        original_file_md5 = op.path_md5(original_path)
        watermarked_file_md5 = op.path_md5(output_path)
        original_file_sha256 = op.path_sha256(original_path)
        watermarked_file_sha256 = op.path_sha256(output_path)
        original_image_sha256 = op.image_content_sha256(image)
        watermarked_image_sha256 = op.image_content_sha256(watermarked)

        record = {
            **payload,
            "name": safe_name,
            "image_width": image.width,
            "image_height": image.height,
            "size": f"{original_path.stat().st_size / 1024 / 1024:.1f} MB",
            "status": "保护中",
            "confidence": 98,
            "original_url": f"/uploads/originals/{original_path.name}",
            "download_url": f"/uploads/watermarked/{output_path.name}",
            "thumbnail_url": f"/uploads/thumbnails/{thumbnail_path.name}",
            "feature_index_path": feature_index_path,
            "original_file_md5": original_file_md5,
            "watermarked_file_md5": watermarked_file_md5,
            "original_file_sha256": original_file_sha256,
            "watermarked_file_sha256": watermarked_file_sha256,
            "original_image_sha256": original_image_sha256,
            "watermarked_image_sha256": watermarked_image_sha256,
            "copyright_enabled": op.parse_bool(copyright_enabled),
            "copyright_text": copyright_text.strip() or "© QQ:757675150",
            "copyright_opacity": op.clamp_float(
                copyright_opacity, 0.16, 0.02, 0.90
            ),
            "copyright_complexity": copyright_complexity,
            "copyright_irregular_enabled": op.parse_bool(
                copyright_irregular_enabled
            ),
            "copyright_prominent_corner_enabled": op.parse_bool(
                copyright_prominent_corner_enabled
            ),
            "fidelity_level": op.clamp_float(fidelity_level, 0.75, 0.0, 1.0),
            "watermark_strength_scale": round(strength_scale, 4),
            "robust_watermark_strength": round(robust_strength, 4),
            "robust_watermark_version": robust_version,
            "robust_watermark_codec": (
                v4_config.codec
                if robust_version == op.robust_version_v4
                else op.robust_codec_v3
                if robust_version == op.robust_version_v3
                else op.robust_codec_v2
                if robust_version == op.robust_version_v2
                else "legacy_robust_64"
            ),
            "robust_auth_code": robust_auth_code.hex()
            if robust_auth_code
            else None,
            "small_crop_trace_enabled": small_crop_enabled,
            "small_crop_trace_strength": small_crop_strength,
            "small_crop_trace_density": small_crop_density,
            "small_crop_trace_version": op.small_trace_version
            if small_crop_enabled
            else None,
            "dot_matrix_trace_enabled": dot_matrix_enabled,
            "dot_matrix_trace_strength": dot_matrix_strength,
            "dot_matrix_trace_version": op.dot_matrix_version
            if dot_matrix_enabled
            else None,
            "robust_watermark": True,
            "watermark_code_version": None
            if robust_version == op.robust_version_v4
            else op.code_watermark_version,
            "watermark_layers": (
                {"dct_authenticated": True, "fft_sync": True}
                if robust_version == op.robust_version_v4
                else op.watermark_layers
            ),
            "layer_scores": (
                {}
                if robust_version == op.robust_version_v4
                else op.layer_scores_for_image(watermarked, trace_id)
            ),
        }
        self.repository.add_record(record)
        self.repository.record_watermark_generation()
        self._remember_generated_trace(trace_id)
        return record

    def extract_image(
        self,
        image: Image.Image,
        records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        op = self._operations()
        current_records = (
            self.repository.read_records() if records is None else records
        )
        v4_candidates = op.v4_candidate_records(current_records)
        return op.watermark_detection_pipeline(
            image,
            records=current_records,
            v4_candidates=v4_candidates,
            detect_v4_watermark=lambda current_image, candidates: (
                op.detect_v4_watermark(
                    current_image, candidates, current_records
                )
            ),
            extract_full_lsb=op.extract_full_lsb,
            extract_block_lsb=op.extract_block_lsb,
            is_registered_original_image=lambda current_image: (
                op.is_registered_original_image(current_image, current_records)
            ),
            should_run_frequency_fallbacks=op.should_run_frequency_fallbacks,
            should_run_visual_match_fallback=lambda current_image: (
                op.should_run_visual_match_fallback(
                    current_image, current_records
                )
            ),
            detect_dot_matrix_trace=lambda current_image: (
                op.detect_dot_matrix_trace(current_image, current_records)
            ),
            detect_aligned_authenticated_watermark=lambda current_image, **kwargs: (
                op.detect_aligned_authenticated_watermark(
                    current_image, current_records, **kwargs
                )
            ),
            detect_by_visual_match=lambda current_image: (
                op.detect_by_visual_match(current_image, current_records)
            ),
            detect_small_crop_trace=lambda current_image: (
                op.detect_small_crop_trace(current_image, current_records)
            ),
            detect_watermark_code=lambda current_image: (
                op.detect_watermark_code(current_image, current_records)
            ),
            detect_robust_watermark=lambda current_image: (
                op.detect_robust_watermark(current_image, current_records)
            ),
            detect_by_residual_match=lambda current_image: (
                op.detect_by_residual_match(current_image, current_records)
            ),
            detect_visible_copyright=lambda current_image: (
                op.detect_visible_copyright(current_image, current_records)
            ),
            record_detection_result=self.repository.record_detection_result,
            with_evidence_fields=op.with_evidence_fields,
            now_text=op.now_text,
            mode_label=op.mode_label,
            layer_scores_for_image=op.layer_scores_for_image,
            not_found_error=lambda: HTTPException(
                status_code=404, detail="未检测到可识别的隐式水印"
            ),
            watermark_layers=op.watermark_layers,
            state_value=op.state_value,
        )

    async def extract_upload(self, file: UploadFile) -> dict[str, Any]:
        op = self._operations()
        content = await file.read()
        records = self.repository.read_records()
        fingerprint_match = op.matched_file_fingerprint(content, records)
        if fingerprint_match:
            self.repository.record_detection_result(True)
            return fingerprint_match
        image = op.load_image_from_bytes(content)
        return self.extract_image(image, records=records)

    def extract_url(self, url: str) -> dict[str, Any]:
        return self.extract_image(self._operations().load_image_from_url(url))
