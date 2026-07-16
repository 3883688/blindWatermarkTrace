import hashlib
import json
import inspect
import mimetypes
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFont
import cv2
import numpy as np
import pywt

cv2.setNumThreads(1)

from watermark_ecc import codeword_phase, decode_expected_codeword, encode_codeword, tile_phase
from watermark_auth import auth_code_from_trace, inverse_permutation, permuted_code_bits, phase_permutation
from candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
from database_store import DatabaseStore
from watermark_v4 import (
    V4Config,
    authentication_tag as v4_authentication_tag,
    embed_codeword as embed_v4_codeword,
    embed_pilot as embed_v4_pilot,
    encode_codeword as encode_v4_codeword,
)
from watermark_v4.features import (
    extract_feature_index as extract_v4_feature_index,
    load_feature_index as load_v4_feature_index,
    save_feature_index as save_v4_feature_index,
)
from watermark_v4.detector import V4Candidate, detect_v4

from trace_app.auth.service import AuthService
from trace_app.config import (
    ADMIN_PASS,
    ADMIN_USER,
    BASE_DIR,
    BLOCK_SIZE,
    BLOCK_STRIDE,
    CODE_CELL,
    CODE_CHANNEL_WEIGHTS,
    CODE_DELTA,
    CODE_GRID,
    CODE_PAYLOAD_BITS,
    CODE_PHYSICAL_BITS,
    CODE_TILE,
    CODE_WATERMARK_VERSION,
    DATA_DIR,
    DB_URL,
    DCT_BLOCK,
    DCT_DELTA,
    DEFAULT_ROBUST_WATERMARK_STRENGTH,
    DEFAULT_ROBUST_WATERMARK_VERSION,
    DEFAULT_ROLES,
    DEFAULT_WATERMARK_AUTH_KEY,
    DOT_MATRIX_CELL,
    DOT_MATRIX_CHANNEL_WEIGHTS,
    DOT_MATRIX_DELTA,
    DOT_MATRIX_GRID,
    DOT_MATRIX_TILE,
    DOT_MATRIX_VERSION,
    DWT_DELTA,
    FFT_DELTA,
    FEATURE_MATCH_MIN_GOOD,
    FEATURE_RECENT_BACKFILL,
    FEATURE_RECENT_RESERVE,
    MAGIC,
    MENU_LABELS,
    ORIGINAL_DIR,
    ROBUST_BITS,
    ROBUST_CELL,
    ROBUST_CHANNEL,
    ROBUST_DELTA,
    ROBUST_GRID,
    ROBUST_MAGIC,
    ROBUST_TILE,
    ROBUST_WATERMARK_CODEC_V2,
    ROBUST_WATERMARK_CODEC_V3,
    ROBUST_WATERMARK_VERSION_V1,
    ROBUST_WATERMARK_VERSION_V2,
    ROBUST_WATERMARK_VERSION_V3,
    ROBUST_WATERMARK_VERSION_V4,
    SMALL_TRACE_CHANNEL_WEIGHTS,
    SMALL_TRACE_DELTA,
    SMALL_TRACE_SHORT_BITS,
    SMALL_TRACE_TILE,
    SMALL_TRACE_VERSION,
    THUMBNAIL_DIR,
    UPLOAD_DIR,
    WATERMARK_LAYERS,
    WATERMARKED_DIR,
    settings,
)
from trace_app.database.connection import (
    create_runtime,
    seed_database_defaults as seed_runtime_defaults,
)
from trace_app.database.repositories import Repository
from trace_app.imaging import feature_matching as imaging_feature_matching
from trace_app.imaging import fingerprints as imaging_fingerprints
from trace_app.imaging import io as imaging_io
from trace_app.imaging import visible_mark as imaging_visible_mark
from trace_app.watermark import dot_matrix as watermark_dot_matrix
from trace_app.watermark import detection as watermark_detection
from trace_app.watermark import frequency as watermark_frequency
from trace_app.watermark import lsb as watermark_lsb
from trace_app.watermark import robust as watermark_robust
from trace_app.watermark import small_crop as watermark_small_crop
from trace_app.watermark.service import WatermarkOperations, WatermarkService
from trace_app.watermark.frequency import (
    apply_dct_layer,
    apply_dwt_layer,
    apply_fft_layer,
    dct_layer_score,
    dwt_layer_score,
    fft_layer_score,
    fft_pattern,
    layer_seed,
    pseudo_random_signs,
    robust_pattern,
)
from trace_app.watermark.lsb import (
    PayloadTooLargeError,
    WatermarkNotFoundError,
    bits_from_bytes,
    bytes_from_bits,
    decode_bits_from_pixels,
    embed_block_lsb,
    extract_block_lsb,
    extract_full_lsb,
    iter_block_origins,
    lsb_bits_from_pixels,
    packet_from_payload,
    valid_watermark_payload,
)
from trace_app.watermark.small_crop import (
    apply_code_layer,
    code_cell_carriers,
    code_from_score_vector,
    code_marker_pattern,
    code_scan_grid,
    code_scan_signal_grid,
    code_tile_carriers,
    code_trace_pattern,
    decode_code_tile,
    decode_code_tile_scores,
    decode_code_tile_signal,
    decode_small_trace_code_scores,
    decode_small_trace_short_scores,
    decode_small_trace_signal,
    iter_aligned_small_trace_tiles,
    iter_code_scan_offsets,
    iter_small_trace_windows,
    match_small_trace_code,
    normalize_carrier,
    normalize_small_crop_density,
    record_from_short_code_match,
    short_code_from_scores,
    small_crop_density_offsets,
    small_crop_strength_to_scale,
    small_trace_code_carriers,
    small_trace_marker_pattern,
    small_trace_pattern,
    small_trace_short_bits,
    small_trace_short_carriers,
    small_trace_short_code,
    trace_tile_agreement,
)

RUNNING_PYTEST = "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None
DB_ENABLED = not RUNNING_PYTEST

app = FastAPI(title=settings.app_name)
app.state.generated_trace_ids = []


def ensure_dirs() -> None:
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


runtime = create_runtime(settings, enabled=False)
repository = Repository(runtime.store, ensure_dirs=ensure_dirs)
db_engine = runtime.engine
db_store = runtime.store
db_error = runtime.db_error


def require_store() -> DatabaseStore:
    return repository.store


def database_ready() -> bool:
    return runtime.store is not None


def seed_database_defaults(store: DatabaseStore) -> None:
    seed_runtime_defaults(store, settings)


def initialize_database() -> None:
    global db_engine, db_error, db_store, repository, runtime
    if not DB_ENABLED:
        return
    try:
        runtime = create_runtime(settings)
    except RuntimeError as exc:
        failed_runtime = getattr(exc, "runtime", None)
        if failed_runtime is not None:
            runtime = failed_runtime
            repository = Repository(runtime.store, ensure_dirs=ensure_dirs)
            db_engine = runtime.engine
            db_store = runtime.store
            db_error = runtime.db_error
        raise
    repository = Repository(runtime.store, ensure_dirs=ensure_dirs)
    db_engine = runtime.engine
    db_store = runtime.store
    db_error = runtime.db_error


def db_clear_all() -> None:
    repository.db_clear_all()


ensure_dirs()
initialize_database()
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("font/ttf", ".ttf")
app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def masked_db_url() -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:****@", DB_URL)


def read_records() -> list[dict[str, Any]]:
    return repository.read_records()


def write_records(records: list[dict[str, Any]]) -> None:
    repository.write_records(records)


def add_record(record: dict[str, Any]) -> None:
    repository.add_record(record)


def read_detection_stats() -> dict[str, int]:
    return repository.read_detection_stats()


def write_detection_stats(stats: dict[str, int]) -> None:
    repository.write_detection_stats(stats)


def record_detection_result(success: bool) -> None:
    repository.record_detection_result(success)


def is_today_record(record: dict[str, Any]) -> bool:
    return repository.is_today_record(record)


def read_watermark_stats() -> dict[str, dict[str, int]]:
    return repository.read_watermark_stats()


def write_watermark_stats(stats: dict[str, Any]) -> None:
    repository.write_watermark_stats(stats)


def read_roles() -> dict[str, Any]:
    return repository.read_roles()


def read_users() -> dict[str, Any]:
    return repository.read_users()


def get_auth_service() -> AuthService:
    return AuthService(repository)


def _records_call_mode(callback) -> str:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return "legacy"
    records_parameter = parameters.get("records")
    if (
        records_parameter is not None
        and records_parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    ):
        return "positional"
    if records_parameter is not None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return "keyword"
    return "legacy"


def _record_aware_callback(callback):
    mode = _records_call_mode(callback)
    if mode == "positional":
        return lambda subject, records, **kwargs: callback(
            subject, records, **kwargs
        )
    if mode == "keyword":
        return lambda subject, records, **kwargs: callback(
            subject, records=records, **kwargs
        )
    return lambda subject, records, **kwargs: callback(subject, **kwargs)


def _record_candidate_callback(callback):
    mode = _records_call_mode(callback)
    if mode == "positional":
        return lambda records: callback(records)
    if mode == "keyword":
        return lambda records: callback(records=records)
    return lambda records: callback()


def _v4_record_aware_callback(callback):
    mode = _records_call_mode(callback)
    if mode == "positional":
        return lambda image, candidates, records: callback(
            image, candidates, records
        )
    if mode == "keyword":
        return lambda image, candidates, records: callback(
            image, candidates, records=records
        )
    return lambda image, candidates, records: callback(image, candidates)


def get_watermark_service() -> WatermarkService:
    generated_trace_ids = getattr(app.state, "generated_trace_ids", [])
    runtime.generated_trace_ids = generated_trace_ids
    effective_settings = settings
    if settings.upload_dir != Path(UPLOAD_DIR) or settings.data_dir != Path(DATA_DIR):
        effective_settings = replace(
            settings,
            upload_dir=Path(UPLOAD_DIR),
            data_dir=Path(DATA_DIR),
        )
    return WatermarkService(
        settings=effective_settings,
        repository=repository,
        runtime=runtime,
        operations=WatermarkOperations(
            ensure_dirs=ensure_dirs,
            evidence_uuid_fields=evidence_uuid_fields,
            load_upload_image=load_upload_image,
            normalize_mode=normalize_mode,
            apply_visible_copyright=apply_visible_copyright,
            parse_bool=parse_bool,
            clamp_float=clamp_float,
            now_text=now_text,
            mode_label=mode_label,
            fidelity_to_strength=fidelity_to_strength,
            robust_strength_to_scale=robust_strength_to_scale,
            normalize_robust_watermark_version=normalize_robust_watermark_version,
            v4_config=V4Config,
            v4_authentication_tag=v4_authentication_tag,
            auth_code_from_trace=auth_code_from_trace,
            state_value=lambda name: getattr(app.state, name),
            small_crop_strength_to_scale=small_crop_strength_to_scale,
            normalize_small_crop_density=normalize_small_crop_density,
            embed_v4_pilot=embed_v4_pilot,
            encode_v4_codeword=encode_v4_codeword,
            embed_v4_codeword=embed_v4_codeword,
            embed_robust_watermark=embed_robust_watermark,
            embed_robust_watermark_v2=embed_robust_watermark_v2,
            embed_robust_watermark_v3=embed_robust_watermark_v3,
            apply_frequency_layers=apply_frequency_layers,
            apply_code_layer=apply_code_layer,
            apply_small_crop_trace_layer=apply_small_crop_trace_layer,
            apply_dot_matrix_trace_layer=apply_dot_matrix_trace_layer,
            embed_lsb=embed_lsb,
            save_thumbnail=save_thumbnail,
            save_record_feature_index=save_record_feature_index,
            save_record_feature_index_v4=save_record_feature_index_v4,
            path_sha256=path_sha256,
            image_content_sha256=image_content_sha256,
            layer_scores_for_image=layer_scores_for_image,
            matched_file_fingerprint=_record_aware_callback(
                matched_file_fingerprint
            ),
            load_image_from_bytes=load_image_from_bytes,
            load_image_from_url=load_image_from_url,
            v4_candidate_records=_record_candidate_callback(v4_candidate_records),
            detect_v4_watermark=_v4_record_aware_callback(detect_v4_watermark),
            extract_full_lsb=extract_full_lsb,
            extract_block_lsb=extract_block_lsb,
            is_registered_original_image=_record_aware_callback(
                is_registered_original_image
            ),
            should_run_frequency_fallbacks=should_run_frequency_fallbacks,
            should_run_visual_match_fallback=_record_aware_callback(
                should_run_visual_match_fallback
            ),
            detect_dot_matrix_trace=_record_aware_callback(detect_dot_matrix_trace),
            detect_aligned_authenticated_watermark=_record_aware_callback(
                detect_aligned_authenticated_watermark
            ),
            detect_by_visual_match=_record_aware_callback(detect_by_visual_match),
            detect_small_crop_trace=_record_aware_callback(detect_small_crop_trace),
            detect_watermark_code=_record_aware_callback(detect_watermark_code),
            detect_robust_watermark=_record_aware_callback(
                detect_robust_watermark
            ),
            detect_by_residual_match=_record_aware_callback(
                detect_by_residual_match
            ),
            detect_visible_copyright=_record_aware_callback(
                detect_visible_copyright
            ),
            with_evidence_fields=with_evidence_fields,
            watermark_detection_pipeline=watermark_detection.extract_watermark_from_image,
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
        ),
    )


def public_users(users: dict[str, Any]) -> dict[str, dict[str, str]]:
    return get_auth_service().public_users(users)


def allowed_menu_keys(menus: Any) -> list[str]:
    return get_auth_service().allowed_menu_keys(menus)


def role_for_username(username: str) -> str:
    return get_auth_service().role_for_username(username)


def record_watermark_generation() -> None:
    repository.record_watermark_generation()


def today_watermark_count(records: list[dict[str, Any]]) -> int:
    return repository.today_watermark_count(
        records, read_watermark_stats(), is_today=is_today_record
    )


def evidence_uuid_fields(evidence_uuid: str) -> dict[str, str]:
    normalized = evidence_uuid.replace("-", "").upper()
    return {
        "evidence_uuid": normalized,
        "evidence_uuid_head": normalized[:4],
        "evidence_uuid_tail": normalized[-4:],
    }


def with_evidence_fields(result: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return result
    for key in ("evidence_uuid", "evidence_uuid_head", "evidence_uuid_tail"):
        if record.get(key) and not result.get(key):
            result[key] = record.get(key)
    return result


def _robust_config() -> watermark_robust.RobustConfig:
    return watermark_robust.RobustConfig(
        robust_bits=ROBUST_BITS,
        robust_cell=ROBUST_CELL,
        robust_channel=ROBUST_CHANNEL,
        robust_delta=ROBUST_DELTA,
        robust_grid=ROBUST_GRID,
        robust_magic=ROBUST_MAGIC,
        robust_tile=ROBUST_TILE,
        code_payload_bits=CODE_PAYLOAD_BITS,
        code_physical_bits=CODE_PHYSICAL_BITS,
    )


def _robust_dependencies() -> watermark_robust.RobustDependencies:
    return watermark_robust.RobustDependencies(
        robust_code_from_trace=robust_code_from_trace,
        robust_bits_from_code=robust_bits_from_code,
        robust_payload_bytes=robust_payload_bytes,
        code_crc16=code_crc16,
        watermark_payload_from_trace=watermark_payload_from_trace,
        hamming_distance=hamming_distance,
        iter_robust_tiles=iter_robust_tiles,
        robust_pattern=robust_pattern,
        encode_codeword=encode_codeword,
        codeword_phase=codeword_phase,
        tile_phase=tile_phase,
        permuted_code_bits=permuted_code_bits,
        phase_permutation=phase_permutation,
        extract_robust_from_grid=extract_robust_from_grid,
        scores_to_byte=_scores_to_byte,
        phase_scores_to_codeword=_phase_scores_to_codeword,
        decode_expected_codeword=decode_expected_codeword,
        record_v3_auth_code=_record_v3_auth_code,
    )


def robust_code_from_trace(trace_id: str) -> int:
    return watermark_robust.robust_code_from_trace(trace_id, config=_robust_config())


def robust_bits_from_code(code: int) -> list[int]:
    return watermark_robust.robust_bits_from_code(code, config=_robust_config())


def robust_payload_bytes(trace_id: str) -> bytes:
    return watermark_robust.robust_payload_bytes(
        trace_id, config=_robust_config(), dependencies=_robust_dependencies()
    )


def code_crc16(value: int) -> int:
    return watermark_robust.code_crc16(value)


def watermark_payload_from_trace(trace_id: str) -> int:
    return watermark_robust.watermark_payload_from_trace(
        trace_id, config=_robust_config(), dependencies=_robust_dependencies()
    )


def watermark_bits_from_trace(trace_id: str) -> list[int]:
    return watermark_robust.watermark_bits_from_trace(
        trace_id, config=_robust_config(), dependencies=_robust_dependencies()
    )






def recover_payload_from_code(code: int) -> tuple[int, int]:
    return watermark_robust.recover_payload_from_code(code, config=_robust_config())


























def fidelity_to_strength(value: str) -> float:
    fidelity = clamp_float(value, 0.75, 0.0, 1.0)
    return 1.0 - fidelity * 0.72




def robust_strength_to_scale(value: str | float | None) -> float:
    default = clamp_float(DEFAULT_ROBUST_WATERMARK_STRENGTH, 1.0, 0.0, 2.0)
    return clamp_float(value, default, 0.0, 2.0)


def normalize_robust_watermark_version(value: str | int | None) -> int:
    return watermark_robust.normalize_robust_watermark_version(
        value,
        version_v1=ROBUST_WATERMARK_VERSION_V1,
        version_v2=ROBUST_WATERMARK_VERSION_V2,
        version_v3=ROBUST_WATERMARK_VERSION_V3,
        version_v4=ROBUST_WATERMARK_VERSION_V4,
    )














































































def robust_code_to_trace(code: int) -> str | None:
    return watermark_robust.robust_code_to_trace(
        code,
        records=read_records,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def hamming_distance(left: int, right: int) -> int:
    return watermark_robust.hamming_distance(left, right)


def robust_code_to_trace_fuzzy(code: int, max_errors: int = 18) -> tuple[str | None, int]:
    return watermark_robust.robust_code_to_trace_fuzzy(
        code,
        max_errors,
        records=read_records,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def robust_candidate_records(
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return watermark_robust.robust_candidate_records(
        read_records() if records is None else records
    )


def legacy_robust_candidate_records(
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return watermark_robust.legacy_robust_candidate_records(
        read_records() if records is None else records,
        normalize_version=normalize_robust_watermark_version,
        version_v1=ROBUST_WATERMARK_VERSION_V1,
    )


def iter_robust_tiles(width: int, height: int):
    yield from watermark_robust.iter_robust_tiles(
        width, height, config=_robust_config()
    )


def embed_robust_watermark(image: Image.Image, trace_id: str, strength_scale: float = 1.0) -> Image.Image:
    return watermark_robust.embed_robust_watermark(
        image,
        trace_id,
        strength_scale,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def embed_robust_watermark_v2(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
) -> Image.Image:
    return watermark_robust.embed_robust_watermark_v2(
        image,
        trace_id,
        strength_scale,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def embed_robust_watermark_v3(
    image: Image.Image,
    auth_code: bytes,
    strength_scale: float = 1.0,
) -> Image.Image:
    return watermark_robust.embed_robust_watermark_v3(
        image,
        auth_code,
        strength_scale,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def extract_robust_from_grid(arr: np.ndarray, cell: int, offset_x: int, offset_y: int) -> tuple[int | None, float, int]:
    return watermark_robust.extract_robust_from_grid(
        arr,
        cell,
        offset_x,
        offset_y,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def decode_aligned_robust_trace(
    alignment: dict[str, Any],
    record: dict[str, Any],
    max_errors: int = 4,
) -> dict[str, Any] | None:
    return watermark_robust.decode_aligned_robust_trace(
        alignment,
        record,
        max_errors,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def _scores_to_byte(scores: np.ndarray) -> tuple[int, float]:
    return watermark_robust._scores_to_byte(scores)


def _phase_scores_to_codeword(
    phase_scores: np.ndarray,
    phase_counts: list[int],
) -> tuple[bytes, list[float]]:
    return watermark_robust._phase_scores_to_codeword(
        phase_scores,
        phase_counts,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def decode_aligned_robust_trace_v2(
    alignment: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any] | None:
    return watermark_robust.decode_aligned_robust_trace_v2(
        alignment,
        record,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def _record_v3_auth_code(record: dict[str, Any]) -> bytes | None:
    return watermark_robust._record_v3_auth_code(record)


def decode_aligned_robust_trace_v3(
    alignment: dict[str, Any],
    record: dict[str, Any],
    max_errors: int = 8,
) -> dict[str, Any] | None:
    return watermark_robust.decode_aligned_robust_trace_v3(
        alignment,
        record,
        max_errors,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def save_record_feature_index(image: Image.Image, record_id: str) -> str:
    return imaging_feature_matching.save_record_feature_index(
        image,
        record_id,
        DATA_DIR,
        extract_feature_descriptors_fn=extract_feature_descriptors,
        save_feature_descriptors_fn=save_feature_descriptors,
    )


def save_record_feature_index_v4(image: Image.Image, record_id: str) -> str:
    return imaging_feature_matching.save_record_feature_index_v4(
        image,
        record_id,
        DATA_DIR,
        extract_v4_feature_index_fn=extract_v4_feature_index,
        save_v4_feature_index_fn=save_v4_feature_index,
    )


def record_feature_index_path(record: dict[str, Any]) -> Path | None:
    return imaging_feature_matching.record_feature_index_path(record, DATA_DIR)


def v4_candidate_records(
    records: list[dict[str, Any]] | None = None,
) -> tuple[V4Candidate, ...]:
    return watermark_detection.v4_candidate_records(
        records=read_records() if records is None else records,
        data_dir=DATA_DIR,
        version_v4=ROBUST_WATERMARK_VERSION_V4,
        config_factory=V4Config,
        record_feature_index_path=lambda record, data_dir: record_feature_index_path(record),
        load_feature_index=load_v4_feature_index,
    )


def detect_v4_watermark(
    image: Image.Image,
    candidates: tuple[V4Candidate, ...] | None = None,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return watermark_detection.detect_v4_watermark(
        image,
        candidates,
        records=read_records if records is None else records,
        generated_trace_ids=list(getattr(app.state, "generated_trace_ids", [])),
        version_v4=ROBUST_WATERMARK_VERSION_V4,
        config_factory=V4Config,
        candidate_records=(
            v4_candidate_records
            if records is None
            else lambda: v4_candidate_records(records)
        ),
        detect=detect_v4,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
    )


def rank_aligned_candidates(
    image: Image.Image,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return imaging_feature_matching.rank_aligned_candidates(
        image,
        records,
        upload_dir=UPLOAD_DIR,
        data_dir=DATA_DIR,
        generated_trace_ids=list(getattr(app.state, "generated_trace_ids", [])),
        feature_match_min_good=FEATURE_MATCH_MIN_GOOD,
        feature_recent_backfill=FEATURE_RECENT_BACKFILL,
        feature_recent_reserve=FEATURE_RECENT_RESERVE,
        record_feature_index_path_fn=record_feature_index_path,
        save_record_feature_index_fn=save_record_feature_index,
        extract_feature_descriptors_fn=extract_feature_descriptors,
        load_feature_descriptors_fn=load_feature_descriptors,
        descriptor_match_score_fn=descriptor_match_score,
    )


def detect_aligned_authenticated_watermark(
    image: Image.Image,
    records: list[dict[str, Any]] | None = None,
    candidate_limit: int = 8,
    budget_seconds: float = 5.0,
) -> dict[str, Any] | None:
    current_records = records if records is not None else robust_candidate_records
    return watermark_robust.detect_aligned_authenticated_watermark(
        image,
        candidate_limit,
        budget_seconds,
        records=current_records,
        rank_candidates=rank_aligned_candidates,
        align_query=align_query_to_record,
        decode_v1=decode_aligned_robust_trace,
        decode_v2=decode_aligned_robust_trace_v2,
        decode_v3=decode_aligned_robust_trace_v3,
        normalize_version=normalize_robust_watermark_version,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
        version_v1=ROBUST_WATERMARK_VERSION_V1,
        version_v2=ROBUST_WATERMARK_VERSION_V2,
        version_v3=ROBUST_WATERMARK_VERSION_V3,
        codec_v2=ROBUST_WATERMARK_CODEC_V2,
        codec_v3=ROBUST_WATERMARK_CODEC_V3,
        watermark_layers=WATERMARK_LAYERS,
        perf_counter=time.perf_counter,
    )


def extract_robust_code(image: Image.Image, records: list[dict[str, Any]] | None = None) -> tuple[str | None, float, int]:
    current_records = records if records is not None else legacy_robust_candidate_records()
    return watermark_robust.extract_robust_code(
        image,
        records=current_records,
        config=_robust_config(),
        dependencies=_robust_dependencies(),
    )


def detect_robust_watermark(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    current_records = legacy_robust_candidate_records(records)
    return watermark_robust.detect_robust_watermark(
        image,
        records=current_records,
        extract_code=lambda current_image, current_records: extract_robust_code(
            current_image, current_records
        ),
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
        layer_scores_for_image=layer_scores_for_image,
        watermark_layers=WATERMARK_LAYERS,
    )


def mode_label(mode: str) -> str:
    labels = {
        "lsb": "仅空间域",
        "dct": "DCT + 空间域",
        "dwt": "DWT + 空间域",
        "fft": "FFT + 空间域",
        "hybrid": "全部算法",
    }
    return labels.get(mode, "DCT + 空间域")


def normalize_mode(raw: str) -> str:
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


def parse_bool(raw: str | bool | None) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").lower() in {"1", "true", "yes", "on", "启用"}


def env_bool(name: str, default: str = "false", legacy_name: str | None = None) -> bool:
    value = os.getenv(name)
    if value is None and legacy_name:
        value = os.getenv(legacy_name)
    return parse_bool(default if value is None else value)


app.state.visible_watermark_detection_enabled = env_bool(
    "ENABLE_VISIBLE_WATERMARK_DETECTION", "false", "VISIBLE_WATERMARK_DETECTION_ENABLED"
)
app.state.visual_match_fallback_enabled = env_bool(
    "ENABLE_VISUAL_MATCH_FALLBACK", "false", "VISUAL_MATCH_FALLBACK_ENABLED"
)
app.state.small_crop_trace_default_enabled = env_bool(
    "ENABLE_SMALL_CROP_TRACE_REDUNDANCY", "true"
)
app.state.aligned_authenticated_detection_enabled = env_bool(
    "ENABLE_ALIGNED_AUTHENTICATED_DETECTION", "true"
)
app.state.dense_watermark_fallback_enabled = env_bool(
    "ENABLE_DENSE_WATERMARK_FALLBACK", "false"
)
try:
    app.state.aligned_candidate_limit = max(1, min(32, int(os.getenv("ALIGNED_CANDIDATE_LIMIT", "8"))))
except ValueError:
    app.state.aligned_candidate_limit = 8


def clamp_float(value: str | float | None, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


app.state.watermark_detection_budget_seconds = clamp_float(
    os.getenv("WATERMARK_DETECTION_BUDGET_SECONDS", "5"),
    5.0,
    0.1,
    60.0,
)




























def should_run_frequency_fallbacks(image: Image.Image) -> bool:
    return watermark_detection.should_run_frequency_fallbacks(image)


def should_run_visual_match_fallback(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> bool:
    return watermark_detection.should_run_visual_match_fallback(
        image, records=read_records() if records is None else records
    )


def detect_visible_copyright(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return imaging_visible_mark.detect_visible_copyright(
        image,
        records=read_records() if records is None else records,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
    )


def image_to_cv_gray(image: Image.Image, max_side: int = 1200):
    return imaging_feature_matching.image_to_cv_gray(image, max_side)


def record_visual_consistency(image: Image.Image, record: dict[str, Any]) -> tuple[bool, int, float, float]:
    return imaging_feature_matching.record_visual_consistency(
        image,
        record,
        UPLOAD_DIR,
        image_to_cv_gray_fn=image_to_cv_gray,
        feature_match_score_fn=feature_match_score,
        robust_residual_score_fn=robust_residual_score,
    )


dot_matrix_position = watermark_dot_matrix.dot_matrix_position
dot_matrix_score_tile = watermark_dot_matrix.dot_matrix_score_tile


def dot_matrix_bits_from_trace(trace_id: str) -> list[int]:
    return watermark_dot_matrix.dot_matrix_bits_from_trace(
        trace_id,
        watermark_payload_from_trace_fn=watermark_payload_from_trace,
    )


def dot_matrix_candidate_records(
    records: list[dict[str, Any]] | None = None,
) -> list[tuple[str, int, dict[str, Any]]]:
    return watermark_dot_matrix.dot_matrix_candidate_records(
        read_records() if records is None else records,
        watermark_payload_from_trace_fn=watermark_payload_from_trace,
    )


def apply_dot_matrix_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 1.0,
) -> Image.Image:
    return watermark_dot_matrix.apply_dot_matrix_trace_layer(
        image,
        trace_id,
        strength,
        clamp_float_fn=clamp_float,
        watermark_payload_from_trace_fn=watermark_payload_from_trace,
    )


def detect_dot_matrix_trace(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return watermark_dot_matrix.detect_dot_matrix_trace(
        image,
        dot_matrix_candidate_records(records),
        hamming_distance_fn=hamming_distance,
        code_crc16_fn=code_crc16,
        now_text_fn=now_text,
        with_evidence_fields_fn=with_evidence_fields,
    )


def apply_small_crop_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 0.25,
    density: str = "low",
    fidelity_scale: float = 1.0,
) -> Image.Image:
    return watermark_small_crop.apply_small_crop_trace_layer(
        image,
        trace_id,
        strength,
        density,
        fidelity_scale,
        watermark_bits_from_trace_fn=watermark_bits_from_trace,
        watermark_payload_from_trace_fn=watermark_payload_from_trace,
    )


def apply_code_layer_shifted(image: Image.Image, trace_id: str) -> Image.Image:
    return watermark_small_crop.apply_code_layer_shifted(
        image,
        trace_id,
        apply_code_layer_fn=apply_code_layer,
    )


def detect_small_crop_trace(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return watermark_small_crop.detect_small_crop_trace(
        image,
        read_records() if records is None else records,
        list(getattr(app.state, "generated_trace_ids", [])),
        watermark_payload_from_trace=watermark_payload_from_trace,
        record_visual_consistency=record_visual_consistency,
        recover_payload_from_code=recover_payload_from_code,
        hamming_distance=hamming_distance,
        code_crc16=code_crc16,
        now_text=now_text,
        with_evidence_fields=with_evidence_fields,
    )


def detect_watermark_code(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return watermark_small_crop.detect_watermark_code(
        image,
        read_records() if records is None else records,
        list(getattr(app.state, "generated_trace_ids", [])),
        watermark_payload_from_trace=watermark_payload_from_trace,
        record_visual_consistency=record_visual_consistency,
        recover_payload_from_code=recover_payload_from_code,
        hamming_distance=hamming_distance,
        code_crc16=code_crc16,
        now_text=now_text,
        with_evidence_fields=with_evidence_fields,
    )


def apply_frequency_layers(image: Image.Image, trace_id: str) -> Image.Image:
    return watermark_frequency.apply_frequency_layers(
        image,
        trace_id,
        apply_dct_layer_fn=apply_dct_layer,
        apply_dwt_layer_fn=apply_dwt_layer,
        apply_fft_layer_fn=apply_fft_layer,
    )


def layer_scores_for_image(image: Image.Image, trace_id: str) -> dict[str, float]:
    return watermark_frequency.layer_scores_for_image(
        image,
        trace_id,
        dct_layer_score_fn=dct_layer_score,
        dwt_layer_score_fn=dwt_layer_score,
        fft_layer_score_fn=fft_layer_score,
    )


def embed_lsb(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    return watermark_lsb.embed_lsb(
        image,
        payload,
        embed_block_lsb_fn=embed_block_lsb,
        write_packet_to_pixels_fn=write_packet_to_pixels,
    )


def write_packet_to_pixels(
    pixels: list[tuple[int, int, int]],
    packet: bytes,
) -> list[tuple[int, int, int]]:
    try:
        return watermark_lsb.write_packet_to_pixels(pixels, packet)
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def extract_lsb(image: Image.Image) -> dict[str, Any]:
    try:
        return watermark_lsb.extract_lsb(
            image,
            extract_full_lsb_fn=extract_full_lsb,
            extract_block_lsb_fn=extract_block_lsb,
        )
    except WatermarkNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def residual_candidate_evidence(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return imaging_feature_matching.residual_candidate_evidence(
        image,
        records=read_records() if records is None else records,
        record_visual_consistency_fn=record_visual_consistency,
    )


def detect_by_residual_match(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return imaging_feature_matching.detect_by_residual_match(image)


def feature_match_score(query_gray, target_gray) -> tuple[int, float]:
    return imaging_feature_matching.feature_match_score(query_gray, target_gray)


def feature_match_homography(query_gray, target_gray):
    return imaging_feature_matching.feature_match_homography(query_gray, target_gray)


def align_query_to_record(image: Image.Image, record: dict[str, Any]) -> dict[str, Any] | None:
    return imaging_feature_matching.align_query_to_record(
        image,
        record,
        UPLOAD_DIR,
        resize_for_residual_fn=resize_for_residual,
        feature_match_homography_fn=feature_match_homography,
    )


def resize_for_residual(image: Image.Image, max_side: int = 1200) -> Image.Image:
    return imaging_feature_matching.resize_for_residual(image, max_side)


def robust_residual_score(
    query_image: Image.Image,
    original_path: Path,
    watermarked_path: Path,
    min_inliers: int = 80,
    min_ratio: float = 0.80,
) -> float:
    return imaging_feature_matching.robust_residual_score(
        query_image, original_path, watermarked_path, min_inliers, min_ratio,
        robust_channel=ROBUST_CHANNEL,
        resize_for_residual_fn=resize_for_residual,
        feature_match_homography_fn=feature_match_homography,
    )


def detect_by_visual_match(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return imaging_feature_matching.detect_by_visual_match(
        image,
        records=read_records() if records is None else records,
        upload_dir=UPLOAD_DIR,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
        watermark_layers=WATERMARK_LAYERS,
        image_to_cv_gray_fn=image_to_cv_gray,
        feature_match_score_fn=feature_match_score,
        robust_residual_score_fn=robust_residual_score,
    )


def is_registered_original_image(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | None = None,
) -> bool:
    return imaging_feature_matching.is_registered_original_image(
        image,
        records=read_records() if records is None else records,
        upload_dir=UPLOAD_DIR,
    )


def load_font(size: int) -> ImageFont.ImageFont:
    return imaging_visible_mark.load_font(size)


def load_random_font(size: int, rng: np.random.Generator) -> ImageFont.ImageFont:
    return imaging_visible_mark.load_random_font(size, rng, load_font_fn=load_font)


def draw_text_pattern(layer: Image.Image, text: str, angle: int, gap: int, opacity: int) -> None:
    return imaging_visible_mark.draw_text_pattern(
        layer, text, angle, gap, opacity, load_font_fn=load_font,
    )


def draw_irregular_text_pattern(layer: Image.Image, text: str, opacity: int, complexity: str) -> None:
    return imaging_visible_mark.draw_irregular_text_pattern(
        layer,
        text,
        opacity,
        complexity,
        load_random_font_fn=load_random_font,
    )


def draw_prominent_corner_label(image: Image.Image, text: str) -> Image.Image:
    return imaging_visible_mark.draw_prominent_corner_label(
        image, text, load_font_fn=load_font,
    )


def apply_visible_copyright(
    image: Image.Image,
    enabled: bool,
    text: str,
    opacity: float,
    complexity: str,
    irregular: bool = True,
    prominent_corner: bool = False,
) -> Image.Image:
    return imaging_visible_mark.apply_visible_copyright(
        image, enabled, text, opacity, complexity, irregular, prominent_corner,
        draw_irregular_text_pattern_fn=draw_irregular_text_pattern,
        draw_text_pattern_fn=draw_text_pattern,
        draw_prominent_corner_label_fn=draw_prominent_corner_label,
    )


async def load_upload_image(file: UploadFile) -> Image.Image:
    return await imaging_io.load_upload_image(
        file, load_image_from_bytes_fn=load_image_from_bytes,
    )


def load_image_from_bytes(content: bytes) -> Image.Image:
    return imaging_io.load_image_from_bytes(content)


def file_sha256(content: bytes) -> str:
    return imaging_fingerprints.file_sha256(content)


def path_sha256(path: Path) -> str:
    return imaging_fingerprints.path_sha256(path)


def image_content_sha256(image: Image.Image) -> str:
    return imaging_fingerprints.image_content_sha256(image)


def matched_file_fingerprint(
    content: bytes,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return imaging_fingerprints.matched_file_fingerprint(
        content,
        read_records=read_records if records is None else lambda: records,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
        watermark_layers=WATERMARK_LAYERS,
        file_sha256_fn=file_sha256,
        image_content_sha256_fn=image_content_sha256,
        load_image_from_bytes_fn=load_image_from_bytes,
    )


def save_thumbnail(image: Image.Image, path: Path, scale: float = 0.20) -> None:
    return imaging_io.save_thumbnail(image, path, scale)


def load_image_from_url(url: str) -> Image.Image:
    return imaging_io.load_image_from_url(url, UPLOAD_DIR)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/site-logo.png")
def site_logo() -> FileResponse:
    return FileResponse(BASE_DIR / "site-logo.png")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "favicon.ico")


@app.get("/favico.ico")
def favico() -> FileResponse:
    return FileResponse(BASE_DIR / "favico.ico")


@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...)) -> dict[str, Any]:
    return get_auth_service().login(username, password)


@app.get("/api/roles")
def get_roles() -> dict[str, Any]:
    return get_auth_service().list_roles()


@app.put("/api/roles/{role_key}")
def update_role(role_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    return get_auth_service().update_role(role_key, payload)


@app.get("/api/users")
def get_users() -> dict[str, Any]:
    return get_auth_service().list_users()


@app.post("/api/users")
def create_user(payload: dict[str, Any]) -> dict[str, Any]:
    return get_auth_service().create_user(payload)


@app.put("/api/users/{username}")
def update_user(username: str, payload: dict[str, Any]) -> dict[str, Any]:
    return get_auth_service().update_user(username, payload)


@app.delete("/api/users/{username}")
def delete_user(username: str) -> dict[str, Any]:
    return get_auth_service().delete_user(username)


@app.post("/api/watermark/embed")
async def embed_watermark(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    mode: str = Form("dct"),
    copyright_enabled: str = Form("false"),
    copyright_text: str = Form("© QQ:757675150"),
    copyright_opacity: str = Form("0.16"),
    copyright_complexity: str = Form("medium"),
    copyright_irregular_enabled: str = Form("true"),
    copyright_prominent_corner_enabled: str = Form("false"),
    fidelity_level: str = Form("0.75"),
    robust_watermark_strength: str = Form(DEFAULT_ROBUST_WATERMARK_STRENGTH),
    robust_watermark_version: str = Form(DEFAULT_ROBUST_WATERMARK_VERSION),
    small_crop_trace_enabled: str = Form(""),
    small_crop_trace_strength: str = Form("1.0"),
    small_crop_trace_density: str = Form("high"),
    dot_matrix_trace_enabled: str = Form("false"),
    dot_matrix_trace_strength: str = Form("0.85"),
) -> dict[str, Any]:
    return await get_watermark_service().embed(
        file=file,
        user_id=user_id,
        mode=mode,
        copyright_enabled=copyright_enabled,
        copyright_text=copyright_text,
        copyright_opacity=copyright_opacity,
        copyright_complexity=copyright_complexity,
        copyright_irregular_enabled=copyright_irregular_enabled,
        copyright_prominent_corner_enabled=copyright_prominent_corner_enabled,
        fidelity_level=fidelity_level,
        robust_watermark_strength=robust_watermark_strength,
        robust_watermark_version=robust_watermark_version,
        small_crop_trace_enabled=small_crop_trace_enabled,
        small_crop_trace_strength=small_crop_trace_strength,
        small_crop_trace_density=small_crop_trace_density,
        dot_matrix_trace_enabled=dot_matrix_trace_enabled,
        dot_matrix_trace_strength=dot_matrix_trace_strength,
    )


def extract_watermark_from_image(image: Image.Image) -> dict[str, Any]:
    return get_watermark_service().extract_image(image)


@app.post("/api/watermark/extract")
async def extract_watermark(file: UploadFile = File(...)) -> dict[str, Any]:
    return await get_watermark_service().extract_upload(file)


@app.post("/api/watermark/extract-url")
def extract_watermark_url(url: str = Form(...)) -> dict[str, Any]:
    return get_watermark_service().extract_url(url)


@app.get("/api/dashboard-stats")
def dashboard_stats() -> dict[str, int | float]:
    records = read_records()
    detection_stats = read_detection_stats()
    attempts = detection_stats["attempts"]
    successes = detection_stats["successes"]
    success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
    return {
        "today": today_watermark_count(records),
        "detection_success_rate": success_rate,
    }


@app.get("/api/images")
def list_images() -> dict[str, Any]:
    records = read_records()
    protected = sum(1 for item in records if item.get("status") == "保护中")
    leaks = sum(1 for item in records if item.get("status") == "泄露预警")
    hits = sum(1 for item in records if item.get("status") == "溯源命中")
    detection_stats = read_detection_stats()
    attempts = detection_stats["attempts"]
    successes = detection_stats["successes"]
    success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
    return {
        "items": records,
        "stats": {
            "total": len(records),
            "protected": protected,
            "leaks": leaks,
            "hits": hits,
            "today": today_watermark_count(records),
            "detection_attempts": attempts,
            "detection_successes": successes,
            "detection_success_rate": success_rate,
        },
        "db_enabled": DB_ENABLED,
        "db_ready": database_ready(),
        "db_error": db_error,
        "db_url": masked_db_url(),
    }


@app.delete("/api/images/{image_id}")
def delete_image(image_id: str) -> dict[str, bool]:
    records = read_records()
    target = next((item for item in records if item.get("id") == image_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="图片不存在")
    kept = [item for item in records if item.get("id") != image_id]
    write_records(kept)
    for key in ("original_url", "download_url", "thumbnail_url"):
        value = target.get(key)
        if value and value.startswith("/uploads/"):
            path = UPLOAD_DIR / value.replace("/uploads/", "")
            if path.exists():
                path.unlink()
    return {"deleted": True}


@app.post("/api/dev/reset")
def reset_dev_data() -> dict[str, bool]:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    if database_ready():
        db_clear_all()
        seed_database_defaults(require_store())
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_dirs()
    return {"reset": True}
