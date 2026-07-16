import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import time
import uuid
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
from trace_app.watermark import frequency as watermark_frequency
from trace_app.watermark import lsb as watermark_lsb
from trace_app.watermark import small_crop as watermark_small_crop
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


def public_users(users: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        username: {"role": str(info.get("role") or "operator")}
        for username, info in users.items()
    }


def allowed_menu_keys(menus: Any) -> list[str]:
    if not isinstance(menus, list):
        return []
    return [key for key in menus if key in MENU_LABELS]


def role_for_username(username: str) -> str:
    users = read_users()["users"]
    return str(users.get(username, {}).get("role") or "operator")


def record_watermark_generation() -> None:
    repository.record_watermark_generation()


def today_watermark_count(records: list[dict[str, Any]]) -> int:
    return repository.today_watermark_count(
        records, read_watermark_stats(), is_today=is_today_record
    )


def remember_generated_trace(trace_id: str) -> None:
    generated = list(getattr(app.state, "generated_trace_ids", []))
    generated.insert(0, trace_id)
    app.state.generated_trace_ids = generated[:24]


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


def robust_code_from_trace(trace_id: str) -> int:
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=6).digest()
    body = int.from_bytes(digest, "big")
    return (ROBUST_MAGIC << 48) | body


def robust_bits_from_code(code: int) -> list[int]:
    return [(code >> shift) & 1 for shift in range(ROBUST_BITS - 1, -1, -1)]


def robust_payload_bytes(trace_id: str) -> bytes:
    return robust_code_from_trace(trace_id).to_bytes(8, "big")


def code_crc16(value: int) -> int:
    return int.from_bytes(hashlib.blake2b(value.to_bytes(4, "big"), digest_size=2).digest(), "big")


def watermark_payload_from_trace(trace_id: str) -> int:
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=2).digest()
    body = int.from_bytes(digest, "big")
    checksum = code_crc16((ROBUST_MAGIC << 16) | body)
    return (ROBUST_MAGIC << 32) | (body << 16) | checksum


def watermark_bits_from_trace(trace_id: str) -> list[int]:
    payload = watermark_payload_from_trace(trace_id)
    base = [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]
    repeated = []
    for index in range(CODE_PHYSICAL_BITS):
        repeated.append(base[index % CODE_PAYLOAD_BITS])
    return repeated






def recover_payload_from_code(code: int) -> tuple[int, int]:
    bits = [(code >> shift) & 1 for shift in range(CODE_PHYSICAL_BITS - 1, -1, -1)]
    recovered = 0
    corrections = 0
    for index in range(CODE_PAYLOAD_BITS):
        votes = [bits[pos] for pos in range(index, CODE_PHYSICAL_BITS, CODE_PAYLOAD_BITS)]
        one_votes = sum(votes)
        zero_votes = len(votes) - one_votes
        bit = 1 if one_votes >= zero_votes else 0
        corrections += min(one_votes, zero_votes)
        recovered = (recovered << 1) | bit
    return recovered, corrections


























def fidelity_to_strength(value: str) -> float:
    fidelity = clamp_float(value, 0.75, 0.0, 1.0)
    return 1.0 - fidelity * 0.72




def robust_strength_to_scale(value: str | float | None) -> float:
    default = clamp_float(DEFAULT_ROBUST_WATERMARK_STRENGTH, 1.0, 0.0, 2.0)
    return clamp_float(value, default, 0.0, 2.0)


def normalize_robust_watermark_version(value: str | int | None) -> int:
    try:
        version = int(
            value if value is not None else ROBUST_WATERMARK_VERSION_V1
        )
    except (TypeError, ValueError):
        version = ROBUST_WATERMARK_VERSION_V1
    if version == ROBUST_WATERMARK_VERSION_V4:
        return ROBUST_WATERMARK_VERSION_V4
    if version == ROBUST_WATERMARK_VERSION_V3:
        return ROBUST_WATERMARK_VERSION_V3
    if version == ROBUST_WATERMARK_VERSION_V2:
        return ROBUST_WATERMARK_VERSION_V2
    return ROBUST_WATERMARK_VERSION_V1














































































def robust_code_to_trace(code: int) -> str | None:
    if (code >> 48) != ROBUST_MAGIC:
        return None
    for record in read_records():
        trace_id = record.get("trace_id")
        if trace_id and robust_code_from_trace(trace_id) == code:
            return trace_id
    return None


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def robust_code_to_trace_fuzzy(code: int, max_errors: int = 18) -> tuple[str | None, int]:
    magic_distance = hamming_distance(code >> 48, ROBUST_MAGIC)
    if magic_distance > 6:
        return None, ROBUST_BITS + 1
    best_trace = None
    best_distance = ROBUST_BITS + 1
    for record in read_records():
        trace_id = record.get("trace_id")
        if not trace_id:
            continue
        distance = hamming_distance(code, robust_code_from_trace(trace_id))
        if distance < best_distance:
            best_trace = trace_id
            best_distance = distance
    if best_trace and best_distance <= max_errors:
        return best_trace, best_distance
    return None, best_distance


def robust_candidate_records() -> list[dict[str, Any]]:
    return [
        record
        for record in read_records()
        if record.get("trace_id") and record.get("robust_watermark")
    ]


def legacy_robust_candidate_records() -> list[dict[str, Any]]:
    return [
        record
        for record in robust_candidate_records()
        if normalize_robust_watermark_version(
            record.get("robust_watermark_version", ROBUST_WATERMARK_VERSION_V1)
        )
        == ROBUST_WATERMARK_VERSION_V1
    ]


def iter_robust_tiles(width: int, height: int):
    for y in range(0, height - ROBUST_TILE + 1, ROBUST_TILE):
        for x in range(0, width - ROBUST_TILE + 1, ROBUST_TILE):
            yield x, y


def embed_robust_watermark(image: Image.Image, trace_id: str, strength_scale: float = 1.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    bits = robust_bits_from_code(robust_code_from_trace(trace_id))
    for x, y in iter_robust_tiles(width, height):
        for bit_index, bit in enumerate(bits):
            row = bit_index // ROBUST_GRID
            col = bit_index % ROBUST_GRID
            y0 = y + row * ROBUST_CELL
            x0 = x + col * ROBUST_CELL
            patch = arr[y0 : y0 + ROBUST_CELL, x0 : x0 + ROBUST_CELL, ROBUST_CHANNEL]
            pattern = robust_pattern(bit_index, ROBUST_CELL)
            delta = pattern * ((ROBUST_DELTA * strength_scale) if bit else -(ROBUST_DELTA * strength_scale))
            if bit:
                patch[:, :] = np.clip(patch + delta, 0, 255)
            else:
                patch[:, :] = np.clip(patch + delta, 0, 255)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v2(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    codeword = encode_codeword(robust_payload_bytes(trace_id))
    for x, y in iter_robust_tiles(image.width, image.height):
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        phase_bytes = codeword_phase(codeword, phase)
        bits = robust_bits_from_code(int.from_bytes(phase_bytes, "big"))
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, ROBUST_GRID)
            y0 = y + row * ROBUST_CELL
            x0 = x + col * ROBUST_CELL
            patch = arr[y0 : y0 + ROBUST_CELL, x0 : x0 + ROBUST_CELL, ROBUST_CHANNEL]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch
                + robust_pattern(bit_index, ROBUST_CELL)
                * ROBUST_DELTA
                * strength_scale
                * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v3(
    image: Image.Image,
    auth_code: bytes,
    strength_scale: float = 1.0,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    for x, y in iter_robust_tiles(image.width, image.height):
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        bits = permuted_code_bits(auth_code, phase)
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, ROBUST_GRID)
            y0 = y + row * ROBUST_CELL
            x0 = x + col * ROBUST_CELL
            patch = arr[y0 : y0 + ROBUST_CELL, x0 : x0 + ROBUST_CELL, ROBUST_CHANNEL]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch
                + robust_pattern(bit_index, ROBUST_CELL)
                * ROBUST_DELTA
                * strength_scale
                * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def extract_robust_from_grid(arr: np.ndarray, cell: int, offset_x: int, offset_y: int) -> tuple[int | None, float, int]:
    height, width = arr.shape[:2]
    tile = cell * ROBUST_GRID
    votes = [[0, 0] for _ in range(ROBUST_BITS)]
    tiles = 0
    for y in range(offset_y, height - tile + 1, tile):
        for x in range(offset_x, width - tile + 1, tile):
            tiles += 1
            for bit_index in range(ROBUST_BITS):
                row = bit_index // ROBUST_GRID
                col = bit_index % ROBUST_GRID
                y0 = y + row * cell
                x0 = x + col * cell
                patch = arr[y0 : y0 + cell, x0 : x0 + cell, :]
                if patch.size == 0:
                    continue
                blue = patch[:, :, ROBUST_CHANNEL]
                if blue.shape != (cell, cell):
                    continue
                pattern = robust_pattern(bit_index, cell).astype(np.float32)
                centered = blue - blue.mean()
                score = float((centered * pattern).mean())
                if score > 0:
                    votes[bit_index][1] += 1
                else:
                    votes[bit_index][0] += 1

    if tiles < 2:
        return None, 0.0, 0

    code = 0
    margins = []
    decided = 0
    for zero_votes, one_votes in votes:
        total = zero_votes + one_votes
        if total == 0:
            code = code << 1
            margins.append(0.0)
            continue
        decided += 1
        bit = 1 if one_votes > zero_votes else 0
        code = (code << 1) | bit
        margins.append(abs(one_votes - zero_votes) / total)
    confidence = sum(margins) / len(margins) if margins else 0.0
    return code, confidence, decided


def decode_aligned_robust_trace(
    alignment: dict[str, Any],
    record: dict[str, Any],
    max_errors: int = 4,
) -> dict[str, Any] | None:
    trace_id = record.get("trace_id")
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None

    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    aggregate_scores = np.zeros(ROBUST_BITS, dtype=np.float64)
    authenticated_tiles = 0
    for x, y in iter_robust_tiles(original_width, original_height):
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + ROBUST_TILE) * target_scale)))
        y1 = min(height, int(round((y + ROBUST_TILE) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(
            aligned[y0:y1, x0:x1, :],
            (ROBUST_TILE, ROBUST_TILE),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        authenticated_tiles += 1
        for bit_index in range(ROBUST_BITS):
            row = bit_index // ROBUST_GRID
            col = bit_index % ROBUST_GRID
            cell = tile[
                row * ROBUST_CELL : (row + 1) * ROBUST_CELL,
                col * ROBUST_CELL : (col + 1) * ROBUST_CELL,
                ROBUST_CHANNEL,
            ]
            centered = cell - cell.mean()
            aggregate_scores[bit_index] += float(
                np.mean(centered * robust_pattern(bit_index, ROBUST_CELL))
            )
    if authenticated_tiles < 2:
        return None

    decoded_code = 0
    for score in aggregate_scores:
        decoded_code = (decoded_code << 1) | int(score > 0)
    expected_code = robust_code_from_trace(trace_id)
    bit_errors = hamming_distance(decoded_code, expected_code)
    magic_errors = hamming_distance(decoded_code >> 48, ROBUST_MAGIC)
    if bit_errors > max_errors or magic_errors > max_errors:
        return None
    average_scores = aggregate_scores / authenticated_tiles
    return {
        "record": record,
        "trace_id": trace_id,
        "decoded_code": decoded_code,
        "bit_errors": bit_errors,
        "magic_errors": magic_errors,
        "authenticated_tiles": authenticated_tiles,
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


def _scores_to_byte(scores: np.ndarray) -> tuple[int, float]:
    value = 0
    absolute = np.abs(scores.astype(np.float64))
    scale = max(1e-6, float(np.median(absolute)))
    for score in scores:
        value = (value << 1) | int(score > 0)
    return value, float(np.min(absolute / scale))


def _phase_scores_to_codeword(
    phase_scores: np.ndarray,
    phase_counts: list[int],
) -> tuple[bytes, list[float]]:
    observed = bytearray()
    confidences = []
    for phase in range(3):
        average = phase_scores[phase] / max(1, phase_counts[phase])
        for start in range(0, ROBUST_BITS, 8):
            value, confidence = _scores_to_byte(average[start : start + 8])
            observed.append(value)
            confidences.append(confidence)
    return bytes(observed), confidences


def decode_aligned_robust_trace_v2(
    alignment: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any] | None:
    trace_id = record.get("trace_id")
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None

    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    phase_scores = np.zeros((3, ROBUST_BITS), dtype=np.float64)
    phase_counts = [0, 0, 0]
    for x, y in iter_robust_tiles(original_width, original_height):
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + ROBUST_TILE) * target_scale)))
        y1 = min(height, int(round((y + ROBUST_TILE) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(
            aligned[y0:y1, x0:x1, :],
            (ROBUST_TILE, ROBUST_TILE),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        phase_counts[phase] += 1
        for bit_index in range(ROBUST_BITS):
            row, col = divmod(bit_index, ROBUST_GRID)
            cell = tile[
                row * ROBUST_CELL : (row + 1) * ROBUST_CELL,
                col * ROBUST_CELL : (col + 1) * ROBUST_CELL,
                ROBUST_CHANNEL,
            ]
            centered = cell - cell.mean()
            phase_scores[phase, bit_index] += float(
                np.mean(centered * robust_pattern(bit_index, ROBUST_CELL))
            )
    if min(phase_counts) < 2:
        return None

    observed, confidences = _phase_scores_to_codeword(phase_scores, phase_counts)
    decoded = decode_expected_codeword(
        observed,
        robust_payload_bytes(trace_id),
        confidences,
    )
    if not decoded:
        return None

    average_scores = np.vstack(
        [phase_scores[index] / phase_counts[index] for index in range(3)]
    )
    return {
        "record": record,
        "trace_id": trace_id,
        "corrected_symbols": decoded["corrected_symbols"],
        "erasure_count": decoded["erasure_count"],
        "bit_errors": decoded["bit_errors"],
        "recovery_method": decoded["recovery_method"],
        "phase_tile_counts": phase_counts,
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


def _record_v3_auth_code(record: dict[str, Any]) -> bytes | None:
    text = str(record.get("robust_auth_code") or "").strip().lower()
    if len(text) != 16 or not re.fullmatch(r"[0-9a-f]{16}", text):
        return None
    try:
        code = bytes.fromhex(text)
    except ValueError:
        return None
    return code if len(code) == 8 else None


def decode_aligned_robust_trace_v3(
    alignment: dict[str, Any],
    record: dict[str, Any],
    max_errors: int = 8,
) -> dict[str, Any] | None:
    trace_id = record.get("trace_id")
    auth_code = _record_v3_auth_code(record)
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or auth_code is None
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None

    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    aggregate_scores = np.zeros(ROBUST_BITS, dtype=np.float64)
    phase_counts = [0, 0, 0]
    authenticated_tiles = 0
    for x, y in iter_robust_tiles(original_width, original_height):
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + ROBUST_TILE) * target_scale)))
        y1 = min(height, int(round((y + ROBUST_TILE) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(
            aligned[y0:y1, x0:x1, :],
            (ROBUST_TILE, ROBUST_TILE),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        physical_scores = np.zeros(ROBUST_BITS, dtype=np.float64)
        for bit_index in range(ROBUST_BITS):
            row, col = divmod(bit_index, ROBUST_GRID)
            cell = tile[
                row * ROBUST_CELL : (row + 1) * ROBUST_CELL,
                col * ROBUST_CELL : (col + 1) * ROBUST_CELL,
                ROBUST_CHANNEL,
            ]
            centered = cell - cell.mean()
            physical_scores[bit_index] = float(
                np.mean(centered * robust_pattern(bit_index, ROBUST_CELL))
            )
        scale = max(1e-6, float(np.median(np.abs(physical_scores))))
        physical_scores = np.clip(physical_scores / scale, -3.0, 3.0)
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        permutation = phase_permutation(phase)
        for logical, physical in enumerate(permutation):
            aggregate_scores[logical] += physical_scores[physical]
        phase_counts[phase] += 1
        authenticated_tiles += 1

    if authenticated_tiles < 2 or sum(count > 0 for count in phase_counts) < 2:
        return None

    expected_value = int.from_bytes(auth_code, "big")
    expected_bits = np.array(
        [(expected_value >> shift) & 1 for shift in range(ROBUST_BITS - 1, -1, -1)],
        dtype=np.int8,
    )
    observed_bits = (aggregate_scores > 0).astype(np.int8)
    bit_errors = int(np.count_nonzero(observed_bits != expected_bits))
    expected_signs = expected_bits.astype(np.float64) * 2.0 - 1.0
    average_scores = aggregate_scores / authenticated_tiles
    mean_signed_agreement = float(np.mean(average_scores * expected_signs))
    if bit_errors > max_errors or mean_signed_agreement <= 0:
        return None

    return {
        "record": record,
        "trace_id": trace_id,
        "bit_errors": bit_errors,
        "authenticated_tiles": authenticated_tiles,
        "phase_tile_counts": phase_counts,
        "mean_signed_agreement": round(mean_signed_agreement, 6),
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


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


def v4_candidate_records() -> tuple[V4Candidate, ...]:
    config = V4Config()
    candidates = []
    for record in read_records():
        if record.get("robust_watermark_version") != ROBUST_WATERMARK_VERSION_V4:
            continue
        if record.get("robust_watermark_codec") != config.codec:
            continue
        record_id = str(record.get("id") or "").strip()
        trace_id = str(record.get("trace_id") or "").strip()
        auth_hex = str(record.get("robust_auth_code") or "").strip()
        if (
            not record_id
            or not trace_id
            or not re.fullmatch(r"[0-9a-f]{8}", auth_hex)
        ):
            continue
        path = record_feature_index_path(record)
        if path is None:
            continue
        feature_index = load_v4_feature_index(path)
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
) -> dict[str, Any] | None:
    available = v4_candidate_records() if candidates is None else candidates
    if not available:
        return None
    records = read_records()
    record_by_id = {
        str(record.get("id")): record
        for record in records
        if record.get("robust_watermark_version") == ROBUST_WATERMARK_VERSION_V4
    }
    generated_trace_ids = list(getattr(app.state, "generated_trace_ids", []))
    recent_record_ids = tuple(
        candidate.record_id
        for trace_id in generated_trace_ids
        for candidate in available
        if candidate.trace_id == trace_id
    )
    result = detect_v4(
        image.convert("RGB"),
        available,
        V4Config(),
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
    started = time.perf_counter()
    candidates = records if records is not None else robust_candidate_records()
    candidates = [
        record
        for record in candidates
        if record.get("trace_id") and record.get("robust_watermark")
    ]
    candidates = rank_aligned_candidates(image, candidates)[: max(1, candidate_limit)]
    authenticated = []
    for record in candidates:
        if budget_seconds > 0 and time.perf_counter() - started >= budget_seconds:
            break
        alignment = align_query_to_record(image, record)
        if not alignment:
            continue
        version = normalize_robust_watermark_version(
            record.get("robust_watermark_version", ROBUST_WATERMARK_VERSION_V1)
        )
        if version == ROBUST_WATERMARK_VERSION_V3:
            decoded = decode_aligned_robust_trace_v3(alignment, record)
        elif version == ROBUST_WATERMARK_VERSION_V2:
            decoded = decode_aligned_robust_trace_v2(alignment, record)
        else:
            decoded = decode_aligned_robust_trace(alignment, record)
        if decoded:
            authenticated.append((decoded, alignment))
    trace_ids = {decoded["trace_id"] for decoded, _ in authenticated}
    if len(authenticated) != 1 or len(trace_ids) != 1:
        return None

    decoded, alignment = authenticated[0]
    record = decoded["record"]
    version = normalize_robust_watermark_version(
        record.get("robust_watermark_version", ROBUST_WATERMARK_VERSION_V1)
    )
    confidence = max(80, min(99, 99 - decoded["bit_errors"] * 6))
    code_recovery = {
        "visual_inliers": alignment["inliers"],
        "visual_ratio": alignment["ratio"],
        "aligned_coverage": alignment["coverage"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if version == ROBUST_WATERMARK_VERSION_V3:
        code_recovery.update({
            "method": "homography_aligned_hmac64_full_repeat_v3",
            "codec": ROBUST_WATERMARK_CODEC_V3,
            "bit_errors": decoded["bit_errors"],
            "authenticated_tiles": decoded["authenticated_tiles"],
            "phase_tile_counts": decoded["phase_tile_counts"],
            "mean_signed_agreement": decoded["mean_signed_agreement"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    elif version == ROBUST_WATERMARK_VERSION_V2:
        code_recovery.update({
            "method": "homography_aligned_rs_24_8_three_phase",
            "codec": ROBUST_WATERMARK_CODEC_V2,
            "bit_errors": decoded["bit_errors"],
            "corrected_symbols": decoded["corrected_symbols"],
            "erasure_count": decoded["erasure_count"],
            "recovery_method": decoded["recovery_method"],
            "phase_tile_counts": decoded["phase_tile_counts"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    else:
        code_recovery.update({
            "method": "homography_aligned_robust_64",
            "bit_errors": decoded["bit_errors"],
            "magic_errors": decoded["magic_errors"],
            "authenticated_tiles": decoded["authenticated_tiles"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": decoded["trace_id"],
        "user_id": record.get("user_id"),
        "mode": (
            "aligned_robust_hmac_v3"
            if version == ROBUST_WATERMARK_VERSION_V3
            else "aligned_robust_rs_v2"
            if version == ROBUST_WATERMARK_VERSION_V2
            else "aligned_robust_code"
        ),
        "mode_label": (
            "几何对齐 HMAC 认证水印"
            if version == ROBUST_WATERMARK_VERSION_V3
            else "几何对齐 RS 认证水印"
            if version == ROBUST_WATERMARK_VERSION_V2
            else "几何对齐 64-bit 认证水印"
        ),
        "created_at": record.get("created_at"),
        "confidence": confidence,
        "phash_match": False,
        "status": "认证水印恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "code_recovery": code_recovery,
    }, record)


def extract_robust_code(image: Image.Image, records: list[dict[str, Any]] | None = None) -> tuple[str | None, float, int]:
    records = records if records is not None else legacy_robust_candidate_records()
    trace_codes = {
        record.get("trace_id"): robust_code_from_trace(record.get("trace_id"))
        for record in records
        if record.get("trace_id")
    }
    if not trace_codes:
        return None, 0.0, 0
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if width < ROBUST_TILE or height < ROBUST_TILE:
        return None, 0.0, 0
    candidates = []
    for cell in (8, 7, 9):
        tile = cell * ROBUST_GRID
        step = max(1, cell * 2)
        for offset_y in range(0, min(tile, height - tile + 1), step):
            for offset_x in range(0, min(tile, width - tile + 1), step):
                code, confidence, decided = extract_robust_from_grid(arr, cell, offset_x, offset_y)
                if code is None or decided < ROBUST_BITS:
                    continue
                trace_id = next((item for item, item_code in trace_codes.items() if item_code == code), None)
                if trace_id is not None:
                    candidates.append((trace_id, confidence, decided))
    if not candidates:
        return None, 0.0, 0
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def detect_robust_watermark(image: Image.Image) -> dict[str, Any] | None:
    records = legacy_robust_candidate_records()
    if not records:
        return None
    trace_id, confidence, decided = extract_robust_code(image, records)
    if not trace_id or confidence < 0.08:
        return None
    record = next((item for item in records if item.get("trace_id") == trace_id), None)
    if not record:
        return None
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": trace_id,
        "user_id": record.get("user_id"),
        "mode": "robust_dct",
        "mode_label": "30% 局部鲁棒水印",
        "created_at": record.get("created_at"),
        "confidence": int(min(98, max(75, confidence * 100))),
        "phash_match": True,
        "status": "鲁棒水印命中",
        "extracted_at": now_text(),
        "robust_decided_bits": decided,
        "robust_score": round(confidence, 3),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": layer_scores_for_image(image, trace_id),
    }, record)


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
    width, height = image.size
    pixels = width * height
    if pixels <= 3_000_000:
        return True
    aspect = max(width, height) / max(1, min(width, height))
    return pixels <= 5_000_000 and aspect >= 2.2


def should_run_visual_match_fallback(image: Image.Image) -> bool:
    records = read_records()
    if not any(record.get("robust_watermark") for record in records):
        return False
    width, height = image.size
    return width * height >= 40_000


def detect_visible_copyright(image: Image.Image) -> dict[str, Any] | None:
    return imaging_visible_mark.detect_visible_copyright(
        image,
        records=read_records(),
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


def dot_matrix_candidate_records() -> list[tuple[str, int, dict[str, Any]]]:
    return watermark_dot_matrix.dot_matrix_candidate_records(
        read_records(),
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


def detect_dot_matrix_trace(image: Image.Image) -> dict[str, Any] | None:
    return watermark_dot_matrix.detect_dot_matrix_trace(
        image,
        dot_matrix_candidate_records(),
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


def detect_small_crop_trace(image: Image.Image) -> dict[str, Any] | None:
    return watermark_small_crop.detect_small_crop_trace(
        image,
        read_records(),
        list(getattr(app.state, "generated_trace_ids", [])),
        watermark_payload_from_trace=watermark_payload_from_trace,
        record_visual_consistency=record_visual_consistency,
        recover_payload_from_code=recover_payload_from_code,
        hamming_distance=hamming_distance,
        code_crc16=code_crc16,
        now_text=now_text,
        with_evidence_fields=with_evidence_fields,
    )


def detect_watermark_code(image: Image.Image) -> dict[str, Any] | None:
    return watermark_small_crop.detect_watermark_code(
        image,
        read_records(),
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


def residual_candidate_evidence(image: Image.Image) -> dict[str, Any] | None:
    return imaging_feature_matching.residual_candidate_evidence(
        image,
        records=read_records(),
        record_visual_consistency_fn=record_visual_consistency,
    )


def detect_by_residual_match(image: Image.Image) -> dict[str, Any] | None:
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


def detect_by_visual_match(image: Image.Image) -> dict[str, Any] | None:
    return imaging_feature_matching.detect_by_visual_match(
        image,
        records=read_records(),
        upload_dir=UPLOAD_DIR,
        with_evidence_fields=with_evidence_fields,
        now_text=now_text,
        watermark_layers=WATERMARK_LAYERS,
        image_to_cv_gray_fn=image_to_cv_gray,
        feature_match_score_fn=feature_match_score,
        robust_residual_score_fn=robust_residual_score,
    )


def is_registered_original_image(image: Image.Image) -> bool:
    return imaging_feature_matching.is_registered_original_image(
        image, records=read_records(), upload_dir=UPLOAD_DIR,
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


def matched_file_fingerprint(content: bytes) -> dict[str, Any] | None:
    return imaging_fingerprints.matched_file_fingerprint(
        content,
        read_records=read_records,
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
    role = require_store().authenticate(username, password)
    if role is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    roles = read_roles()["roles"]
    menus = allowed_menu_keys(roles.get(role, {}).get("menus", []))
    return {"token": f"local-{uuid.uuid4().hex}", "username": username, "role": role, "menus": menus}


@app.get("/api/roles")
def get_roles() -> dict[str, Any]:
    return {"menus": MENU_LABELS, "roles": read_roles()["roles"]}


@app.put("/api/roles/{role_key}")
def update_role(role_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    roles = read_roles()["roles"]
    if role_key not in roles:
        raise HTTPException(status_code=404, detail="角色不存在")
    require_store().update_role_menus(
        role_key, allowed_menu_keys(payload.get("menus"))
    )
    return {"menus": MENU_LABELS, "roles": read_roles()["roles"]}


@app.get("/api/users")
def get_users() -> dict[str, Any]:
    return {"users": public_users(read_users()["users"]), "roles": read_roles()["roles"]}


@app.post("/api/users")
def create_user(payload: dict[str, Any]) -> dict[str, Any]:
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    role = str(payload.get("role") or "operator")
    roles = read_roles()["roles"]
    if not username:
        raise HTTPException(status_code=400, detail="请输入用户名")
    if not password:
        raise HTTPException(status_code=400, detail="请输入密码")
    if role not in roles:
        raise HTTPException(status_code=400, detail="角色不存在")
    store = require_store()
    if username in store.list_users():
        raise HTTPException(status_code=409, detail="用户已存在")
    store.create_user(username, password, role)
    return {"users": store.list_users(), "roles": roles}


@app.put("/api/users/{username}")
def update_user(username: str, payload: dict[str, Any]) -> dict[str, Any]:
    store = require_store()
    users = store.list_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="用户不存在")
    role = str(payload.get("role") or "")
    roles = read_roles()["roles"]
    if role not in roles:
        raise HTTPException(status_code=400, detail="角色不存在")
    store.update_user_role(username, role)
    return {"users": store.list_users(), "roles": roles}


@app.delete("/api/users/{username}")
def delete_user(username: str) -> dict[str, Any]:
    store = require_store()
    if not store.delete_user(username):
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"users": store.list_users(), "roles": read_roles()["roles"]}


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
    ensure_dirs()
    image_id = uuid.uuid4().hex
    trace_id = f"TR-{uuid.uuid4().hex[:16].upper()}"
    evidence_uuid = uuid.uuid4().hex.upper()
    evidence_fields = evidence_uuid_fields(evidence_uuid)
    safe_name = Path(file.filename or "image.png").name
    original_path = ORIGINAL_DIR / f"{image_id}-{safe_name}"
    output_path = WATERMARKED_DIR / f"{image_id}-watermarked.png"
    thumbnail_path = THUMBNAIL_DIR / f"{image_id}-thumb.png"

    image = await load_upload_image(file)
    image.save(original_path)
    normalized_mode = normalize_mode(mode)
    visible = apply_visible_copyright(
        image,
        parse_bool(copyright_enabled),
        copyright_text,
        clamp_float(copyright_opacity, 0.16, 0.02, 0.90),
        copyright_complexity,
        parse_bool(copyright_irregular_enabled),
        parse_bool(copyright_prominent_corner_enabled),
    )
    created_at = now_text()
    payload = {
        "id": image_id,
        "trace_id": trace_id,
        **evidence_fields,
        "user_id": user_id,
        "mode": normalized_mode,
        "mode_label": mode_label(normalized_mode),
        "created_at": created_at,
    }
    strength_scale = fidelity_to_strength(fidelity_level)
    robust_strength = robust_strength_to_scale(robust_watermark_strength)
    robust_version = normalize_robust_watermark_version(robust_watermark_version)
    robust_auth_code = None
    v4_config = V4Config()
    if robust_version == ROBUST_WATERMARK_VERSION_V4:
        try:
            robust_auth_code = v4_authentication_tag(
                trace_id,
                DEFAULT_WATERMARK_AUTH_KEY,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    elif robust_version == ROBUST_WATERMARK_VERSION_V3:
        try:
            robust_auth_code = auth_code_from_trace(trace_id, DEFAULT_WATERMARK_AUTH_KEY)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    small_crop_enabled = (
        app.state.small_crop_trace_default_enabled
        if str(small_crop_trace_enabled or "").strip() == ""
        else parse_bool(small_crop_trace_enabled)
    )
    small_crop_strength = small_crop_strength_to_scale(small_crop_trace_strength)
    small_crop_density = normalize_small_crop_density(small_crop_trace_density)
    dot_matrix_enabled = parse_bool(dot_matrix_trace_enabled)
    dot_matrix_strength = clamp_float(dot_matrix_trace_strength, 0.85, 0.0, 1.0)
    if robust_version == ROBUST_WATERMARK_VERSION_V4:
        small_crop_enabled = False
        dot_matrix_enabled = False
        try:
            watermarked = embed_v4_codeword(
                embed_v4_pilot(visible, v4_config),
                encode_v4_codeword(robust_auth_code),
                v4_config,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if robust_version == ROBUST_WATERMARK_VERSION_V3:
            robust = embed_robust_watermark_v3(visible, robust_auth_code, robust_strength)
        elif robust_version == ROBUST_WATERMARK_VERSION_V2:
            robust = embed_robust_watermark_v2(visible, trace_id, robust_strength)
        else:
            robust = embed_robust_watermark(visible, trace_id, robust_strength)
        frequency_marked = apply_frequency_layers(robust, trace_id)
        code_marked = apply_code_layer(frequency_marked, trace_id, strength_scale)
        small_crop_marked = (
            apply_small_crop_trace_layer(
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
            apply_dot_matrix_trace_layer(small_crop_marked, trace_id, dot_matrix_strength)
            if dot_matrix_enabled
            else small_crop_marked
        )
        watermarked = embed_lsb(dot_matrix_marked, payload)
    watermarked.save(output_path, format="PNG")
    save_thumbnail(watermarked, thumbnail_path)
    feature_index_path = (
        save_record_feature_index_v4(watermarked, image_id)
        if robust_version == ROBUST_WATERMARK_VERSION_V4
        else save_record_feature_index(watermarked, image_id)
    )
    original_file_sha256 = path_sha256(original_path)
    watermarked_file_sha256 = path_sha256(output_path)
    original_image_sha256 = image_content_sha256(image)
    watermarked_image_sha256 = image_content_sha256(watermarked)

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
        "original_file_sha256": original_file_sha256,
        "watermarked_file_sha256": watermarked_file_sha256,
        "original_image_sha256": original_image_sha256,
        "watermarked_image_sha256": watermarked_image_sha256,
        "copyright_enabled": parse_bool(copyright_enabled),
        "copyright_text": copyright_text.strip() or "© QQ:757675150",
        "copyright_opacity": clamp_float(copyright_opacity, 0.16, 0.02, 0.90),
        "copyright_complexity": copyright_complexity,
        "copyright_irregular_enabled": parse_bool(copyright_irregular_enabled),
        "copyright_prominent_corner_enabled": parse_bool(copyright_prominent_corner_enabled),
        "fidelity_level": clamp_float(fidelity_level, 0.75, 0.0, 1.0),
        "watermark_strength_scale": round(strength_scale, 4),
        "robust_watermark_strength": round(robust_strength, 4),
        "robust_watermark_version": robust_version,
        "robust_watermark_codec": (
            v4_config.codec
            if robust_version == ROBUST_WATERMARK_VERSION_V4
            else ROBUST_WATERMARK_CODEC_V3
            if robust_version == ROBUST_WATERMARK_VERSION_V3
            else ROBUST_WATERMARK_CODEC_V2
            if robust_version == ROBUST_WATERMARK_VERSION_V2
            else "legacy_robust_64"
        ),
        "robust_auth_code": robust_auth_code.hex() if robust_auth_code else None,
        "small_crop_trace_enabled": small_crop_enabled,
        "small_crop_trace_strength": small_crop_strength,
        "small_crop_trace_density": small_crop_density,
        "small_crop_trace_version": SMALL_TRACE_VERSION if small_crop_enabled else None,
        "dot_matrix_trace_enabled": dot_matrix_enabled,
        "dot_matrix_trace_strength": dot_matrix_strength,
        "dot_matrix_trace_version": DOT_MATRIX_VERSION if dot_matrix_enabled else None,
        "robust_watermark": True,
        "watermark_code_version": (
            None
            if robust_version == ROBUST_WATERMARK_VERSION_V4
            else CODE_WATERMARK_VERSION
        ),
        "watermark_layers": (
            {"dct_authenticated": True, "fft_sync": True}
            if robust_version == ROBUST_WATERMARK_VERSION_V4
            else WATERMARK_LAYERS
        ),
        "layer_scores": (
            {}
            if robust_version == ROBUST_WATERMARK_VERSION_V4
            else layer_scores_for_image(watermarked, trace_id)
        ),
    }
    add_record(record)
    record_watermark_generation()
    remember_generated_trace(trace_id)
    return record


def extract_watermark_from_image(image: Image.Image) -> dict[str, Any]:
    v4_candidates = v4_candidate_records()
    if v4_candidates:
        if is_registered_original_image(image):
            record_detection_result(False)
            raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")
        v4_match = detect_v4_watermark(image, v4_candidates)
        if v4_match:
            record_detection_result(True)
            return v4_match
        record_detection_result(False)
        raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")

    payload = extract_full_lsb(image)
    if not payload:
        if is_registered_original_image(image):
            record_detection_result(False)
            raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")
        if should_run_frequency_fallbacks(image):
            dot_matrix_match = detect_dot_matrix_trace(image)
            if dot_matrix_match:
                record_detection_result(True)
                return dot_matrix_match
            if app.state.aligned_authenticated_detection_enabled:
                aligned_match = detect_aligned_authenticated_watermark(
                    image,
                    candidate_limit=app.state.aligned_candidate_limit,
                    budget_seconds=app.state.watermark_detection_budget_seconds,
                )
                if aligned_match:
                    record_detection_result(True)
                    return aligned_match
            if app.state.dense_watermark_fallback_enabled:
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
        if app.state.visual_match_fallback_enabled:
            visual_match = detect_by_visual_match(image)
            if visual_match:
                record_detection_result(True)
                return visual_match
        residual_match = detect_by_residual_match(image)
        if residual_match:
            record_detection_result(True)
            return residual_match
        if app.state.visible_watermark_detection_enabled:
            fallback = detect_visible_copyright(image)
            if fallback:
                record_detection_result(True)
                return fallback
        payload = extract_block_lsb(image)
    if not payload:
        record_detection_result(False)
        raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")
    records = read_records()
    matched = next((item for item in records if item.get("trace_id") == payload.get("trace_id")), None)
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
        "watermark_layers": matched.get("watermark_layers", WATERMARK_LAYERS) if matched else WATERMARK_LAYERS,
        "layer_scores": layer_scores_for_image(image, payload.get("trace_id")) if payload.get("trace_id") else {},
    }, matched)


@app.post("/api/watermark/extract")
async def extract_watermark(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    fingerprint_match = matched_file_fingerprint(content)
    if fingerprint_match:
        if fingerprint_match.get("matched_file_type") == "original":
            record_detection_result(False)
            raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")
        record_detection_result(True)
        return fingerprint_match
    image = load_image_from_bytes(content)
    return extract_watermark_from_image(image)


@app.post("/api/watermark/extract-url")
def extract_watermark_url(url: str = Form(...)) -> dict[str, Any]:
    image = load_image_from_url(url)
    return extract_watermark_from_image(image)


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
