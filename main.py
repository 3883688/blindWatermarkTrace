import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
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

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DB_URL = os.getenv("DB_URL", "").strip()
ADMIN_USER = os.getenv("ADMIN_USER", "").strip()
ADMIN_PASS = os.getenv("ADMIN_PASS", "")
RUNNING_PYTEST = "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None
DB_ENABLED = not RUNNING_PYTEST

if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = BASE_DIR / UPLOAD_DIR
if not DATA_DIR.is_absolute():
    DATA_DIR = BASE_DIR / DATA_DIR

ORIGINAL_DIR = UPLOAD_DIR / "originals"
WATERMARKED_DIR = UPLOAD_DIR / "watermarked"
THUMBNAIL_DIR = UPLOAD_DIR / "thumbnails"
MAGIC = b"MWM1"
BLOCK_SIZE = 32
BLOCK_STRIDE = 32
ROBUST_MAGIC = 0b1010110011010011
ROBUST_BITS = 64
ROBUST_CELL = 16
ROBUST_GRID = 8
ROBUST_TILE = ROBUST_CELL * ROBUST_GRID
ROBUST_CHANNEL = 2
ROBUST_DELTA = 2
DEFAULT_ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
DEFAULT_ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
DEFAULT_WATERMARK_AUTH_KEY = os.getenv("WATERMARK_AUTH_KEY", "")
ROBUST_WATERMARK_VERSION_V1 = 1
ROBUST_WATERMARK_VERSION_V2 = 2
ROBUST_WATERMARK_VERSION_V3 = 3
ROBUST_WATERMARK_VERSION_V4 = 4
ROBUST_WATERMARK_CODEC_V2 = "rs_24_8_three_phase"
ROBUST_WATERMARK_CODEC_V3 = "hmac64_full_repeat_phase_permutation_v3"
FEATURE_MATCH_MIN_GOOD = 12
FEATURE_RECENT_RESERVE = 2
FEATURE_RECENT_BACKFILL = 4
DCT_BLOCK = 8
DCT_DELTA = 5.0
DWT_DELTA = 3.0
FFT_DELTA = 0.45
CODE_TILE = 160
CODE_CELL = 20
CODE_GRID = 8
CODE_DELTA = 9.0
CODE_WATERMARK_VERSION = 4
CODE_PHYSICAL_BITS = 64
CODE_PAYLOAD_BITS = 48
CODE_CHANNEL_WEIGHTS = (0.45, 0.75, 0.75)
SMALL_TRACE_TILE = 96
SMALL_TRACE_DELTA = 8.0
SMALL_TRACE_VERSION = 1
SMALL_TRACE_CHANNEL_WEIGHTS = (0.25, 0.85, 0.85)
SMALL_TRACE_SHORT_BITS = 16
DOT_MATRIX_VERSION = 1
DOT_MATRIX_TILE = 96
DOT_MATRIX_GRID = 8
DOT_MATRIX_CELL = DOT_MATRIX_TILE // DOT_MATRIX_GRID
DOT_MATRIX_DELTA = 7.5
DOT_MATRIX_CHANNEL_WEIGHTS = (0.80, 0.80, -0.28)


WATERMARK_LAYERS = {
    "lsb": True,
    "block": True,
    "dct": True,
    "dwt": True,
    "fft": True,
}

MENU_LABELS = {
    "watermark": "生成水印",
    "trace": "图片溯源",
    "manage": "图片管理",
    "role": "角色管理",
}

DEFAULT_ROLES = {
    "admin": {
        "label": "管理员",
        "menus": ["watermark", "trace", "manage", "role"],
    },
    "operator": {
        "label": "操作员",
        "menus": ["watermark", "trace", "manage"],
    },
    "viewer": {
        "label": "查看员",
        "menus": ["trace", "manage"],
    },
}

app = FastAPI(title=os.getenv("APP_NAME", "WatermarkSystem"))
app.state.generated_trace_ids = []
db_engine = None
db_store: DatabaseStore | None = None
db_error = ""


def ensure_dirs() -> None:
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def require_store() -> DatabaseStore:
    if db_store is None:
        raise HTTPException(status_code=503, detail="数据库不可用")
    return db_store


def database_ready() -> bool:
    return db_store is not None


def seed_database_defaults(store: DatabaseStore) -> None:
    if not store.read_roles():
        store.replace_roles(DEFAULT_ROLES)
    if ADMIN_USER and ADMIN_PASS and ADMIN_USER not in store.list_users():
        store.create_user(ADMIN_USER, ADMIN_PASS, "admin")


def initialize_database() -> None:
    global db_engine, db_error, db_store
    if not DB_ENABLED:
        return
    missing = [
        name
        for name, value in (
            ("DB_URL", DB_URL),
            ("ADMIN_USER", ADMIN_USER),
            ("ADMIN_PASS", ADMIN_PASS),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable: {missing[0]}")
    try:
        db_engine = create_engine(
            DB_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )
        db_store = DatabaseStore(db_engine)
        db_store.create_schema()
        seed_database_defaults(db_store)
        db_error = ""
    except SQLAlchemyError as exc:
        db_error = type(exc).__name__
        db_store = None
        raise RuntimeError("Database initialization failed") from exc


def db_clear_all() -> None:
    require_store().clear_all()


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
    return require_store().read_records()


def write_records(records: list[dict[str, Any]]) -> None:
    require_store().replace_records(records)


def add_record(record: dict[str, Any]) -> None:
    records = read_records()
    records.insert(0, record)
    write_records(records)


def read_detection_stats() -> dict[str, int]:
    stats = require_store().get_stats("detection_stats", {})
    return {
        "attempts": int(stats.get("attempts", 0) or 0),
        "successes": int(stats.get("successes", 0) or 0),
    }


def write_detection_stats(stats: dict[str, int]) -> None:
    ensure_dirs()
    normalized = {
        "attempts": int(stats.get("attempts", 0) or 0),
        "successes": int(stats.get("successes", 0) or 0),
    }
    require_store().set_stats("detection_stats", normalized)


def record_detection_result(success: bool) -> None:
    stats = read_detection_stats()
    stats["attempts"] += 1
    if success:
        stats["successes"] += 1
    write_detection_stats(stats)


def is_today_record(record: dict[str, Any]) -> bool:
    created_at = str(record.get("created_at") or "")
    return created_at.startswith(datetime.now().strftime("%Y-%m-%d"))


def read_watermark_stats() -> dict[str, dict[str, int]]:
    stats = require_store().get_stats("watermark_stats", {})
    daily = stats.get("daily", {})
    if not isinstance(daily, dict):
        daily = {}
    return {"daily": {str(day): int(count or 0) for day, count in daily.items()}}


def write_watermark_stats(stats: dict[str, Any]) -> None:
    ensure_dirs()
    daily = stats.get("daily", {})
    if not isinstance(daily, dict):
        daily = {}
    normalized = {"daily": {str(day): int(count or 0) for day, count in daily.items()}}
    require_store().set_stats("watermark_stats", normalized)


def read_roles() -> dict[str, Any]:
    return {"roles": require_store().read_roles()}


def read_users() -> dict[str, Any]:
    return {"users": require_store().list_users()}


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
    stats = read_watermark_stats()
    today = datetime.now().strftime("%Y-%m-%d")
    stats["daily"][today] = int(stats["daily"].get(today, 0)) + 1
    write_watermark_stats(stats)


def today_watermark_count(records: list[dict[str, Any]]) -> int:
    stats = read_watermark_stats()
    today = datetime.now().strftime("%Y-%m-%d")
    if today in stats["daily"]:
        return int(stats["daily"][today])
    return sum(1 for item in records if is_today_record(item))


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


def small_trace_short_code(trace_id: str) -> int:
    return watermark_payload_from_trace(trace_id) & ((1 << SMALL_TRACE_SHORT_BITS) - 1)


def small_trace_short_bits(trace_id: str) -> np.ndarray:
    code = small_trace_short_code(trace_id)
    return np.array(
        [1.0 if ((code >> shift) & 1) else -1.0 for shift in range(SMALL_TRACE_SHORT_BITS - 1, -1, -1)],
        dtype=np.float32,
    )


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


def robust_pattern(bit_index: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(ROBUST_MAGIC + bit_index * 7919)
    coarse = rng.choice(np.array([-1, 1], dtype=np.int16), size=(4, 4))
    repeat = max(1, int(np.ceil(size / 4)))
    pattern = np.kron(coarse, np.ones((repeat, repeat), dtype=np.int16))
    return pattern[:size, :size]


def layer_seed(trace_id: str, layer: str) -> int:
    digest = hashlib.blake2b(f"{trace_id}:{layer}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def pseudo_random_signs(trace_id: str, layer: str, count: int) -> np.ndarray:
    rng = np.random.default_rng(layer_seed(trace_id, layer))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count)


def apply_dct_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    signs = pseudo_random_signs(trace_id, "dct", blocks_y * blocks_x)
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            block = channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK]
            coeff = cv2.dct(block)
            coeff[3, 4] += signs[idx] * DCT_DELTA
            coeff[4, 3] += signs[idx] * DCT_DELTA
            channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK] = cv2.idct(coeff)
            idx += 1
    arr[:, :, 1] = channel
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dct_layer_score(image: Image.Image, trace_id: str) -> float:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    count = blocks_y * blocks_x
    if count < 16:
        return 0.0
    signs = pseudo_random_signs(trace_id, "dct", count)
    values = []
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            coeff = cv2.dct(channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK])
            values.append((coeff[3, 4] + coeff[4, 3]) * signs[idx])
            idx += 1
    values = np.array(values, dtype=np.float32)
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def apply_dwt_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 0]
    coeffs = pywt.dwt2(channel, "haar")
    ll, (lh, hl, hh) = coeffs
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    lh = lh + signs * DWT_DELTA
    rebuilt = pywt.idwt2((ll, (lh, hl, hh)), "haar")
    arr[: rebuilt.shape[0], : rebuilt.shape[1], 0] = rebuilt[: arr.shape[0], : arr.shape[1]]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dwt_layer_score(image: Image.Image, trace_id: str) -> float:
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 0]
    _, (lh, _, _) = pywt.dwt2(channel, "haar")
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    values = (lh * signs).ravel()
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def fft_pattern(shape: tuple[int, int], trace_id: str) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(layer_seed(trace_id, "fft"))
    pattern = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    radius_min = max(12, min(height, width) // 10)
    radius_max = max(radius_min + 4, min(height, width) // 4)
    for _ in range(96):
        angle = rng.uniform(0, np.pi)
        radius = rng.integers(radius_min, radius_max)
        y = int(round(center_y + np.sin(angle) * radius))
        x = int(round(center_x + np.cos(angle) * radius))
        y2 = int(round(center_y - np.sin(angle) * radius))
        x2 = int(round(center_x - np.cos(angle) * radius))
        if 0 <= y < height and 0 <= x < width:
            pattern[y, x] = 1.0
        if 0 <= y2 < height and 0 <= x2 < width:
            pattern[y2, x2] = 1.0
    return cv2.GaussianBlur(pattern, (0, 0), 1.2)


def apply_fft_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 2]
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    pattern = fft_pattern(channel.shape, trace_id)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    magnitude = magnitude * (1.0 + pattern * FFT_DELTA)
    rebuilt = np.real(np.fft.ifft2(np.fft.ifftshift(magnitude * np.exp(1j * phase))))
    arr[:, :, 2] = rebuilt
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def fft_layer_score(image: Image.Image, trace_id: str) -> float:
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 2]
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(channel))))
    pattern = fft_pattern(channel.shape, trace_id)
    mask = pattern > 0.05
    if int(mask.sum()) < 10:
        return 0.0
    selected = magnitude[mask]
    background = magnitude[~mask]
    return float(max(0.0, (selected.mean() - background.mean()) / (background.std() + 1e-6)))


def apply_frequency_layers(image: Image.Image, trace_id: str) -> Image.Image:
    return apply_fft_layer(apply_dwt_layer(apply_dct_layer(image, trace_id), trace_id), trace_id)


def layer_scores_for_image(image: Image.Image, trace_id: str) -> dict[str, float]:
    return {
        "dct": round(dct_layer_score(image, trace_id), 4),
        "dwt": round(dwt_layer_score(image, trace_id), 4),
        "fft": round(fft_layer_score(image, trace_id), 4),
    }


def fidelity_to_strength(value: str) -> float:
    fidelity = clamp_float(value, 0.75, 0.0, 1.0)
    return 1.0 - fidelity * 0.72


def small_crop_strength_to_scale(value: str | float | None) -> float:
    return clamp_float(value, 1.0, 0.0, 1.0)


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


def normalize_small_crop_density(value: str | None) -> str:
    text = str(value or "low").strip().lower()
    if text in {"medium", "中"}:
        return "medium"
    if text in {"high", "高"}:
        return "high"
    return "low"


def small_crop_density_offsets(density: str) -> list[tuple[int, int]]:
    quarter = SMALL_TRACE_TILE // 4
    half = SMALL_TRACE_TILE // 2
    offsets = [(0, 0), (half, half)]
    if density in {"medium", "high"}:
        offsets.extend([(half, 0), (0, half)])
    if density == "high":
        offsets.extend([(quarter, quarter), (quarter * 3, quarter), (quarter, quarter * 3), (quarter * 3, quarter * 3)])
    return offsets


def iter_aligned_small_trace_tiles(
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    record: dict[str, Any],
    target_scale: float = 1.0,
):
    if aligned.ndim != 3 or aligned.shape[2] != 3:
        return
    if valid_mask.shape != aligned.shape[:2] or target_scale <= 0:
        return
    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    density = normalize_small_crop_density(record.get("small_crop_trace_density"))
    for offset_x, offset_y in small_crop_density_offsets(density):
        for y in range(offset_y, original_height - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
            for x in range(offset_x, original_width - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
                x0 = max(0, int(round(x * target_scale)))
                y0 = max(0, int(round(y * target_scale)))
                x1 = min(width, int(round((x + SMALL_TRACE_TILE) * target_scale)))
                y1 = min(height, int(round((y + SMALL_TRACE_TILE) * target_scale)))
                if x1 <= x0 or y1 <= y0:
                    continue
                mask_tile = valid_mask[y0:y1, x0:x1]
                coverage = float(mask_tile.mean()) if mask_tile.size else 0.0
                if coverage < 0.70:
                    continue
                tile = aligned[y0:y1, x0:x1, :]
                normalized = cv2.resize(
                    tile,
                    (SMALL_TRACE_TILE, SMALL_TRACE_TILE),
                    interpolation=cv2.INTER_CUBIC,
                )
                yield {
                    "tile": normalized.astype(np.float32),
                    "position": (x, y),
                    "offset": (offset_x, offset_y),
                    "coverage": round(coverage, 4),
                }


def dot_matrix_bits_from_trace(trace_id: str) -> list[int]:
    payload = watermark_payload_from_trace(trace_id)
    return [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]


def dot_matrix_candidate_records() -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in read_records()
        if record.get("trace_id")
        and record.get("dot_matrix_trace_enabled")
        and record.get("dot_matrix_trace_version") == DOT_MATRIX_VERSION
    ]


def dot_matrix_position(bit_index: int, tile_size: int = DOT_MATRIX_TILE) -> tuple[int, int]:
    cell = max(2, tile_size // DOT_MATRIX_GRID)
    row = bit_index // DOT_MATRIX_GRID
    col = bit_index % DOT_MATRIX_GRID
    return col * cell + cell // 2, row * cell + cell // 2


def apply_dot_matrix_trace_layer(image: Image.Image, trace_id: str, strength: float = 1.0) -> Image.Image:
    strength = clamp_float(strength, 1.0, 0.0, 1.0)
    if strength <= 0:
        return image.convert("RGB")
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if height < DOT_MATRIX_TILE or width < DOT_MATRIX_TILE:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    bits = dot_matrix_bits_from_trace(trace_id)
    offsets = [
        (0, 0),
        (DOT_MATRIX_TILE // 2, 0),
        (0, DOT_MATRIX_TILE // 2),
        (DOT_MATRIX_TILE // 2, DOT_MATRIX_TILE // 2),
    ]
    radius = 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spot = np.exp(-(xx * xx + yy * yy) / 2.0).astype(np.float32)
    spot = spot / max(float(spot.max()), 1e-6)
    delta = DOT_MATRIX_DELTA * strength

    for offset_x, offset_y in offsets:
        for y in range(offset_y, height - DOT_MATRIX_TILE + 1, DOT_MATRIX_TILE):
            for x in range(offset_x, width - DOT_MATRIX_TILE + 1, DOT_MATRIX_TILE):
                for bit_index, bit in enumerate(bits):
                    cx, cy = dot_matrix_position(bit_index)
                    px = x + cx
                    py = y + cy
                    if px - radius < 0 or py - radius < 0 or px + radius >= width or py + radius >= height:
                        continue
                    sign = 1.0 if bit else -1.0
                    patch = arr[py - radius : py + radius + 1, px - radius : px + radius + 1, :]
                    for channel, weight in enumerate(DOT_MATRIX_CHANNEL_WEIGHTS):
                        patch[:, :, channel] = np.clip(patch[:, :, channel] + spot * delta * sign * weight, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dot_matrix_score_tile(tile: np.ndarray) -> tuple[int, float, int]:
    tile_size = min(tile.shape[:2])
    cell = max(2, tile_size // DOT_MATRIX_GRID)
    radius = max(1, min(3, cell // 5))
    votes = []
    strengths = []
    for bit_index in range(CODE_PAYLOAD_BITS):
        cx, cy = dot_matrix_position(bit_index, tile_size)
        y0 = max(0, cy - radius)
        y1 = min(tile.shape[0], cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(tile.shape[1], cx + radius + 1)
        patch = tile[y0:y1, x0:x1, :]
        if patch.size == 0:
            votes.append(0)
            strengths.append(0.0)
            continue
        yellow = (patch[:, :, 0] + patch[:, :, 1]) * 0.5 - patch[:, :, 2] * 0.35
        cell_y0 = max(0, cy - cell // 2)
        cell_y1 = min(tile.shape[0], cy + cell // 2)
        cell_x0 = max(0, cx - cell // 2)
        cell_x1 = min(tile.shape[1], cx + cell // 2)
        background_cell = tile[cell_y0:cell_y1, cell_x0:cell_x1, :]
        background = (background_cell[:, :, 0] + background_cell[:, :, 1]) * 0.5 - background_cell[:, :, 2] * 0.35
        score = float(yellow.mean() - background.mean())
        votes.append(1 if score >= 0 else 0)
        strengths.append(abs(score))
    code = 0
    for bit in votes:
        code = (code << 1) | bit
    return code, float(np.mean(strengths) if strengths else 0.0), len(votes)


def detect_dot_matrix_trace(image: Image.Image) -> dict[str, Any] | None:
    records = dot_matrix_candidate_records()
    if not records:
        return None
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    min_tile = 32
    if height < min_tile or width < min_tile:
        return None

    best_trace = None
    best_votes = 0
    best_distance = CODE_PAYLOAD_BITS + 1
    strength_counts: dict[str, float] = {}
    vote_counts: dict[str, int] = {}
    distance_counts: dict[str, int] = {}
    max_tile = min(DOT_MATRIX_TILE, height, width)
    tile_sizes = [size for size in (32, 36, 40, 44, 48, 56, 64, 72, 80, 88, 96) if size <= max_tile]
    if max_tile not in tile_sizes:
        tile_sizes.append(max_tile)
    for tile_size in tile_sizes:
        step = max(12, tile_size)
        offsets = sorted(set((0, tile_size // 2)))
        for offset_y in offsets:
            for offset_x in offsets:
                if offset_y > height - tile_size or offset_x > width - tile_size:
                    continue
                for y in range(offset_y, height - tile_size + 1, step):
                    for x in range(offset_x, width - tile_size + 1, step):
                        tile = arr[y : y + tile_size, x : x + tile_size, :]
                    code, strength, decided = dot_matrix_score_tile(tile)
                    if decided < CODE_PAYLOAD_BITS or strength < 0.10:
                        continue
                    payload = code
                    corrections = 0
                    magic_distance = hamming_distance(payload >> 32, ROBUST_MAGIC)
                    if magic_distance > 4:
                        continue
                    checksum = payload & 0xFFFF
                    crc_distance = hamming_distance(checksum, code_crc16(payload >> 16))
                    if crc_distance > 10:
                        continue
                    best_record = None
                    best_record_distance = CODE_PAYLOAD_BITS + 1
                    for _, expected_payload, record in records:
                        distance = hamming_distance(payload, expected_payload)
                        if distance < best_record_distance:
                            best_record = record
                            best_record_distance = distance
                    total_distance = best_record_distance + magic_distance + crc_distance + corrections
                    if not best_record or best_record_distance > 8 or total_distance > 24:
                        continue
                    trace_id = best_record.get("trace_id")
                    vote_weight = max(1, int(strength * 100)) * max(1, tile_size // 32)
                    vote_counts[trace_id] = vote_counts.get(trace_id, 0) + vote_weight
                    strength_counts[trace_id] = strength_counts.get(trace_id, 0.0) + strength * vote_weight
                    distance_counts[trace_id] = min(distance_counts.get(trace_id, CODE_PAYLOAD_BITS + 1), best_record_distance)

    for trace_id, votes in vote_counts.items():
        distance = distance_counts.get(trace_id, CODE_PAYLOAD_BITS + 1)
        if votes < 12 or distance > 8:
            continue
        if votes > best_votes or (votes == best_votes and distance < best_distance):
            best_trace = trace_id
            best_votes = votes
            best_distance = distance
    if not best_trace:
        return None
    record = next((item for trace_id, _, item in records if trace_id == best_trace), None)
    if not record:
        return None
    avg_strength = strength_counts[best_trace] / max(1, vote_counts[best_trace])
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "dot_matrix_trace",
        "mode_label": "点阵追溯水印",
        "created_at": record.get("created_at"),
        "confidence": int(min(96, max(76, 72 + best_votes * 2))),
        "phash_match": False,
        "status": "点阵水印恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(avg_strength, 4),
            "dwt": round(avg_strength, 4),
            "fft": round(avg_strength, 4),
        },
        "code_recovery": {
            "method": "dot_matrix_trace",
            "version": DOT_MATRIX_VERSION,
            "votes": best_votes,
            "distance": best_distance,
            "strength": round(avg_strength, 4),
        },
    }, record)


@lru_cache(maxsize=None)
def small_trace_marker_pattern(size: int) -> np.ndarray:
    rng = np.random.default_rng(ROBUST_MAGIC * 2039 + SMALL_TRACE_VERSION)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(10):
        fx = int(rng.integers(8, 20))
        fy = int(rng.integers(8, 20))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(12, 12))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=0.8, sigmaY=0.8) * 1.8
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_pattern(trace_id: str, size: int) -> np.ndarray:
    seed = layer_seed(trace_id, f"small-crop-trace-v{SMALL_TRACE_VERSION}")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(8):
        fx = int(rng.integers(7, 18))
        fy = int(rng.integers(7, 18))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    for _ in range(8):
        u = int(rng.integers(5, 16))
        v = int(rng.integers(5, 16))
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(10, 10))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=0.9, sigmaY=0.9) * 2.6
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_code_carriers(size: int) -> np.ndarray:
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(ROBUST_BITS):
        rng = np.random.default_rng(ROBUST_MAGIC * 2657 + SMALL_TRACE_VERSION * 977 + bit_index * 43691)

        fx = int(rng.integers(6, 16))
        fy = int(rng.integers(6, 16))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(5, 14))
        v = int(rng.integers(5, 14))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=0.8, sigmaY=0.8)

        carriers.append(normalize_carrier((fft * 0.44 + dct * 0.36 + dwt * 0.20) * window))
    return np.stack(carriers, axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_short_carriers(size: int) -> np.ndarray:
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(SMALL_TRACE_SHORT_BITS):
        rng = np.random.default_rng(ROBUST_MAGIC * 3251 + SMALL_TRACE_VERSION * 1543 + bit_index * 104729)
        fx = int(rng.integers(8, 18))
        fy = int(rng.integers(8, 18))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(6, 15))
        v = int(rng.integers(6, 15))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=0.75, sigmaY=0.75)
        carriers.append(normalize_carrier((fft * 0.44 + dct * 0.36 + dwt * 0.20) * window))
    return np.stack(carriers, axis=0).astype(np.float32)


def apply_small_crop_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 0.25,
    density: str = "low",
    fidelity_scale: float = 1.0,
) -> Image.Image:
    strength = small_crop_strength_to_scale(strength)
    if strength <= 0:
        return image.convert("RGB")
    density = normalize_small_crop_density(density)
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if height < SMALL_TRACE_TILE or width < SMALL_TRACE_TILE:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    marker = small_trace_marker_pattern(SMALL_TRACE_TILE)
    trace = small_trace_pattern(trace_id, SMALL_TRACE_TILE)
    bits = np.array([1.0 if bit else -1.0 for bit in watermark_bits_from_trace(trace_id)], dtype=np.float32)
    code = normalize_carrier(np.tensordot(bits, small_trace_code_carriers(SMALL_TRACE_TILE), axes=([0], [0])))
    short_code = normalize_carrier(
        np.tensordot(small_trace_short_bits(trace_id), small_trace_short_carriers(SMALL_TRACE_TILE), axes=([0], [0]))
    )
    spread_pattern = normalize_carrier(marker * 0.38 + trace * 0.16 + code * 0.20 + short_code * 0.42)
    delta = SMALL_TRACE_DELTA * strength * max(0.15, min(1.0, fidelity_scale))
    for offset_x, offset_y in small_crop_density_offsets(density):
        for y in range(offset_y, height - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
            for x in range(offset_x, width - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
                tile_rgb = arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, :]
                tile_gray = tile_rgb.mean(axis=2)
                texture = float(tile_gray.std())
                adaptive = min(0.82, max(0.24, texture / 42.0))
                for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
                    tile = arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, channel]
                    arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, channel] = np.clip(
                        tile + spread_pattern * delta * adaptive * weight,
                        0,
                        255,
                    )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def apply_code_layer(image: Image.Image, trace_id: str, strength_scale: float = 1.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    spread_pattern = normalize_carrier(code_marker_pattern(CODE_TILE) * 0.70 + code_trace_pattern(trace_id, CODE_TILE) * 0.30)
    for tile_y in range(0, height - CODE_TILE + 1, CODE_TILE):
        for tile_x in range(0, width - CODE_TILE + 1, CODE_TILE):
            tile_rgb = arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, :]
            tile_gray = tile_rgb.mean(axis=2)
            texture = float(tile_gray.std())
            adaptive = min(0.72, max(0.15, texture / 44.0))
            for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
                tile = arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, channel]
                arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, channel] = np.clip(
                    tile + spread_pattern * CODE_DELTA * adaptive * weight * strength_scale,
                    0,
                    255,
                )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def normalize_carrier(carrier: np.ndarray) -> np.ndarray:
    carrier = carrier.astype(np.float32)
    carrier = carrier - carrier.mean()
    rms = float(np.sqrt(np.mean(carrier * carrier)))
    if rms < 1e-6:
        return carrier
    return carrier / rms


@lru_cache(maxsize=None)
def code_cell_carriers(bit_index: int, size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(ROBUST_MAGIC * 31 + bit_index * 104729)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)

    mid_freq = [(2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2), (4, 5), (5, 4)]
    u, v = mid_freq[int(rng.integers(0, len(mid_freq)))]
    dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

    split_x = int(rng.integers(size // 3, size * 2 // 3))
    split_y = int(rng.integers(size // 3, size * 2 // 3))
    dwt = np.where(xx < split_x, 1.0, -1.0) * np.where(yy < split_y, 1.0, -1.0)

    fx = int(rng.integers(2, 6))
    fy = int(rng.integers(2, 6))
    phase = float(rng.random() * np.pi * 2)
    fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

    carriers = {
        "dct": normalize_carrier(dct * window),
        "dwt": normalize_carrier(dwt * window),
        "fft": normalize_carrier(fft * window),
    }
    carriers["combined"] = normalize_carrier(carriers["dct"] * 0.45 + carriers["dwt"] * 0.30 + carriers["fft"] * 0.25)
    return carriers


@lru_cache(maxsize=None)
def code_tile_carriers(size: int) -> np.ndarray:
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(ROBUST_BITS):
        rng = np.random.default_rng(ROBUST_MAGIC * 131 + bit_index * 65537)

        fx = int(rng.integers(7, 18))
        fy = int(rng.integers(7, 18))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(5, 15))
        v = int(rng.integers(5, 15))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=1.1, sigmaY=1.1)

        carrier = normalize_carrier((fft * 0.45 + dct * 0.35 + dwt * 0.20) * window)
        carriers.append(carrier)
    return np.stack(carriers, axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def code_trace_pattern(trace_id: str, size: int) -> np.ndarray:
    seed = layer_seed(trace_id, "trace-code-v4")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)

    for _ in range(10):
        fx = int(rng.integers(6, 19))
        fy = int(rng.integers(6, 19))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

    for _ in range(10):
        u = int(rng.integers(5, 17))
        v = int(rng.integers(5, 17))
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(12, 12))
    dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
    dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=1.3, sigmaY=1.3)
    pattern += dwt * 3.0
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def code_marker_pattern(size: int) -> np.ndarray:
    rng = np.random.default_rng(ROBUST_MAGIC * 1009 + CODE_WATERMARK_VERSION)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(16):
        fx = int(rng.integers(8, 22))
        fy = int(rng.integers(8, 22))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(16, 16))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=1.0, sigmaY=1.0)
    return normalize_carrier(pattern * window).astype(np.float32)


def decode_code_tile_signal(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < CODE_TILE or tile.shape[1] < CODE_TILE:
        return np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    signal = np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
        plane = tile[:CODE_TILE, :CODE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=2.6, sigmaY=2.6)
        signal += (plane - low) * weight
        total_weight += weight
    return normalize_carrier(signal / max(total_weight, 1e-6))


def code_scan_signal_grid(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int) -> tuple[np.ndarray, float, int]:
    height, width = arr.shape[:2]
    summed = np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    tiles = 0
    strengths = []
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_code_tile_signal(normalized)
            summed += signal
            strengths.append(float(np.std(signal)))
            tiles += 1
    if tiles < 2:
        return summed, 0.0, tiles
    return normalize_carrier(summed / tiles), float(np.mean(strengths)) if strengths else 0.0, tiles


def trace_tile_agreement(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int, pattern: np.ndarray) -> float:
    height, width = arr.shape[:2]
    scores = []
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_code_tile_signal(normalized)
            scores.append(float((signal * pattern).mean()))
    if len(scores) < 3:
        return 0.0
    positive = sum(1 for score in scores if score > 0.006)
    return positive / len(scores)


def apply_code_layer_shifted(image: Image.Image, trace_id: str) -> Image.Image:
    base = image.convert("RGB")
    shifted = Image.new("RGB", base.size)
    shifted.paste(base.crop((CODE_TILE // 2, CODE_TILE // 2, base.width, base.height)), (0, 0))
    marked = apply_code_layer(shifted, trace_id)
    restored = Image.new("RGB", base.size)
    restored.paste(marked.crop((0, 0, base.width - CODE_TILE // 2, base.height - CODE_TILE // 2)), (CODE_TILE // 2, CODE_TILE // 2))
    original_arr = np.array(base, dtype=np.float32)
    restored_arr = np.array(restored, dtype=np.float32)
    mask = restored_arr.sum(axis=2, keepdims=True) > 0
    mixed = np.where(mask, original_arr * 0.55 + restored_arr * 0.45, original_arr)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), "RGB")


def decode_code_tile_scores(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < CODE_TILE or tile.shape[1] < CODE_TILE:
        return np.zeros(ROBUST_BITS, dtype=np.float32)
    carriers = code_tile_carriers(CODE_TILE)
    scores = np.zeros(ROBUST_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
        plane = tile[:CODE_TILE, :CODE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=2.6, sigmaY=2.6)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * CODE_TILE * CODE_TILE)


def code_from_score_vector(scores: np.ndarray) -> tuple[int, float]:
    code = 0
    margins = []
    for score in scores:
        code = (code << 1) | (1 if score > 0 else 0)
        margins.append(abs(float(score)))
    return code, float(np.mean(margins)) if margins else 0.0


def decode_code_tile(tile: np.ndarray) -> tuple[int, float]:
    scores = decode_code_tile_scores(tile)
    return code_from_score_vector(scores)


def iter_code_scan_offsets(width: int, height: int, tile_w: int, tile_h: int):
    max_x = width - tile_w + 1
    max_y = height - tile_h + 1
    if max_x <= 0 or max_y <= 0:
        return
    step_x = max(16, tile_w // 4)
    step_y = max(16, tile_h // 4)
    for offset_y in range(0, min(tile_h, max_y), step_y):
        for offset_x in range(0, min(tile_w, max_x), step_x):
            yield offset_x, offset_y


def code_scan_grid(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int) -> tuple[int, float, int]:
    height, width = arr.shape[:2]
    summed = np.zeros(ROBUST_BITS, dtype=np.float32)
    tiles = 0
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            summed += decode_code_tile_scores(normalized)
            tiles += 1
    if tiles < 2:
        return 0, 0.0, tiles
    averaged = summed / max(1, tiles)
    code, strength = code_from_score_vector(averaged)
    agreement = float(np.mean(np.abs(summed) / (np.abs(summed).max() + 1e-6)))
    return code, strength * agreement, tiles


def decode_small_trace_signal(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros((SMALL_TRACE_TILE, SMALL_TRACE_TILE), dtype=np.float32)
    signal = np.zeros((SMALL_TRACE_TILE, SMALL_TRACE_TILE), dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        signal += (plane - low) * weight
        total_weight += weight
    return normalize_carrier(signal / max(total_weight, 1e-6))


def decode_small_trace_code_scores(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros(CODE_PHYSICAL_BITS, dtype=np.float32)
    carriers = small_trace_code_carriers(SMALL_TRACE_TILE)
    scores = np.zeros(CODE_PHYSICAL_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * SMALL_TRACE_TILE * SMALL_TRACE_TILE)


def decode_small_trace_short_scores(tile: np.ndarray) -> np.ndarray:
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros(SMALL_TRACE_SHORT_BITS, dtype=np.float32)
    carriers = small_trace_short_carriers(SMALL_TRACE_TILE)
    scores = np.zeros(SMALL_TRACE_SHORT_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * SMALL_TRACE_TILE * SMALL_TRACE_TILE)


def short_code_from_scores(scores: np.ndarray) -> tuple[int, float]:
    code = 0
    margins = []
    for score in scores:
        code = (code << 1) | (1 if score > 0 else 0)
        margins.append(abs(float(score)))
    return code, float(np.mean(margins)) if margins else 0.0


def record_from_short_code_match(
    short_code: int,
    code_records: list[tuple[str, int, dict[str, Any]]],
    max_errors: int,
    min_gap: int,
) -> dict[str, Any] | None:
    best_record = None
    best_distance = SMALL_TRACE_SHORT_BITS + 1
    second_distance = SMALL_TRACE_SHORT_BITS + 1
    for trace_id, _, record in code_records:
        distance = hamming_distance(short_code, small_trace_short_code(trace_id))
        if distance < best_distance:
            second_distance = best_distance
            best_record = record
            best_distance = distance
        elif distance < second_distance:
            second_distance = distance
    if best_record and best_distance <= max_errors and second_distance - best_distance >= min_gap:
        return best_record
    return None


def match_small_trace_code(code: int, code_records: list[tuple[str, int, dict[str, Any]]], max_errors: int = 10) -> tuple[dict[str, Any] | None, int, int]:
    payload, corrections = recover_payload_from_code(code)
    magic_distance = hamming_distance(payload >> 32, ROBUST_MAGIC)
    if magic_distance > 3:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    body_and_magic = payload >> 16
    checksum = payload & 0xFFFF
    crc_distance = hamming_distance(checksum, code_crc16(body_and_magic))
    if crc_distance > 12:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    best_record = None
    best_distance = CODE_PAYLOAD_BITS + 1
    second_distance = CODE_PAYLOAD_BITS + 1
    for _, expected_payload, record in code_records:
        distance = hamming_distance(payload, expected_payload)
        if distance < best_distance:
            second_distance = best_distance
            best_record = record
            best_distance = distance
        elif distance < second_distance:
            second_distance = distance
    total_distance = best_distance + magic_distance + crc_distance + corrections
    if best_record and best_distance <= max_errors and total_distance <= 28 and second_distance - best_distance >= 3:
        return best_record, best_distance, second_distance
    return None, best_distance, second_distance


def iter_small_trace_windows(width: int, height: int):
    tile_shapes = []
    for tile_w in (40, 48, 56, 64, 80, 96, 112, 128):
        for aspect in (0.85, 1.0, 1.15):
            tile_h = int(round(tile_w * aspect / 8)) * 8
            shape = (tile_w, max(40, tile_h))
            if shape not in tile_shapes:
                tile_shapes.append(shape)
    for tile_w, tile_h in tile_shapes:
        if width < tile_w or height < tile_h:
            continue
        step_x = max(20, tile_w // 2)
        step_y = max(20, tile_h // 2)
        for y in range(0, height - tile_h + 1, step_y):
            for x in range(0, width - tile_w + 1, step_x):
                yield x, y, tile_w, tile_h


def detect_small_crop_trace(image: Image.Image) -> dict[str, Any] | None:
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    records = read_records()
    raw_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
        and record.get("small_crop_trace_enabled")
        and record.get("small_crop_trace_version") == SMALL_TRACE_VERSION
    ]
    generated_trace_ids = list(getattr(app.state, "generated_trace_ids", []))
    persistent_candidate_mode = False
    if generated_trace_ids:
        order = {trace_id: index for index, trace_id in enumerate(generated_trace_ids)}
        candidate_records = [item for item in raw_records if item[0] in order]
        candidate_records.sort(key=lambda item: order[item[0]])
        candidate_records = candidate_records[: min(len(candidate_records), 8)]
    else:
        candidate_records = raw_records[: min(len(raw_records), 80)]
        persistent_candidate_mode = True
    if not candidate_records:
        return None

    visual_evidence: dict[str, tuple[int, float, float]] = {}
    verified_candidates = []
    for trace_id, payload, record in candidate_records:
        consistent, inliers, ratio, residual_score = record_visual_consistency(image, record)
        if not consistent:
            continue
        verified_candidates.append((trace_id, payload, record))
        visual_evidence[trace_id] = (inliers, ratio, residual_score)
    candidate_records = verified_candidates
    if not candidate_records:
        return None

    marker = small_trace_marker_pattern(SMALL_TRACE_TILE)
    scales = (1.0, 0.92, 1.08)
    best_trace = None
    best_votes = 0
    best_strength = 0.0
    vote_counts: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    strength_counts: dict[str, float] = {}
    for scale in scales:
        if scale == 1.0:
            arr = arr0
        else:
            arr = cv2.resize(arr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = arr.shape[:2]
        if height < 120 or width < 120:
            continue
        for x, y, tile_w, tile_h in iter_small_trace_windows(width, height):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (SMALL_TRACE_TILE, SMALL_TRACE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_small_trace_signal(normalized)
            marker_score = float((signal * marker).mean())
            if marker_score < 0.052:
                continue
            scores = decode_small_trace_code_scores(normalized)
            code, code_strength = code_from_score_vector(scores)
            best_record, _, _ = match_small_trace_code(code, candidate_records)
            short_strength = 0.0
            matched_by_short_code = False
            if not best_record:
                short_code, short_strength = short_code_from_scores(decode_small_trace_short_scores(normalized))
                best_record = record_from_short_code_match(
                    short_code,
                    candidate_records,
                    4 if not persistent_candidate_mode else 2,
                    2 if not persistent_candidate_mode else 3,
                )
                matched_by_short_code = bool(best_record)
            if not best_record:
                continue
            trace_id = best_record.get("trace_id")
            trace_score = float((signal * small_trace_pattern(trace_id, SMALL_TRACE_TILE)).mean())
            score = marker_score * 0.40 + max(code_strength, short_strength) * 2.8 + max(0.0, trace_score) * 0.24
            if matched_by_short_code and trace_score < 0.034:
                continue
            if (
                marker_score < 0.060
                or max(code_strength, short_strength) < 0.005
                or trace_score < 0.020
                or score < 0.060
            ):
                continue
            weighted_vote = int(max(1, score * 10000) * max(1, min(5, int(round(tile_w * tile_h / (96 * 96))))))
            vote_counts[trace_id] = vote_counts.get(trace_id, 0) + weighted_vote
            hit_counts[trace_id] = hit_counts.get(trace_id, 0) + 1
            strength_counts[trace_id] = strength_counts.get(trace_id, 0.0) + score * weighted_vote
    for trace_id, votes in vote_counts.items():
        hits = hit_counts.get(trace_id, 0)
        min_hits = 5 if not persistent_candidate_mode else 10
        min_votes = 24000 if not persistent_candidate_mode else 36000
        if hits < min_hits or votes < min_votes:
            continue
        avg_strength = strength_counts[trace_id] / max(1, votes)
        if votes > best_votes or (votes == best_votes and avg_strength > best_strength):
            best_trace = trace_id
            best_votes = votes
            best_strength = avg_strength
    min_final_votes = 24000 if not persistent_candidate_mode else 36000
    if not best_trace or best_votes < min_final_votes:
        return None
    record = next((item for trace_id, _, item in candidate_records if trace_id == best_trace), None)
    if not record:
        return None
    evidence = visual_evidence.get(best_trace)
    if not evidence:
        return None
    visual_inliers, visual_ratio, residual_score = evidence
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "watermark_code",
        "mode_label": "小面积截图频域水印码",
        "created_at": record.get("created_at"),
        "confidence": int(min(96, max(78, 72 + best_votes / 1200))),
        "phash_match": False,
        "status": "小截图水印码恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(best_strength, 4),
            "dwt": round(best_strength, 4),
            "fft": round(best_strength, 4),
        },
        "code_recovery": {
            "method": "small_crop_trace_redundancy",
            "version": SMALL_TRACE_VERSION,
            "votes": best_votes,
            "strength": round(best_strength, 4),
            "visual_inliers": visual_inliers,
            "visual_ratio": round(visual_ratio, 3),
            "residual_score": round(residual_score, 4),
        },
    }, record)


def detect_watermark_code(image: Image.Image) -> dict[str, Any] | None:
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    scales = (1.0, 0.95, 1.05)
    best_trace = None
    best_strength = 0.0
    best_votes = 0
    records = read_records()
    raw_code_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
    ]
    generated_trace_ids = list(getattr(app.state, "generated_trace_ids", []))
    persistent_candidate_mode = False
    if generated_trace_ids:
        order = {trace_id: index for index, trace_id in enumerate(generated_trace_ids)}
        code_records = [item for item in raw_code_records if item[0] in order]
        code_records.sort(key=lambda item: order[item[0]])
        code_records = code_records[: min(len(code_records), 8)]
    else:
        code_records = raw_code_records[: min(len(raw_code_records), 100)]
        persistent_candidate_mode = True
    if not code_records:
        return None

    visual_evidence: dict[str, tuple[int, float, float]] = {}
    verified_code_records = []
    for trace_id, payload, record in code_records:
        consistent, inliers, ratio, residual_score = record_visual_consistency(image, record)
        if not consistent:
            continue
        verified_code_records.append((trace_id, payload, record))
        visual_evidence[trace_id] = (inliers, ratio, residual_score)
    code_records = verified_code_records
    if not code_records:
        return None

    def match_trace_code(code: int, max_errors: int = 7) -> tuple[str | None, int, int]:
        payload, corrections = recover_payload_from_code(code)
        magic_distance = hamming_distance(payload >> 32, ROBUST_MAGIC)
        if magic_distance > 2:
            return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
        body_and_magic = payload >> 16
        checksum = payload & 0xFFFF
        crc_distance = hamming_distance(checksum, code_crc16(body_and_magic))
        if crc_distance > 12:
            return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
        best = None
        best_distance = CODE_PAYLOAD_BITS + 1
        second_distance = CODE_PAYLOAD_BITS + 1
        for trace_id, expected_code, _ in code_records:
            distance = hamming_distance(payload, expected_code)
            if distance < best_distance:
                second_distance = best_distance
                best = trace_id
                best_distance = distance
            elif distance < second_distance:
                second_distance = distance
        total_distance = best_distance + magic_distance + crc_distance + corrections
        if best and best_distance <= max_errors and total_distance <= 24 and second_distance - best_distance >= 4:
            return best, best_distance, second_distance
        return None, best_distance, second_distance

    def match_trace_signal(signal: np.ndarray) -> tuple[str | None, float, float]:
        marker_score = float((signal * code_marker_pattern(CODE_TILE)).mean())
        if marker_score < 0.055:
            return None, marker_score, -1.0
        best_trace = None
        best_score = -1.0
        second_score = -1.0
        for trace_id, _, _ in code_records:
            pattern = code_trace_pattern(trace_id, CODE_TILE)
            score = float((signal * pattern).mean())
            if score > best_score:
                second_score = best_score
                best_trace = trace_id
                best_score = score
            elif score > second_score:
                second_score = score
        return best_trace, marker_score * 0.7 + best_score * 0.3, second_score

    for scale in scales:
        if scale == 1.0:
            arr = arr0
        else:
            arr = cv2.resize(arr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = arr.shape[:2]
        if height < 120 or width < 120:
            continue
        vote_counts: dict[str, int] = {}
        hit_counts: dict[str, int] = {}
        strength_counts: dict[str, float] = {}
        distance_counts: dict[str, int] = {}
        tile_shapes = []
        for tile_w in (144, 152, 160, 168, 176):
            for aspect in (0.94, 1.0, 1.06):
                tile_h = int(round(tile_w * aspect / 8)) * 8
                shape = (tile_w, max(48, tile_h))
                if shape not in tile_shapes:
                    tile_shapes.append(shape)
        for tile_w, tile_h in tile_shapes:
            if height < tile_h or width < tile_w:
                continue
            for offset_x, offset_y in iter_code_scan_offsets(width, height, tile_w, tile_h):
                signal, signal_strength, signal_tiles = code_scan_signal_grid(arr, tile_w, tile_h, offset_x, offset_y)
                if signal_tiles >= 3:
                    signal_trace, score, second_score = match_trace_signal(signal)
                    if signal_trace and score >= 0.05 and (score - second_score) >= 0.003:
                        agreement = trace_tile_agreement(
                            arr,
                            tile_w,
                            tile_h,
                            offset_x,
                            offset_y,
                            code_marker_pattern(CODE_TILE),
                        )
                        if agreement < 0.40:
                            continue
                        weighted_vote = int(max(1, score * 10000) * min(signal_tiles, 16))
                        vote_counts[signal_trace] = vote_counts.get(signal_trace, 0) + weighted_vote
                        hit_counts[signal_trace] = hit_counts.get(signal_trace, 0) + 1
                        strength_counts[signal_trace] = strength_counts.get(signal_trace, 0.0) + score * weighted_vote
                        distance_counts[signal_trace] = distance_counts.get(signal_trace, 0) + 0
        for trace_id, votes in vote_counts.items():
            min_hits = 2 if persistent_candidate_mode else 3
            min_votes = 30000 if persistent_candidate_mode else 30000
            strong_single_hit = persistent_candidate_mode and hit_counts.get(trace_id, 0) >= 1 and votes >= 23000
            if hit_counts.get(trace_id, 0) < min_hits and not strong_single_hit:
                continue
            if votes < min_votes and not strong_single_hit:
                continue
            avg_distance = distance_counts[trace_id] / max(1, hit_counts[trace_id])
            if avg_distance > 7:
                continue
            avg_strength = strength_counts[trace_id] / max(1, votes)
            if votes > best_votes or (votes == best_votes and avg_strength > best_strength):
                best_trace = trace_id
                best_votes = votes
                best_strength = avg_strength
    min_final_votes = 23000 if persistent_candidate_mode else 30000
    if not best_trace or best_votes < min_final_votes:
        return None
    record = next((item for trace_id, _, item in code_records if trace_id == best_trace), None)
    if not record:
        return None
    evidence = visual_evidence.get(best_trace)
    if not evidence:
        return None
    visual_inliers, visual_ratio, residual_score = evidence
    confidence = int(min(98, max(75, 70 + best_votes * 8)))
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "watermark_code",
        "mode_label": "多尺度频域水印码",
        "created_at": record.get("created_at"),
        "confidence": confidence,
        "phash_match": False,
        "status": "水印码恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(best_strength, 4),
            "dwt": round(best_strength, 4),
            "fft": round(best_strength, 4),
        },
        "code_recovery": {
            "method": "multi_scale_dct_dwt_fft",
            "version": CODE_WATERMARK_VERSION,
            "votes": best_votes,
            "strength": round(best_strength, 4),
            "visual_inliers": visual_inliers,
            "visual_ratio": round(visual_ratio, 3),
            "residual_score": round(residual_score, 4),
        },
    }, record)


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
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index") / f"{safe_id}.npz"
    descriptors = extract_feature_descriptors(image)
    save_feature_descriptors(DATA_DIR / relative, descriptors)
    return relative.as_posix()


def save_record_feature_index_v4(image: Image.Image, record_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index_v4") / f"{safe_id}.npz"
    index = extract_v4_feature_index(image)
    save_v4_feature_index(DATA_DIR / relative, index)
    return relative.as_posix()


def record_feature_index_path(record: dict[str, Any]) -> Path | None:
    raw = str(record.get("feature_index_path") or "").strip()
    if raw:
        relative = Path(raw.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        return DATA_DIR / relative
    record_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record.get("id") or ""))
    if not record_id:
        return None
    return DATA_DIR / "feature_index" / f"{record_id}.npz"


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
    query_ratio = image.width / max(1, image.height)
    generated_trace_ids = list(getattr(app.state, "generated_trace_ids", []))
    recent_trace_ids = list(generated_trace_ids[:FEATURE_RECENT_BACKFILL])
    for record in records:
        trace_id = record.get("trace_id")
        if len(recent_trace_ids) >= FEATURE_RECENT_BACKFILL:
            break
        if trace_id and record.get("created_at") and trace_id not in recent_trace_ids:
            recent_trace_ids.append(trace_id)
    recent_order = {
        trace_id: index
        for index, trace_id in enumerate(recent_trace_ids[:FEATURE_RECENT_RESERVE])
    }
    backfill_trace_ids = set(recent_trace_ids)

    for record in records:
        if record.get("trace_id") not in backfill_trace_ids:
            continue
        path = record_feature_index_path(record)
        if path and path.exists():
            continue
        url = record.get("download_url")
        record_id = record.get("id")
        if not record_id or not url or not url.startswith("/uploads/"):
            continue
        image_path = UPLOAD_DIR / url.replace("/uploads/", "")
        try:
            with Image.open(image_path) as target:
                save_record_feature_index(target.convert("RGB"), str(record_id))
        except (OSError, ValueError):
            continue

    query_descriptors = extract_feature_descriptors(image)
    feature_ranked = []
    remaining = []
    for record in records:
        path = record_feature_index_path(record)
        descriptors = (
            load_feature_descriptors(path)
            if path is not None and path.exists()
            else np.empty((0, 32), dtype=np.uint8)
        )
        match_count, match_quality = descriptor_match_score(query_descriptors, descriptors)
        if match_count >= FEATURE_MATCH_MIN_GOOD:
            feature_ranked.append({
                **record,
                "_feature_match_count": match_count,
                "_feature_match_quality": match_quality,
            })
        else:
            remaining.append(record)

    feature_ranked.sort(
        key=lambda record: (
            -int(record.get("_feature_match_count", 0)),
            -float(record.get("_feature_match_quality", 0.0)),
        )
    )

    def ratio_distance(record: dict[str, Any]) -> float:
        recorded_width = record.get("image_width")
        recorded_height = record.get("image_height")
        if recorded_width and recorded_height:
            try:
                target_ratio = float(recorded_width) / max(1.0, float(recorded_height))
                return abs(target_ratio - query_ratio)
            except (TypeError, ValueError):
                pass
        url = record.get("download_url")
        if not url or not url.startswith("/uploads/"):
            return float("inf")
        path = UPLOAD_DIR / url.replace("/uploads/", "")
        try:
            with Image.open(path) as target:
                target_ratio = target.width / max(1, target.height)
        except Exception:
            return float("inf")
        return abs(target_ratio - query_ratio)

    feature_trace_ids = {record.get("trace_id") for record in feature_ranked}
    recent_ranked = sorted(
        [
            record
            for record in remaining
            if record.get("trace_id") in recent_order
            and record.get("trace_id") not in feature_trace_ids
        ],
        key=lambda record: recent_order[record.get("trace_id")],
    )[:FEATURE_RECENT_RESERVE]
    reserved_ids = {id(record) for record in recent_ranked}
    aspect_ranked = sorted(
        [record for record in remaining if id(record) not in reserved_ids],
        key=ratio_distance,
    )
    return feature_ranked + recent_ranked + aspect_ranked


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


def bits_from_bytes(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_from_bits(bits: list[int]) -> bytes:
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def lsb_bits_from_pixels(pixels: Any):
    for pixel in pixels:
        yield pixel[0] & 1
        yield pixel[1] & 1
        yield pixel[2] & 1


def valid_watermark_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required_strings = ("id", "trace_id", "user_id", "mode", "created_at")
    for key in required_strings:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    if not re.fullmatch(r"[0-9a-f]{32}", payload["id"]):
        return False
    if not re.fullmatch(r"[0-9A-F]{32}", str(payload.get("evidence_uuid", ""))):
        return False
    if payload.get("mode") not in {"lsb", "dct", "dwt", "fft", "hybrid"}:
        return False
    return True


def packet_from_payload(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(data).to_bytes(4, "big") + data


def write_packet_to_pixels(pixels: list[tuple[int, int, int]], packet: bytes) -> list[tuple[int, int, int]]:
    bits = bits_from_bytes(packet)
    capacity = len(pixels) * 3
    if len(bits) > capacity:
        raise HTTPException(status_code=400, detail="图片尺寸过小，无法嵌入水印信息")

    out = []
    idx = 0
    for pixel in pixels:
        channels = list(pixel)
        for channel in range(3):
            if idx < len(bits):
                channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                idx += 1
        out.append(tuple(channels))
    return out


def embed_lsb(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    pixels = list(rgb.getdata())
    rgb.putdata(write_packet_to_pixels(pixels, packet))
    return embed_block_lsb(rgb, payload)


def iter_block_origins(width: int, height: int):
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return
    for y in range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE):
            yield x, y


def embed_block_lsb(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    bits = bits_from_bytes(packet)
    width, height = rgb.size
    pixels = rgb.load()
    required = len(bits)
    block_capacity = BLOCK_SIZE * BLOCK_SIZE * 3
    if block_capacity < required:
        return rgb

    for ox, oy in iter_block_origins(width, height) or []:
        idx = 0
        for y in range(oy, oy + BLOCK_SIZE):
            for x in range(ox, ox + BLOCK_SIZE):
                channels = list(pixels[x, y])
                for channel in range(3):
                    if idx < required:
                        channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                        idx += 1
                pixels[x, y] = tuple(channels)
                if idx >= required:
                    break
            if idx >= required:
                break
    return rgb


def decode_bits_from_pixels(pixels: Any) -> dict[str, Any] | None:
    bit_iter = lsb_bits_from_pixels(pixels)
    header_bits = []
    for _ in range(64):
        try:
            header_bits.append(next(bit_iter))
        except StopIteration:
            return None
    if len(header_bits) < 64:
        return None
    header = bytes_from_bits(header_bits)
    if len(header) < 8 or header[:4] != MAGIC:
        return None

    size = int.from_bytes(header[4:8], "big")
    if size <= 0 or size > 8192:
        return None
    payload_bits = []
    for _ in range(size * 8):
        try:
            payload_bits.append(next(bit_iter))
        except StopIteration:
            return None
    payload = bytes_from_bits(payload_bits)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not valid_watermark_payload(decoded):
        return None
    return decoded


def extract_lsb(image: Image.Image) -> dict[str, Any]:
    payload = extract_full_lsb(image)
    if payload:
        return payload
    payload = extract_block_lsb(image)
    if payload:
        return payload
    raise HTTPException(status_code=404, detail="未检测到可识别的隐式水印")


def extract_full_lsb(image: Image.Image) -> dict[str, Any] | None:
    rgb = image.convert("RGB")
    payload = decode_bits_from_pixels(list(rgb.getdata()))
    if payload:
        return payload
    return None


def extract_block_lsb(image: Image.Image) -> dict[str, Any] | None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return None
    pixels = rgb.load()
    step = 1 if width * height <= 250_000 else 8
    y_origins = range(0, height - BLOCK_SIZE + 1, step)
    x_origins = range(0, width - BLOCK_SIZE + 1, step)
    if width * height > 1_000_000:
        y_origins = range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE)
        x_origins = range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE)
    for oy in y_origins:
        for ox in x_origins:
            payload = decode_bits_from_pixels(
                pixels[x, y]
                for y in range(oy, oy + BLOCK_SIZE)
                for x in range(ox, ox + BLOCK_SIZE)
            )
            if payload:
                return payload
    return None


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
    records = read_records()
    copyright_records = [item for item in records if item.get("copyright_enabled") and item.get("copyright_text")]
    if not copyright_records:
        return None

    # The visible copyright layer is human-readable but not OCR-backed in this lightweight version.
    # If hidden extraction fails and the image contains strong watermark-like bright overlays,
    # return the configured copyright source as a lower-confidence fallback.
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    total = max(1, sum(histogram))
    bright_ratio = sum(histogram[205:]) / total
    if bright_ratio < 0.05:
        return None

    record = copyright_records[0]
    text = str(record.get("copyright_text", "")).strip()
    user_id = "QQ:757675150" if "757675150" in text else text.replace("©", "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12].upper()
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": f"VISIBLE-{digest}",
        "user_id": user_id or record.get("user_id") or "VISIBLE-WATERMARK",
        "mode": "visible",
        "mode_label": "可见版权水印",
        "created_at": record.get("created_at"),
        "confidence": 68,
        "phash_match": False,
        "status": "检测到可见版权水印",
        "extracted_at": now_text(),
    }, record)


def image_to_cv_gray(image: Image.Image, max_side: int = 1200):
    rgb = image.convert("RGB")
    arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    height, width = arr.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        arr = cv2.resize(arr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return arr


def record_visual_consistency(image: Image.Image, record: dict[str, Any]) -> tuple[bool, int, float, float]:
    url = record.get("download_url")
    original_url = record.get("original_url")
    if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
        return False, 0, 0.0, 0.0
    path = UPLOAD_DIR / url.replace("/uploads/", "")
    original_path = UPLOAD_DIR / original_url.replace("/uploads/", "")
    if not path.exists() or not original_path.exists():
        return False, 0, 0.0, 0.0
    try:
        query = image_to_cv_gray(image)
        target = image_to_cv_gray(Image.open(path))
    except Exception:
        return False, 0, 0.0, 0.0
    inliers, ratio = feature_match_score(query, target)
    residual_score = robust_residual_score(image, original_path, path, min_inliers=18, min_ratio=0.32)
    standard_match = inliers >= 18 and ratio >= 0.32 and residual_score >= 0.08
    strong_visual_small_crop_match = inliers >= 30 and ratio >= 0.65 and residual_score >= 0.06
    return (standard_match or strong_visual_small_crop_match), inliers, ratio, residual_score


def residual_candidate_evidence(image: Image.Image) -> dict[str, Any] | None:
    records = [record for record in read_records() if record.get("robust_watermark")]
    if not records:
        return None

    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    best_residual = 0.0
    for record in records:
        consistent, inliers, ratio, residual_score = record_visual_consistency(image, record)
        if not consistent:
            continue
        if residual_score > best_residual or (
            residual_score == best_residual and (inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio))
        ):
            best_record = record
            best_inliers = inliers
            best_ratio = ratio
            best_residual = residual_score

    if not best_record or best_residual < 0.12:
        return None

    return {
        "candidate_id": best_record.get("id"),
        "candidate_trace_id": best_record.get("trace_id"),
        "visual_inliers": best_inliers,
        "visual_ratio": round(best_ratio, 3),
        "residual_score": round(best_residual, 4),
    }


def detect_by_residual_match(image: Image.Image) -> dict[str, Any] | None:
    # Visual and residual similarity can rank candidates but cannot prove that the
    # query contains a watermark. Code-backed detectors perform final attribution.
    return None


def feature_match_score(query_gray, target_gray) -> tuple[int, float]:
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good.append(first)

    if len(good) < 10:
        return len(good), 0.0

    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None:
        return len(good), 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    return inliers, ratio


def feature_match_homography(query_gray, target_gray):
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return None, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good.append(first)

    if len(good) < 10:
        return None, len(good), 0.0

    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    query_to_target, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None or query_to_target is None:
        return None, len(good), 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    try:
        target_to_query = np.linalg.inv(query_to_target)
    except np.linalg.LinAlgError:
        return None, inliers, ratio
    return target_to_query, inliers, ratio


def align_query_to_record(image: Image.Image, record: dict[str, Any]) -> dict[str, Any] | None:
    url = record.get("download_url")
    if not url or not url.startswith("/uploads/"):
        return None
    target_path = UPLOAD_DIR / url.replace("/uploads/", "")
    if not target_path.exists():
        return None
    try:
        query_image = resize_for_residual(image)
        original_target = Image.open(target_path).convert("RGB")
        target_image = resize_for_residual(original_target)
    except Exception:
        return None

    query = np.asarray(query_image, dtype=np.uint8)
    target = np.asarray(target_image, dtype=np.uint8)
    query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    target_to_query, inliers, ratio = feature_match_homography(query_gray, target_gray)
    if target_to_query is None or inliers < 18 or ratio < 0.32:
        return None
    try:
        query_to_target = np.linalg.inv(target_to_query)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(query_to_target).all() or abs(float(np.linalg.det(query_to_target))) < 1e-9:
        return None

    target_height, target_width = target.shape[:2]
    aligned = cv2.warpPerspective(
        query,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid_mask = cv2.warpPerspective(
        np.ones(query.shape[:2], dtype=np.uint8) * 255,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    coverage = float(valid_mask.mean())
    if coverage < 0.05 or coverage > 1.0:
        return None
    target_scale = target_width / max(1, original_target.width)
    return {
        "image": aligned,
        "valid_mask": valid_mask,
        "inliers": inliers,
        "ratio": round(ratio, 4),
        "coverage": round(coverage, 4),
        "target_scale": target_scale,
        "target_size": (target_width, target_height),
        "homography": query_to_target,
    }


def resize_for_residual(image: Image.Image, max_side: int = 1200) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    return rgb


def robust_residual_score(
    query_image: Image.Image,
    original_path: Path,
    watermarked_path: Path,
    min_inliers: int = 80,
    min_ratio: float = 0.80,
) -> float:
    query = np.array(resize_for_residual(query_image), dtype=np.float32)
    watermarked = np.array(resize_for_residual(Image.open(watermarked_path)), dtype=np.float32)
    original = np.array(resize_for_residual(Image.open(original_path)), dtype=np.float32)
    query_gray = cv2.cvtColor(query.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(watermarked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    homography, inliers, ratio = feature_match_homography(query_gray, target_gray)
    if homography is None or inliers < min_inliers or ratio < min_ratio:
        return 0.0

    query_height, query_width = query.shape[:2]
    warped_watermarked = cv2.warpPerspective(watermarked, homography, (query_width, query_height))
    warped_original = cv2.warpPerspective(original, homography, (query_width, query_height))
    valid = cv2.warpPerspective(
        np.ones(watermarked.shape[:2], dtype=np.uint8) * 255,
        homography,
        (query_width, query_height),
    ) > 0
    if int(valid.sum()) < query_width * query_height * 0.30:
        return 0.0

    expected = (warped_watermarked[:, :, ROBUST_CHANNEL] - warped_original[:, :, ROBUST_CHANNEL])[valid]
    observed = (query[:, :, ROBUST_CHANNEL] - warped_original[:, :, ROBUST_CHANNEL])[valid]
    expected = expected - expected.mean()
    observed = observed - observed.mean()
    expected_norm = float(np.linalg.norm(expected))
    observed_norm = float(np.linalg.norm(observed))
    if expected_norm < 1e-6 or observed_norm < 1e-6:
        return 0.0
    return float(np.dot(expected, observed) / (expected_norm * observed_norm))


def detect_by_visual_match(image: Image.Image) -> dict[str, Any] | None:
    records = read_records()
    if not records:
        return None

    query = image_to_cv_gray(image)
    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    for record in records:
        if not record.get("robust_watermark"):
            continue
        url = record.get("download_url")
        original_url = record.get("original_url")
        if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
            continue
        path = UPLOAD_DIR / url.replace("/uploads/", "")
        original_path = UPLOAD_DIR / original_url.replace("/uploads/", "")
        if not path.exists() or not original_path.exists():
            continue
        try:
            target = image_to_cv_gray(Image.open(path))
        except Exception:
            continue
        inliers, ratio = feature_match_score(query, target)
        if inliers >= 80 and ratio >= 0.80:
            residual_score = robust_residual_score(image, original_path, path)
            if residual_score < 0.18:
                continue
        else:
            residual_score = 0.0
        if inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio):
            best_record = {**record, "_residual_score": residual_score}
            best_inliers = inliers
            best_ratio = ratio

    if not best_record or best_inliers < 80 or best_ratio < 0.80:
        return None

    confidence = min(96, max(75, int(75 + best_record.get("_residual_score", 0) * 25)))
    return with_evidence_fields({
        "id": best_record.get("id"),
        "trace_id": best_record.get("trace_id"),
        "user_id": best_record.get("user_id"),
        "mode": "robust_dct",
        "mode_label": "30% 局部截图匹配",
        "created_at": best_record.get("created_at"),
        "confidence": confidence,
        "phash_match": True,
        "status": "局部截图命中",
        "extracted_at": now_text(),
        "match_inliers": best_inliers,
        "match_ratio": round(best_ratio, 3),
        "watermark_layers": best_record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(float(best_record.get("_residual_score", 0.0)), 4),
            "dwt": round(float(best_record.get("_residual_score", 0.0)), 4),
            "fft": round(float(best_record.get("_residual_score", 0.0)), 4),
        },
    }, best_record)


def is_registered_original_image(image: Image.Image) -> bool:
    query = np.array(image.convert("RGB"), dtype=np.int16)
    query_height, query_width = query.shape[:2]
    for record in read_records():
        original_url = record.get("original_url")
        if not original_url or not original_url.startswith("/uploads/"):
            continue
        original_path = UPLOAD_DIR / original_url.replace("/uploads/", "")
        if not original_path.exists():
            continue
        try:
            with Image.open(original_path) as original:
                if original.size != (query_width, query_height):
                    continue
                original_arr = np.array(original.convert("RGB"), dtype=np.int16)
        except Exception:
            continue
        diff = np.abs(query - original_arr)
        if float(diff.mean()) <= 0.05 and int(diff.max()) <= 1:
            return True
    return False


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "simhei.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_random_font(size: int, rng: np.random.Generator) -> ImageFont.ImageFont:
    font_names = [
        "arial.ttf",
        "arialbd.ttf",
        "simhei.ttf",
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simkai.ttf",
        "consola.ttf",
        "verdana.ttf",
        "tahoma.ttf",
        "times.ttf",
    ]
    font_paths = []
    windows_font_dir = Path(os.getenv("WINDIR", "C:\\Windows")) / "Fonts"
    for name in font_names:
        font_paths.append(windows_font_dir / name)
        font_paths.append(Path(name))
    rng.shuffle(font_paths)
    for path in font_paths:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return load_font(size)


def draw_text_pattern(layer: Image.Image, text: str, angle: int, gap: int, opacity: int) -> None:
    width, height = layer.size
    tile = Image.new("RGBA", (width * 2, height * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    font = load_font(max(18, min(width, height) // 18))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        text = text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), text, font=font)
    text_width = max(80, bbox[2] - bbox[0])
    text_height = max(24, bbox[3] - bbox[1])
    step_x = text_width + gap
    step_y = text_height + gap
    for y in range(-height, height * 2, step_y):
        for x in range(-width, width * 2, step_x):
            draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
    rotated = tile.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
    layer.alpha_composite(rotated.crop((width // 2, height // 2, width // 2 + width, height // 2 + height)))


def draw_irregular_text_pattern(layer: Image.Image, text: str, opacity: int, complexity: str) -> None:
    width, height = layer.size
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "big"))
    base_size = max(16, min(width, height) // 20)
    density = {
        "low": 0.55,
        "medium": 0.90,
        "high": 1.25,
        "extreme": 1.75,
        "低": 0.55,
        "中": 0.90,
        "高": 1.25,
        "极": 1.75,
    }.get(complexity, 0.90)
    count = max(10, int((width * height / 130_000) * density))
    colors = [
        (255, 255, 255),
        (255, 248, 196),
        (210, 245, 255),
        (235, 235, 255),
    ]
    safe_text = text
    for index in range(count):
        size = int(base_size * float(rng.uniform(0.70, 1.35)))
        font = load_random_font(size, rng)
        if rng.random() < 0.18:
            draw_text = safe_text.replace(" ", "")
            size = max(10, int(size * float(rng.uniform(0.45, 0.65))))
            font = load_random_font(size, rng)
        else:
            draw_text = safe_text
        try:
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        except UnicodeEncodeError:
            safe_text = safe_text.replace("©", "Copyright")
            draw_text = draw_text.replace("©", "Copyright")
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        patch = Image.new("RGBA", (text_width + 24, text_height + 24), (0, 0, 0, 0))
        patch_draw = ImageDraw.Draw(patch)
        color = colors[int(rng.integers(0, len(colors)))]
        alpha = max(8, min(220, int(opacity * float(rng.uniform(0.45, 1.25)))))
        patch_draw.text((12, 12), draw_text, fill=(*color, alpha), font=font)
        angle = float(rng.uniform(-38, 38))
        if rng.random() < 0.25:
            angle += float(rng.choice(np.array([-58, 58], dtype=np.int16)))
        rotated = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        x = int(rng.integers(-rotated.width // 3, max(1, width - rotated.width * 2 // 3)))
        y = int(rng.integers(-rotated.height // 3, max(1, height - rotated.height * 2 // 3)))
        layer.alpha_composite(rotated, (x, y))

    micro_count = max(18, int(count * 1.8))
    micro_font = load_random_font(max(9, base_size // 2), rng)
    micro_text = text.replace(" ", "")
    for _ in range(micro_count):
        x = int(rng.integers(0, max(1, width - 24)))
        y = int(rng.integers(0, max(1, height - 12)))
        alpha = max(5, int(opacity * float(rng.uniform(0.18, 0.45))))
        ImageDraw.Draw(layer).text((x, y), micro_text, fill=(255, 255, 255, alpha), font=micro_font)


def draw_prominent_corner_label(image: Image.Image, text: str) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)
    safe_text = text.strip() or "© QQ:757675150"
    font_size = max(22, min(base.size) // 14)
    font = load_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    except UnicodeEncodeError:
        safe_text = safe_text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    max_text_width = max(120, int(base.width * 0.72))
    while bbox[2] - bbox[0] > max_text_width and font_size > 16:
        font_size -= 2
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))

    stroke_width = max(2, font_size // 18)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x = max(12, font_size // 3)
    padding_y = max(8, font_size // 4)
    margin = max(14, min(base.size) // 40)
    right = base.width - margin
    bottom = base.height - margin
    left = max(margin, right - text_width - padding_x * 2)
    top = max(margin, bottom - text_height - padding_y * 2)
    radius = max(5, font_size // 6)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=(0, 0, 0, 205))
    draw.text(
        (left + padding_x, top + padding_y - bbox[1]),
        safe_text,
        font=font,
        fill=(255, 212, 0, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    return base.convert("RGB")


def apply_visible_copyright(
    image: Image.Image,
    enabled: bool,
    text: str,
    opacity: float,
    complexity: str,
    irregular: bool = True,
    prominent_corner: bool = False,
) -> Image.Image:
    if not enabled:
        return image.convert("RGB")

    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text = text.strip() or "© QQ:757675150"
    alpha = int(255 * opacity)
    settings = {
        "low": [(-24, 220)],
        "medium": [(-24, 110)],
        "high": [(-24, 105), (24, 105)],
        "extreme": [(-32, 75), (0, 75), (32, 75)],
        "低": [(-24, 220)],
        "中": [(-24, 110)],
        "高": [(-24, 105), (24, 105)],
        "极": [(-32, 75), (0, 75), (32, 75)],
    }.get(complexity, [(-24, 110)])
    if irregular:
        draw_irregular_text_pattern(layer, text, alpha, complexity)
    else:
        for angle, gap in settings:
            draw_text_pattern(layer, text, angle, gap, alpha)
    result = Image.alpha_composite(base, layer).convert("RGB")
    if prominent_corner:
        result = draw_prominent_corner_label(result, text)
    return result


async def load_upload_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    return load_image_from_bytes(content)


def load_image_from_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片") from exc
    return image


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def path_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def image_content_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest().upper()


def matched_file_fingerprint(content: bytes) -> dict[str, Any] | None:
    digest = file_sha256(content)
    query_image_digest = None
    for record in read_records():
        for file_type in ("original", "watermarked"):
            stored_file_digest = str(
                record.get(f"{file_type}_file_sha256") or ""
            ).upper()
            stored_image_digest = str(
                record.get(f"{file_type}_image_sha256") or ""
            ).upper()
            if stored_file_digest and stored_file_digest == digest:
                matched_hash_type = "file_bytes"
                matched_hash = digest
            elif stored_image_digest:
                try:
                    if query_image_digest is None:
                        query_image_digest = image_content_sha256(
                            load_image_from_bytes(content)
                        )
                except Exception:
                    return None
                if stored_image_digest != query_image_digest:
                    continue
                matched_hash_type = "image_pixels"
                matched_hash = query_image_digest
            else:
                continue
            return with_evidence_fields({
                "id": record.get("id"),
                "trace_id": record.get("trace_id"),
                "user_id": record.get("user_id"),
                "mode": "file_fingerprint",
                "mode_label": "文件指纹一样",
                "created_at": record.get("created_at"),
                "confidence": 100,
                "phash_match": False,
                "status": "文件指纹一样",
                "extracted_at": now_text(),
                "file_hash": digest,
                "image_hash": query_image_digest,
                "matched_hash": matched_hash,
                "matched_hash_type": matched_hash_type,
                "matched_file_type": file_type,
                "matched_file_url": record.get(
                    "original_url" if file_type == "original" else "download_url"
                ),
                "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
                "layer_scores": {},
            }, record)
    return None


def save_thumbnail(image: Image.Image, path: Path, scale: float = 0.20) -> None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    thumbnail = rgb.resize(thumb_size, Image.Resampling.LANCZOS)
    thumbnail.save(path, format="PNG", optimize=True)


def load_image_from_url(url: str) -> Image.Image:
    text = str(url or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入图片链接")
    if text.startswith("/uploads/"):
        path = UPLOAD_DIR / text.replace("/uploads/", "")
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="图片链接不存在")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
    if not text.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="仅支持 http(s) 或 /uploads/ 图片链接")
    try:
        request = urllib.request.Request(text, headers={"User-Agent": "WatermarkSystem/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read(20 * 1024 * 1024 + 1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取图片链接") from exc
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
    if content_type and "image" not in content_type.lower():
        raise HTTPException(status_code=400, detail="链接内容不是图片")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc


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
