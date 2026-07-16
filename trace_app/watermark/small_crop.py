import hashlib
from functools import lru_cache
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from trace_app.config import (
    CODE_CHANNEL_WEIGHTS, CODE_DELTA, CODE_PAYLOAD_BITS, CODE_PHYSICAL_BITS,
    CODE_TILE, CODE_WATERMARK_VERSION, ROBUST_BITS, ROBUST_MAGIC,
    SMALL_TRACE_CHANNEL_WEIGHTS, SMALL_TRACE_DELTA, SMALL_TRACE_SHORT_BITS,
    SMALL_TRACE_TILE, SMALL_TRACE_VERSION, WATERMARK_LAYERS,
)

from trace_app.watermark.frequency import layer_seed


def _code_crc16(value: int) -> int:
    return int.from_bytes(hashlib.blake2b(value.to_bytes(4, "big"), digest_size=2).digest(), "big")


def _watermark_payload_from_trace(trace_id: str) -> int:
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=2).digest()
    body = int.from_bytes(digest, "big")
    checksum = _code_crc16((ROBUST_MAGIC << 16) | body)
    return (ROBUST_MAGIC << 32) | (body << 16) | checksum


def _watermark_bits_from_trace(trace_id: str) -> list[int]:
    payload = _watermark_payload_from_trace(trace_id)
    base = [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]
    return [base[index % CODE_PAYLOAD_BITS] for index in range(CODE_PHYSICAL_BITS)]


def _recover_payload_from_code(code: int) -> tuple[int, int]:
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


def _clamp_float(value: str | float | None, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def small_trace_short_code(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> int:
    payload_fn = watermark_payload_from_trace_fn or _watermark_payload_from_trace
    return payload_fn(trace_id) & ((1 << SMALL_TRACE_SHORT_BITS) - 1)


def small_trace_short_bits(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> np.ndarray:
    code = small_trace_short_code(
        trace_id,
        watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
    )
    return np.array(
        [1.0 if ((code >> shift) & 1) else -1.0 for shift in range(SMALL_TRACE_SHORT_BITS - 1, -1, -1)],
        dtype=np.float32,
    )


def small_crop_strength_to_scale(value: str | float | None) -> float:
    return _clamp_float(value, 1.0, 0.0, 1.0)


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
    *,
    watermark_bits_from_trace_fn: Callable[[str], list[int]] | None = None,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
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
    bits_fn = watermark_bits_from_trace_fn or _watermark_bits_from_trace
    bits = np.array([1.0 if bit else -1.0 for bit in bits_fn(trace_id)], dtype=np.float32)
    code = normalize_carrier(np.tensordot(bits, small_trace_code_carriers(SMALL_TRACE_TILE), axes=([0], [0])))
    short_code = normalize_carrier(
        np.tensordot(
            small_trace_short_bits(
                trace_id,
                watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
            ),
            small_trace_short_carriers(SMALL_TRACE_TILE),
            axes=([0], [0]),
        )
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


def apply_code_layer_shifted(
    image: Image.Image,
    trace_id: str,
    *,
    apply_code_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
) -> Image.Image:
    base = image.convert("RGB")
    shifted = Image.new("RGB", base.size)
    shifted.paste(base.crop((CODE_TILE // 2, CODE_TILE // 2, base.width, base.height)), (0, 0))
    apply_fn = apply_code_layer_fn or apply_code_layer
    marked = apply_fn(shifted, trace_id)
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
    *,
    hamming_distance_fn: Callable[[int, int], int] | None = None,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> dict[str, Any] | None:
    distance_fn = hamming_distance_fn or _hamming_distance
    best_record = None
    best_distance = SMALL_TRACE_SHORT_BITS + 1
    second_distance = SMALL_TRACE_SHORT_BITS + 1
    for trace_id, _, record in code_records:
        distance = distance_fn(
            short_code,
            small_trace_short_code(
                trace_id,
                watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
            ),
        )
        if distance < best_distance:
            second_distance = best_distance
            best_record = record
            best_distance = distance
        elif distance < second_distance:
            second_distance = distance
    if best_record and best_distance <= max_errors and second_distance - best_distance >= min_gap:
        return best_record
    return None


def match_small_trace_code(
    code: int,
    code_records: list[tuple[str, int, dict[str, Any]]],
    max_errors: int = 10,
    *,
    recover_payload_from_code_fn: Callable[[int], tuple[int, int]] | None = None,
    hamming_distance_fn: Callable[[int, int], int] | None = None,
    code_crc16_fn: Callable[[int], int] | None = None,
) -> tuple[dict[str, Any] | None, int, int]:
    recover_fn = recover_payload_from_code_fn or _recover_payload_from_code
    distance_fn = hamming_distance_fn or _hamming_distance
    crc_fn = code_crc16_fn or _code_crc16
    payload, corrections = recover_fn(code)
    magic_distance = distance_fn(payload >> 32, ROBUST_MAGIC)
    if magic_distance > 3:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    body_and_magic = payload >> 16
    checksum = payload & 0xFFFF
    crc_distance = distance_fn(checksum, crc_fn(body_and_magic))
    if crc_distance > 12:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    best_record = None
    best_distance = CODE_PAYLOAD_BITS + 1
    second_distance = CODE_PAYLOAD_BITS + 1
    for _, expected_payload, record in code_records:
        distance = distance_fn(payload, expected_payload)
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


def detect_small_crop_trace(
    image: Image.Image,
    records: list[dict[str, Any]],
    generated_trace_ids: list[str],
    *,
    watermark_payload_from_trace: Callable[[str], int],
    record_visual_consistency: Callable[[Image.Image, dict[str, Any]], tuple[bool, int, float, float]],
    recover_payload_from_code: Callable[[int], tuple[int, int]],
    hamming_distance: Callable[[int, int], int],
    code_crc16: Callable[[int], int],
    now_text: Callable[[], str],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    raw_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
        and record.get("small_crop_trace_enabled")
        and record.get("small_crop_trace_version") == SMALL_TRACE_VERSION
    ]
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
            best_record, _, _ = match_small_trace_code(
                code,
                candidate_records,
                recover_payload_from_code_fn=recover_payload_from_code,
                hamming_distance_fn=hamming_distance,
                code_crc16_fn=code_crc16,
            )
            short_strength = 0.0
            matched_by_short_code = False
            if not best_record:
                short_code, short_strength = short_code_from_scores(decode_small_trace_short_scores(normalized))
                best_record = record_from_short_code_match(
                    short_code,
                    candidate_records,
                    4 if not persistent_candidate_mode else 2,
                    2 if not persistent_candidate_mode else 3,
                    hamming_distance_fn=hamming_distance,
                    watermark_payload_from_trace_fn=watermark_payload_from_trace,
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


def detect_watermark_code(
    image: Image.Image,
    records: list[dict[str, Any]],
    generated_trace_ids: list[str],
    *,
    watermark_payload_from_trace: Callable[[str], int],
    record_visual_consistency: Callable[[Image.Image, dict[str, Any]], tuple[bool, int, float, float]],
    recover_payload_from_code: Callable[[int], tuple[int, int]],
    hamming_distance: Callable[[int, int], int],
    code_crc16: Callable[[int], int],
    now_text: Callable[[], str],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    scales = (1.0, 0.95, 1.05)
    best_trace = None
    best_strength = 0.0
    best_votes = 0
    raw_code_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
    ]
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
