from typing import Any, Callable

import numpy as np
from PIL import Image

from trace_app.config import (
    CODE_PAYLOAD_BITS, DOT_MATRIX_CHANNEL_WEIGHTS, DOT_MATRIX_DELTA,
    DOT_MATRIX_GRID, DOT_MATRIX_TILE, DOT_MATRIX_VERSION, ROBUST_MAGIC,
    WATERMARK_LAYERS,
)


def dot_matrix_bits_from_trace(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> list[int]:
    payload = watermark_payload_from_trace_fn(trace_id)
    return [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]


def dot_matrix_candidate_records(
    records: list[dict[str, Any]],
    *,
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (record.get("trace_id"), watermark_payload_from_trace_fn(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("dot_matrix_trace_enabled")
        and record.get("dot_matrix_trace_version") == DOT_MATRIX_VERSION
    ]


def dot_matrix_position(bit_index: int, tile_size: int = DOT_MATRIX_TILE) -> tuple[int, int]:
    cell = max(2, tile_size // DOT_MATRIX_GRID)
    row = bit_index // DOT_MATRIX_GRID
    col = bit_index % DOT_MATRIX_GRID
    return col * cell + cell // 2, row * cell + cell // 2


def apply_dot_matrix_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 1.0,
    *,
    clamp_float_fn: Callable[[str | float | None, float, float, float], float],
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> Image.Image:
    strength = clamp_float_fn(strength, 1.0, 0.0, 1.0)
    if strength <= 0:
        return image.convert("RGB")
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if height < DOT_MATRIX_TILE or width < DOT_MATRIX_TILE:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    bits = dot_matrix_bits_from_trace(
        trace_id,
        watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
    )
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


def detect_dot_matrix_trace(
    image: Image.Image,
    candidate_records: list[tuple[str, int, dict[str, Any]]],
    *,
    hamming_distance_fn: Callable[[int, int], int],
    code_crc16_fn: Callable[[int], int],
    now_text_fn: Callable[[], str],
    with_evidence_fields_fn: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    records = candidate_records
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
                    magic_distance = hamming_distance_fn(payload >> 32, ROBUST_MAGIC)
                    if magic_distance > 4:
                        continue
                    checksum = payload & 0xFFFF
                    crc_distance = hamming_distance_fn(checksum, code_crc16_fn(payload >> 16))
                    if crc_distance > 10:
                        continue
                    best_record = None
                    best_record_distance = CODE_PAYLOAD_BITS + 1
                    for _, expected_payload, record in records:
                        distance = hamming_distance_fn(payload, expected_payload)
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
    return with_evidence_fields_fn({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "dot_matrix_trace",
        "mode_label": "点阵追溯水印",
        "created_at": record.get("created_at"),
        "confidence": int(min(96, max(76, 72 + best_votes * 2))),
        "phash_match": False,
        "status": "点阵水印恢复",
        "extracted_at": now_text_fn(),
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
