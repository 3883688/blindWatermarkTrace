import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from watermark_auth import permuted_code_bits, phase_permutation
from watermark_ecc import codeword_phase, decode_expected_codeword, encode_codeword, tile_phase
from trace_app.config import (
    CODE_PAYLOAD_BITS,
    CODE_PHYSICAL_BITS,
    ROBUST_BITS,
    ROBUST_CELL,
    ROBUST_CHANNEL,
    ROBUST_DELTA,
    ROBUST_GRID,
    ROBUST_MAGIC,
    ROBUST_TILE,
)
from trace_app.watermark.frequency import robust_pattern


Record = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RobustConfig:
    robust_bits: int = ROBUST_BITS
    robust_cell: int = ROBUST_CELL
    robust_channel: int = ROBUST_CHANNEL
    robust_delta: float = ROBUST_DELTA
    robust_grid: int = ROBUST_GRID
    robust_magic: int = ROBUST_MAGIC
    robust_tile: int = ROBUST_TILE
    code_payload_bits: int = CODE_PAYLOAD_BITS
    code_physical_bits: int = CODE_PHYSICAL_BITS


@dataclass(frozen=True, slots=True)
class RobustDependencies:
    robust_code_from_trace: Callable[..., int] | None = None
    robust_bits_from_code: Callable[..., list[int]] | None = None
    robust_payload_bytes: Callable[..., bytes] | None = None
    code_crc16: Callable[[int], int] | None = None
    watermark_payload_from_trace: Callable[..., int] | None = None
    hamming_distance: Callable[[int, int], int] | None = None
    iter_robust_tiles: Callable[..., Iterable[tuple[int, int]]] | None = None
    robust_pattern: Callable[[int, int], np.ndarray] | None = None
    encode_codeword: Callable[[bytes], bytes] | None = None
    codeword_phase: Callable[[bytes, int], bytes] | None = None
    tile_phase: Callable[[int, int], int] | None = None
    permuted_code_bits: Callable[[bytes, int], list[int]] | None = None
    phase_permutation: Callable[[int], Any] | None = None
    extract_robust_from_grid: Callable[..., tuple[int | None, float, int]] | None = None
    scores_to_byte: Callable[[np.ndarray], tuple[int, float]] | None = None
    phase_scores_to_codeword: Callable[..., tuple[bytes, list[float]]] | None = None
    decode_expected_codeword: Callable[..., Any] | None = None
    record_v3_auth_code: Callable[[Record], bytes | None] | None = None


DEFAULT_CONFIG = RobustConfig()
DEFAULT_DEPENDENCIES = RobustDependencies()


def robust_code_from_trace(
    trace_id: str, *, config: RobustConfig = DEFAULT_CONFIG
) -> int:
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=6).digest()
    body = int.from_bytes(digest, "big")
    return (config.robust_magic << 48) | body


def robust_bits_from_code(
    code: int, *, config: RobustConfig = DEFAULT_CONFIG
) -> list[int]:
    return [(code >> shift) & 1 for shift in range(config.robust_bits - 1, -1, -1)]


def robust_payload_bytes(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> bytes:
    code = (
        dependencies.robust_code_from_trace(trace_id)
        if dependencies.robust_code_from_trace
        else robust_code_from_trace(trace_id, config=config)
    )
    return code.to_bytes(8, "big")


def code_crc16(value: int) -> int:
    return int.from_bytes(hashlib.blake2b(value.to_bytes(4, "big"), digest_size=2).digest(), "big")


def watermark_payload_from_trace(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> int:
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=2).digest()
    body = int.from_bytes(digest, "big")
    crc16 = dependencies.code_crc16 or code_crc16
    checksum = crc16((config.robust_magic << 16) | body)
    return (config.robust_magic << 32) | (body << 16) | checksum


def watermark_bits_from_trace(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> list[int]:
    payload = (
        dependencies.watermark_payload_from_trace(trace_id)
        if dependencies.watermark_payload_from_trace
        else watermark_payload_from_trace(
            trace_id, config=config, dependencies=dependencies
        )
    )
    base = [(payload >> shift) & 1 for shift in range(config.code_payload_bits - 1, -1, -1)]
    repeated = []
    for index in range(config.code_physical_bits):
        repeated.append(base[index % config.code_payload_bits])
    return repeated


def recover_payload_from_code(
    code: int, *, config: RobustConfig = DEFAULT_CONFIG
) -> tuple[int, int]:
    bits = [(code >> shift) & 1 for shift in range(config.code_physical_bits - 1, -1, -1)]
    recovered = 0
    corrections = 0
    for index in range(config.code_payload_bits):
        votes = [bits[pos] for pos in range(index, config.code_physical_bits, config.code_payload_bits)]
        one_votes = sum(votes)
        zero_votes = len(votes) - one_votes
        bit = 1 if one_votes >= zero_votes else 0
        corrections += min(one_votes, zero_votes)
        recovered = (recovered << 1) | bit
    return recovered, corrections


def normalize_robust_watermark_version(
    value: str | int | None,
    *,
    version_v1: int,
    version_v2: int,
    version_v3: int,
    version_v4: int,
) -> int:
    try:
        version = int(value if value is not None else version_v1)
    except (TypeError, ValueError):
        version = version_v1
    if version == version_v4:
        return version_v4
    if version == version_v3:
        return version_v3
    if version == version_v2:
        return version_v2
    return version_v1


def robust_code_to_trace(
    code: int,
    *,
    records: Iterable[Record],
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> str | None:
    if (code >> 48) != config.robust_magic:
        return None
    code_from_trace = dependencies.robust_code_from_trace
    for record in records:
        trace_id = record.get("trace_id")
        expected = (
            code_from_trace(trace_id)
            if code_from_trace
            else robust_code_from_trace(trace_id, config=config)
        ) if trace_id else None
        if trace_id and expected == code:
            return trace_id
    return None


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def robust_code_to_trace_fuzzy(
    code: int,
    max_errors: int = 18,
    *,
    records: Iterable[Record],
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[str | None, int]:
    distance_fn = dependencies.hamming_distance or hamming_distance
    magic_distance = distance_fn(code >> 48, config.robust_magic)
    if magic_distance > 6:
        return None, config.robust_bits + 1
    best_trace = None
    best_distance = config.robust_bits + 1
    for record in records:
        trace_id = record.get("trace_id")
        if not trace_id:
            continue
        expected = (
            dependencies.robust_code_from_trace(trace_id)
            if dependencies.robust_code_from_trace
            else robust_code_from_trace(trace_id, config=config)
        )
        distance = distance_fn(code, expected)
        if distance < best_distance:
            best_trace = trace_id
            best_distance = distance
    if best_trace and best_distance <= max_errors:
        return best_trace, best_distance
    return None, best_distance


def robust_candidate_records(records: Iterable[Record]) -> list[Record]:
    return [record for record in records if record.get("trace_id") and record.get("robust_watermark")]


def legacy_robust_candidate_records(
    records: Iterable[Record],
    *,
    normalize_version: Callable[[str | int | None], int],
    version_v1: int,
) -> list[Record]:
    return [
        record
        for record in robust_candidate_records(records)
        if normalize_version(record.get("robust_watermark_version", version_v1)) == version_v1
    ]


def iter_robust_tiles(
    width: int, height: int, *, config: RobustConfig = DEFAULT_CONFIG
):
    for y in range(0, height - config.robust_tile + 1, config.robust_tile):
        for x in range(0, width - config.robust_tile + 1, config.robust_tile):
            yield x, y


def embed_robust_watermark(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    code = (
        dependencies.robust_code_from_trace(trace_id)
        if dependencies.robust_code_from_trace
        else robust_code_from_trace(trace_id, config=config)
    )
    bits = (
        dependencies.robust_bits_from_code(code)
        if dependencies.robust_bits_from_code
        else robust_bits_from_code(code, config=config)
    )
    tiles = (
        dependencies.iter_robust_tiles(width, height)
        if dependencies.iter_robust_tiles
        else iter_robust_tiles(width, height, config=config)
    )
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        for bit_index, bit in enumerate(bits):
            row = bit_index // config.robust_grid
            col = bit_index % config.robust_grid
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            pattern = pattern_fn(bit_index, config.robust_cell)
            delta = pattern * ((config.robust_delta * strength_scale) if bit else -(config.robust_delta * strength_scale))
            patch[:, :] = np.clip(patch + delta, 0, 255)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v2(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    payload = (
        dependencies.robust_payload_bytes(trace_id)
        if dependencies.robust_payload_bytes
        else robust_payload_bytes(trace_id, config=config, dependencies=dependencies)
    )
    codeword = (dependencies.encode_codeword or encode_codeword)(payload)
    tiles = dependencies.iter_robust_tiles(image.width, image.height) if dependencies.iter_robust_tiles else iter_robust_tiles(image.width, image.height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    phase_fn = dependencies.codeword_phase or codeword_phase
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        phase_bytes = phase_fn(codeword, phase)
        value = int.from_bytes(phase_bytes, "big")
        bits = dependencies.robust_bits_from_code(value) if dependencies.robust_bits_from_code else robust_bits_from_code(value, config=config)
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, config.robust_grid)
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch + pattern_fn(bit_index, config.robust_cell) * config.robust_delta * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v3(
    image: Image.Image,
    auth_code: bytes,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    tiles = dependencies.iter_robust_tiles(image.width, image.height) if dependencies.iter_robust_tiles else iter_robust_tiles(image.width, image.height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    bits_fn = dependencies.permuted_code_bits or permuted_code_bits
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        bits = bits_fn(auth_code, phase)
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, config.robust_grid)
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch + pattern_fn(bit_index, config.robust_cell) * config.robust_delta * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def extract_robust_from_grid(
    arr: np.ndarray,
    cell: int,
    offset_x: int,
    offset_y: int,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[int | None, float, int]:
    height, width = arr.shape[:2]
    tile = cell * config.robust_grid
    votes = [[0, 0] for _ in range(config.robust_bits)]
    pattern_fn = dependencies.robust_pattern or robust_pattern
    tiles = 0
    for y in range(offset_y, height - tile + 1, tile):
        for x in range(offset_x, width - tile + 1, tile):
            tiles += 1
            for bit_index in range(config.robust_bits):
                row = bit_index // config.robust_grid
                col = bit_index % config.robust_grid
                y0 = y + row * cell
                x0 = x + col * cell
                patch = arr[y0 : y0 + cell, x0 : x0 + cell, :]
                if patch.size == 0:
                    continue
                blue = patch[:, :, config.robust_channel]
                if blue.shape != (cell, cell):
                    continue
                pattern = pattern_fn(bit_index, cell).astype(np.float32)
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
    alignment: Record,
    record: Record,
    max_errors: int = 4,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
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
    aggregate_scores = np.zeros(config.robust_bits, dtype=np.float64)
    authenticated_tiles = 0
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        authenticated_tiles += 1
        for bit_index in range(config.robust_bits):
            row = bit_index // config.robust_grid
            col = bit_index % config.robust_grid
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            aggregate_scores[bit_index] += float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
    if authenticated_tiles < 2:
        return None
    decoded_code = 0
    for score in aggregate_scores:
        decoded_code = (decoded_code << 1) | int(score > 0)
    expected_code = dependencies.robust_code_from_trace(trace_id) if dependencies.robust_code_from_trace else robust_code_from_trace(trace_id, config=config)
    distance_fn = dependencies.hamming_distance or hamming_distance
    bit_errors = distance_fn(decoded_code, expected_code)
    magic_errors = distance_fn(decoded_code >> 48, config.robust_magic)
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
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[bytes, list[float]]:
    observed = bytearray()
    confidences = []
    for phase in range(3):
        average = phase_scores[phase] / max(1, phase_counts[phase])
        for start in range(0, config.robust_bits, 8):
            value, confidence = (dependencies.scores_to_byte or _scores_to_byte)(average[start : start + 8])
            observed.append(value)
            confidences.append(confidence)
    return bytes(observed), confidences


def decode_aligned_robust_trace_v2(
    alignment: Record,
    record: Record,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
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
    phase_scores = np.zeros((3, config.robust_bits), dtype=np.float64)
    phase_counts = [0, 0, 0]
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        phase_counts[phase] += 1
        for bit_index in range(config.robust_bits):
            row, col = divmod(bit_index, config.robust_grid)
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            phase_scores[phase, bit_index] += float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
    if min(phase_counts) < 2:
        return None
    observed, confidences = (
        dependencies.phase_scores_to_codeword(phase_scores, phase_counts)
        if dependencies.phase_scores_to_codeword
        else _phase_scores_to_codeword(phase_scores, phase_counts, config=config, dependencies=dependencies)
    )
    payload = dependencies.robust_payload_bytes(trace_id) if dependencies.robust_payload_bytes else robust_payload_bytes(trace_id, config=config, dependencies=dependencies)
    decoded = (dependencies.decode_expected_codeword or decode_expected_codeword)(observed, payload, confidences)
    if not decoded:
        return None
    average_scores = np.vstack([phase_scores[index] / phase_counts[index] for index in range(3)])
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


def _record_v3_auth_code(record: Record) -> bytes | None:
    text = str(record.get("robust_auth_code") or "").strip().lower()
    if len(text) != 16 or not re.fullmatch(r"[0-9a-f]{16}", text):
        return None
    try:
        code = bytes.fromhex(text)
    except ValueError:
        return None
    return code if len(code) == 8 else None


def decode_aligned_robust_trace_v3(
    alignment: Record,
    record: Record,
    max_errors: int = 8,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
    trace_id = record.get("trace_id")
    auth_code = (dependencies.record_v3_auth_code or _record_v3_auth_code)(record)
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
    aggregate_scores = np.zeros(config.robust_bits, dtype=np.float64)
    phase_counts = [0, 0, 0]
    authenticated_tiles = 0
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    phase_permutation_fn = dependencies.phase_permutation or phase_permutation
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        physical_scores = np.zeros(config.robust_bits, dtype=np.float64)
        for bit_index in range(config.robust_bits):
            row, col = divmod(bit_index, config.robust_grid)
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            physical_scores[bit_index] = float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
        scale = max(1e-6, float(np.median(np.abs(physical_scores))))
        physical_scores = np.clip(physical_scores / scale, -3.0, 3.0)
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        permutation = phase_permutation_fn(phase)
        for logical, physical in enumerate(permutation):
            aggregate_scores[logical] += physical_scores[physical]
        phase_counts[phase] += 1
        authenticated_tiles += 1
    if authenticated_tiles < 2 or sum(count > 0 for count in phase_counts) < 2:
        return None
    expected_value = int.from_bytes(auth_code, "big")
    expected_bits = np.array([(expected_value >> shift) & 1 for shift in range(config.robust_bits - 1, -1, -1)], dtype=np.int8)
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


def detect_aligned_authenticated_watermark(
    image: Image.Image,
    candidate_limit: int = 8,
    budget_seconds: float = 5.0,
    *,
    records: Iterable[Record],
    generated_trace_ids: Iterable[str] | None = None,
    rank_candidates: Callable[[Image.Image, list[Record]], list[Record]],
    align_query: Callable[[Image.Image, Record], Record | None],
    decode_v1: Callable[[Record, Record], Record | None],
    decode_v2: Callable[[Record, Record], Record | None],
    decode_v3: Callable[[Record, Record], Record | None],
    normalize_version: Callable[[str | int | None], int],
    with_evidence_fields: Callable[[Record, Record | None], Record],
    now_text: Callable[[], str],
    version_v1: int,
    version_v2: int,
    version_v3: int,
    codec_v2: str,
    codec_v3: str,
    watermark_layers: Any,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> Record | None:
    del generated_trace_ids
    started = perf_counter()
    candidates = [record for record in records if record.get("trace_id") and record.get("robust_watermark")]
    candidates = rank_candidates(image, candidates)[: max(1, candidate_limit)]
    authenticated = []
    for record in candidates:
        if budget_seconds > 0 and perf_counter() - started >= budget_seconds:
            break
        alignment = align_query(image, record)
        if not alignment:
            continue
        version = normalize_version(record.get("robust_watermark_version", version_v1))
        if version == version_v3:
            decoded = decode_v3(alignment, record)
        elif version == version_v2:
            decoded = decode_v2(alignment, record)
        else:
            decoded = decode_v1(alignment, record)
        if decoded:
            authenticated.append((decoded, alignment))
    trace_ids = {decoded["trace_id"] for decoded, _ in authenticated}
    if len(authenticated) != 1 or len(trace_ids) != 1:
        return None
    decoded, alignment = authenticated[0]
    record = decoded["record"]
    version = normalize_version(record.get("robust_watermark_version", version_v1))
    confidence = max(80, min(99, 99 - decoded["bit_errors"] * 6))
    code_recovery = {
        "visual_inliers": alignment["inliers"],
        "visual_ratio": alignment["ratio"],
        "aligned_coverage": alignment["coverage"],
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if version == version_v3:
        code_recovery.update({
            "method": "homography_aligned_hmac64_full_repeat_v3",
            "codec": codec_v3,
            "bit_errors": decoded["bit_errors"],
            "authenticated_tiles": decoded["authenticated_tiles"],
            "phase_tile_counts": decoded["phase_tile_counts"],
            "mean_signed_agreement": decoded["mean_signed_agreement"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    elif version == version_v2:
        code_recovery.update({
            "method": "homography_aligned_rs_24_8_three_phase",
            "codec": codec_v2,
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
        "mode": "aligned_robust_hmac_v3" if version == version_v3 else "aligned_robust_rs_v2" if version == version_v2 else "aligned_robust_code",
        "mode_label": "几何对齐 HMAC 认证水印" if version == version_v3 else "几何对齐 RS 认证水印" if version == version_v2 else "几何对齐 64-bit 认证水印",
        "created_at": record.get("created_at"),
        "confidence": confidence,
        "phash_match": False,
        "status": "认证水印恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", watermark_layers),
        "code_recovery": code_recovery,
    }, record)


def extract_robust_code(
    image: Image.Image,
    *,
    records: Iterable[Record],
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[str | None, float, int]:
    trace_codes = {
        record.get("trace_id"): (
            dependencies.robust_code_from_trace(record.get("trace_id"))
            if dependencies.robust_code_from_trace
            else robust_code_from_trace(record.get("trace_id"), config=config)
        )
        for record in records
        if record.get("trace_id")
    }
    if not trace_codes:
        return None, 0.0, 0
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if width < config.robust_tile or height < config.robust_tile:
        return None, 0.0, 0
    candidates = []
    for cell in (8, 7, 9):
        tile = cell * config.robust_grid
        step = max(1, cell * 2)
        for offset_y in range(0, min(tile, height - tile + 1), step):
            for offset_x in range(0, min(tile, width - tile + 1), step):
                code, confidence, decided = (
                    dependencies.extract_robust_from_grid(arr, cell, offset_x, offset_y)
                    if dependencies.extract_robust_from_grid
                    else extract_robust_from_grid(
                        arr,
                        cell,
                        offset_x,
                        offset_y,
                        config=config,
                        dependencies=dependencies,
                    )
                )
                if code is None or decided < config.robust_bits:
                    continue
                trace_id = next((item for item, item_code in trace_codes.items() if item_code == code), None)
                if trace_id is not None:
                    candidates.append((trace_id, confidence, decided))
    if not candidates:
        return None, 0.0, 0
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def detect_robust_watermark(
    image: Image.Image,
    *,
    records: Iterable[Record],
    extract_code: Callable[[Image.Image, list[Record]], tuple[str | None, float, int]],
    with_evidence_fields: Callable[[Record, Record | None], Record],
    now_text: Callable[[], str],
    layer_scores_for_image: Callable[[Image.Image, str], Any],
    watermark_layers: Any,
) -> Record | None:
    records = list(records)
    if not records:
        return None
    trace_id, confidence, decided = extract_code(image, records)
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
        "watermark_layers": record.get("watermark_layers", watermark_layers),
        "layer_scores": layer_scores_for_image(image, trace_id),
    }, record)
