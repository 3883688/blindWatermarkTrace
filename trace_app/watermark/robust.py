import hashlib
import re
import time
from collections.abc import Callable, Iterable
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


def robust_code_to_trace(code: int, *, records: Iterable[Record]) -> str | None:
    if (code >> 48) != ROBUST_MAGIC:
        return None
    for record in records:
        trace_id = record.get("trace_id")
        if trace_id and robust_code_from_trace(trace_id) == code:
            return trace_id
    return None


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def robust_code_to_trace_fuzzy(
    code: int,
    max_errors: int = 18,
    *,
    records: Iterable[Record],
) -> tuple[str | None, int]:
    magic_distance = hamming_distance(code >> 48, ROBUST_MAGIC)
    if magic_distance > 6:
        return None, ROBUST_BITS + 1
    best_trace = None
    best_distance = ROBUST_BITS + 1
    for record in records:
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
            patch[:, :] = np.clip(patch + delta, 0, 255)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v2(image: Image.Image, trace_id: str, strength_scale: float = 1.0) -> Image.Image:
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
                patch + robust_pattern(bit_index, ROBUST_CELL) * ROBUST_DELTA * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v3(image: Image.Image, auth_code: bytes, strength_scale: float = 1.0) -> Image.Image:
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
                patch + robust_pattern(bit_index, ROBUST_CELL) * ROBUST_DELTA * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def extract_robust_from_grid(
    arr: np.ndarray, cell: int, offset_x: int, offset_y: int
) -> tuple[int | None, float, int]:
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
    alignment: Record, record: Record, max_errors: int = 4
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
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (ROBUST_TILE, ROBUST_TILE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        authenticated_tiles += 1
        for bit_index in range(ROBUST_BITS):
            row = bit_index // ROBUST_GRID
            col = bit_index % ROBUST_GRID
            cell = tile[row * ROBUST_CELL : (row + 1) * ROBUST_CELL, col * ROBUST_CELL : (col + 1) * ROBUST_CELL, ROBUST_CHANNEL]
            centered = cell - cell.mean()
            aggregate_scores[bit_index] += float(np.mean(centered * robust_pattern(bit_index, ROBUST_CELL)))
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


def _phase_scores_to_codeword(phase_scores: np.ndarray, phase_counts: list[int]) -> tuple[bytes, list[float]]:
    observed = bytearray()
    confidences = []
    for phase in range(3):
        average = phase_scores[phase] / max(1, phase_counts[phase])
        for start in range(0, ROBUST_BITS, 8):
            value, confidence = _scores_to_byte(average[start : start + 8])
            observed.append(value)
            confidences.append(confidence)
    return bytes(observed), confidences


def decode_aligned_robust_trace_v2(alignment: Record, record: Record) -> Record | None:
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
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (ROBUST_TILE, ROBUST_TILE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        phase_counts[phase] += 1
        for bit_index in range(ROBUST_BITS):
            row, col = divmod(bit_index, ROBUST_GRID)
            cell = tile[row * ROBUST_CELL : (row + 1) * ROBUST_CELL, col * ROBUST_CELL : (col + 1) * ROBUST_CELL, ROBUST_CHANNEL]
            centered = cell - cell.mean()
            phase_scores[phase, bit_index] += float(np.mean(centered * robust_pattern(bit_index, ROBUST_CELL)))
    if min(phase_counts) < 2:
        return None
    observed, confidences = _phase_scores_to_codeword(phase_scores, phase_counts)
    decoded = decode_expected_codeword(observed, robust_payload_bytes(trace_id), confidences)
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


def decode_aligned_robust_trace_v3(alignment: Record, record: Record, max_errors: int = 8) -> Record | None:
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
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (ROBUST_TILE, ROBUST_TILE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        physical_scores = np.zeros(ROBUST_BITS, dtype=np.float64)
        for bit_index in range(ROBUST_BITS):
            row, col = divmod(bit_index, ROBUST_GRID)
            cell = tile[row * ROBUST_CELL : (row + 1) * ROBUST_CELL, col * ROBUST_CELL : (col + 1) * ROBUST_CELL, ROBUST_CHANNEL]
            centered = cell - cell.mean()
            physical_scores[bit_index] = float(np.mean(centered * robust_pattern(bit_index, ROBUST_CELL)))
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
    expected_bits = np.array([(expected_value >> shift) & 1 for shift in range(ROBUST_BITS - 1, -1, -1)], dtype=np.int8)
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
) -> Record | None:
    del generated_trace_ids
    started = time.perf_counter()
    candidates = [record for record in records if record.get("trace_id") and record.get("robust_watermark")]
    candidates = rank_candidates(image, candidates)[: max(1, candidate_limit)]
    authenticated = []
    for record in candidates:
        if budget_seconds > 0 and time.perf_counter() - started >= budget_seconds:
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
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
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


def extract_robust_code(image: Image.Image, *, records: Iterable[Record]) -> tuple[str | None, float, int]:
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
