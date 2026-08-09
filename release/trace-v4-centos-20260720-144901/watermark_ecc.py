from typing import Any

from reedsolo import RSCodec, ReedSolomonError


RS_DATA_BYTES = 8
RS_PARITY_BYTES = 16
RS_CODEWORD_BYTES = 24
RS_ERASURE_COUNTS = (0, 2, 4, 6)
RS_CANDIDATE_MAX_BIT_ERRORS = 32
RS_PHASES = 3
RS_PHASE_BYTES = 8

_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_CODEWORD_BYTES)


def encode_codeword(payload: bytes) -> bytes:
    if len(payload) != RS_DATA_BYTES:
        raise ValueError("RS watermark payload must be exactly 8 bytes")
    encoded = bytes(_CODEC.encode(payload))
    if len(encoded) != RS_CODEWORD_BYTES:
        raise RuntimeError("unexpected RS watermark codeword length")
    return encoded


def decode_expected_codeword(
    observed: bytes,
    expected_payload: bytes,
    byte_confidences: list[float],
) -> dict[str, Any] | None:
    if len(observed) != RS_CODEWORD_BYTES or len(expected_payload) != RS_DATA_BYTES:
        return None
    if len(byte_confidences) != RS_CODEWORD_BYTES:
        return None

    expected_codeword = encode_codeword(expected_payload)
    confidence_order = sorted(
        range(RS_CODEWORD_BYTES),
        key=lambda index: byte_confidences[index],
    )
    for erasure_count in RS_ERASURE_COUNTS:
        erasures = confidence_order[:erasure_count]
        try:
            decoded, corrected, errata = _CODEC.decode(observed, erase_pos=erasures)
        except (ReedSolomonError, ValueError, IndexError):
            continue
        if bytes(decoded) != expected_payload or bytes(corrected) != expected_codeword:
            continue
        return {
            "payload": bytes(decoded),
            "corrected_codeword": bytes(corrected),
            "corrected_symbols": len(set(int(index) for index in errata)),
            "erasure_count": erasure_count,
            "bit_errors": sum(
                (left ^ right).bit_count()
                for left, right in zip(observed, expected_codeword)
            ),
            "recovery_method": "reed_solomon",
        }
    bit_errors = sum(
        (left ^ right).bit_count()
        for left, right in zip(observed, expected_codeword)
    )
    if bit_errors <= RS_CANDIDATE_MAX_BIT_ERRORS:
        return {
            "payload": expected_payload,
            "corrected_codeword": expected_codeword,
            "corrected_symbols": sum(
                left != right for left, right in zip(observed, expected_codeword)
            ),
            "erasure_count": 0,
            "bit_errors": bit_errors,
            "recovery_method": "expected_codeword_distance",
        }
    return None


def tile_phase(tile_x: int, tile_y: int) -> int:
    return (int(tile_x) + 2 * int(tile_y)) % RS_PHASES


def codeword_phase(codeword: bytes, phase: int) -> bytes:
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("RS watermark codeword must be exactly 24 bytes")
    if phase not in range(RS_PHASES):
        raise ValueError("RS watermark phase must be 0, 1, or 2")
    start = phase * RS_PHASE_BYTES
    return codeword[start : start + RS_PHASE_BYTES]
