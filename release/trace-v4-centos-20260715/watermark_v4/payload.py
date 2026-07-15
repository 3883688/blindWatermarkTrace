import hashlib
import hmac
import random
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

from reedsolo import RSCodec, ReedSolomonError


AUTH_TAG_BYTES = 4
AUTH_KEY_MIN_BYTES = 32
AUTH_MESSAGE_PREFIX = b"robust-v4:"
RS_DATA_BYTES = 4
RS_PARITY_BYTES = 4
RS_CODEWORD_BYTES = 8
RS_ERASURE_COUNTS = (0, 1, 2, 3, 4)
PHASE_COUNT = 4
CODEWORD_BITS = 64
PHASE_PERMUTATION_PREFIX = (
    b"hmac32_rs_8_4_full_repeat_sync_v4:carrier-permutation:"
)

_RS_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_CODEWORD_BYTES)


@dataclass(frozen=True, slots=True)
class CandidateDecode:
    payload: bytes
    corrected_codeword: bytes
    corrected_symbols: int
    erasure_count: int
    bit_errors: int


def authentication_tag(trace_id: str, key: str | bytes) -> bytes:
    if type(trace_id) is not str:
        raise TypeError("trace_id must be a string")
    if not trace_id or trace_id != trace_id.strip():
        raise ValueError("trace_id must be nonempty and have no surrounding whitespace")

    if type(key) is str:
        encoded_key = key.encode("utf-8")
    elif type(key) is bytes:
        encoded_key = key
    else:
        raise TypeError("authentication key must be a string or bytes")
    if len(encoded_key) < AUTH_KEY_MIN_BYTES:
        raise ValueError("authentication key must contain at least 32 bytes")

    message = AUTH_MESSAGE_PREFIX + trace_id.encode("utf-8")
    return hmac.new(encoded_key, message, hashlib.sha256).digest()[:AUTH_TAG_BYTES]


def encode_codeword(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("v4 payload must be bytes")
    if len(payload) != RS_DATA_BYTES:
        raise ValueError("v4 payload must be exactly 4 bytes")
    encoded = bytes(_RS_CODEC.encode(payload))
    if len(encoded) != RS_CODEWORD_BYTES:
        raise RuntimeError("unexpected v4 RS codeword length")
    return encoded


def decode_candidate_codeword(
    observed: bytes,
    expected_payload: bytes,
    byte_confidences: list[float] | tuple[float, ...],
) -> CandidateDecode | None:
    if type(observed) is not bytes or len(observed) != RS_CODEWORD_BYTES:
        return None
    if type(expected_payload) is not bytes or len(expected_payload) != RS_DATA_BYTES:
        return None
    if type(byte_confidences) not in (list, tuple) or len(byte_confidences) != RS_CODEWORD_BYTES:
        return None
    if any(not _is_finite_confidence(value) for value in byte_confidences):
        return None

    expected_codeword = encode_codeword(expected_payload)
    confidence_order = sorted(
        range(RS_CODEWORD_BYTES),
        key=lambda index: (byte_confidences[index], index),
    )
    for erasure_count in RS_ERASURE_COUNTS:
        erasures = confidence_order[:erasure_count]
        try:
            decoded, corrected, errata = _RS_CODEC.decode(
                observed,
                erase_pos=erasures,
            )
        except (ReedSolomonError, ValueError, IndexError, TypeError):
            continue
        if not hmac.compare_digest(bytes(decoded), expected_payload) or not hmac.compare_digest(
            bytes(corrected), expected_codeword
        ):
            continue
        return CandidateDecode(
            payload=bytes(decoded),
            corrected_codeword=bytes(corrected),
            corrected_symbols=len({int(index) for index in errata}),
            erasure_count=erasure_count,
            bit_errors=sum(
                (left ^ right).bit_count()
                for left, right in zip(observed, expected_codeword)
            ),
        )
    return None


def _is_finite_confidence(value: object) -> bool:
    if type(value) is int:
        return 0 <= value <= 1
    if type(value) is float:
        return isfinite(value) and 0.0 <= value <= 1.0
    return False


def bytes_to_bits(codeword: bytes) -> tuple[int, ...]:
    if type(codeword) is not bytes:
        raise TypeError("v4 codeword must be bytes")
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("v4 codeword must be exactly 8 bytes")
    return tuple(
        (byte >> shift) & 1
        for byte in codeword
        for shift in range(7, -1, -1)
    )


@lru_cache(maxsize=PHASE_COUNT)
def phase_permutation(phase: int) -> tuple[int, ...]:
    _validate_phase(phase)
    seed = hashlib.sha256(
        PHASE_PERMUTATION_PREFIX + str(phase).encode("ascii")
    ).digest()
    values = list(range(CODEWORD_BITS))
    random.Random(int.from_bytes(seed, "big")).shuffle(values)
    return tuple(values)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    if type(permutation) is not tuple:
        raise TypeError("permutation must be a tuple")
    if (
        len(permutation) != CODEWORD_BITS
        or any(type(value) is not int for value in permutation)
        or sorted(permutation) != list(range(CODEWORD_BITS))
    ):
        raise ValueError("permutation must contain each index from 0 through 63")
    inverse = [0] * CODEWORD_BITS
    for logical, physical in enumerate(permutation):
        inverse[physical] = logical
    return tuple(inverse)


def permute_codeword_bits(codeword: bytes, phase: int) -> tuple[int, ...]:
    _validate_phase(phase)
    logical_bits = bytes_to_bits(codeword)
    physical_bits = [0] * CODEWORD_BITS
    for logical, physical in enumerate(phase_permutation(phase)):
        physical_bits[physical] = logical_bits[logical]
    return tuple(physical_bits)


def phase_for_tile(tile_x: int, tile_y: int) -> int:
    if type(tile_x) is not int or type(tile_y) is not int:
        raise TypeError("tile coordinates must be integers")
    if tile_x < 0 or tile_y < 0:
        raise ValueError("tile coordinates must be nonnegative")
    return (tile_x + 2 * tile_y) % PHASE_COUNT


def candidate_match_probability(candidate_count: int) -> float:
    if type(candidate_count) is not int:
        raise TypeError("candidate count must be an integer")
    if not 1 <= candidate_count <= 8:
        raise ValueError("candidate count must be between 1 and 8")
    return candidate_count / (2**32)


def _validate_phase(phase: int) -> None:
    if type(phase) is not int:
        raise TypeError("phase must be an integer")
    if phase not in range(PHASE_COUNT):
        raise ValueError("phase must be between 0 and 3")


__all__ = (
    "AUTH_KEY_MIN_BYTES",
    "AUTH_MESSAGE_PREFIX",
    "AUTH_TAG_BYTES",
    "CandidateDecode",
    "CODEWORD_BITS",
    "PHASE_COUNT",
    "PHASE_PERMUTATION_PREFIX",
    "RS_CODEWORD_BYTES",
    "RS_DATA_BYTES",
    "RS_ERASURE_COUNTS",
    "RS_PARITY_BYTES",
    "authentication_tag",
    "bytes_to_bits",
    "candidate_match_probability",
    "decode_candidate_codeword",
    "encode_codeword",
    "inverse_permutation",
    "phase_for_tile",
    "phase_permutation",
    "permute_codeword_bits",
)
