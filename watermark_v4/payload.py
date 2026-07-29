"""Source-bound HMAC64 payload and RS(16,8) codec for V4."""

from __future__ import annotations

import hashlib
import hmac
import random
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

from reedsolo import RSCodec, ReedSolomonError


CODEC_ID = "hmac64_rs_16_8_split_repeat_sync_v4"
AUTH_TAG_BYTES = 8
AUTH_KEY_MIN_BYTES = 32
AUTH_MESSAGE_PREFIX = b"trace-v4-authentication-context-v1"
RS_DATA_BYTES = 8
RS_PARITY_BYTES = 8
RS_CODEWORD_BYTES = 16
RS_ERASURE_COUNTS = tuple(range(RS_PARITY_BYTES + 1))
PHASE_COUNT = 4
CODEWORD_BITS = RS_CODEWORD_BYTES * 8
CARRIER_BITS = CODEWORD_BITS // 2
PHASE_PERMUTATION_PREFIX = (
    b"hmac64_rs_16_8_split_repeat_sync_v4:carrier-permutation:"
)

_RS_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_CODEWORD_BYTES)


@dataclass(frozen=True, slots=True)
class AuthContext:
    codec_version: str
    key_id: str
    owner_user_id: int
    source_pixel_sha256: bytes
    trace_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("codec_version", self.codec_version),
            ("key_id", self.key_id),
            ("trace_id", self.trace_id),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty canonical string")
        try:
            self.codec_version.encode("ascii")
            self.key_id.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("codec_version and key_id must be ASCII") from error
        if type(self.owner_user_id) is not int or not 0 < self.owner_user_id < 2**64:
            raise ValueError("owner_user_id must be an unsigned 64-bit positive integer")
        if (
            type(self.source_pixel_sha256) is not bytes
            or len(self.source_pixel_sha256) != 32
        ):
            raise ValueError("source_pixel_sha256 must contain exactly 32 bytes")


@dataclass(frozen=True, slots=True)
class CandidateDecode:
    payload: bytes
    corrected_codeword: bytes
    corrected_symbols: int
    erasure_count: int
    bit_errors: int


def _field(value: bytes) -> bytes:
    if len(value) >= 2**32:
        raise ValueError("canonical authentication field is too large")
    return len(value).to_bytes(4, "big") + value


def canonical_auth_message(context: AuthContext) -> bytes:
    if type(context) is not AuthContext:
        raise TypeError("authentication context must be an exact AuthContext")
    values = (
        context.codec_version.encode("ascii"),
        context.key_id.encode("ascii"),
        context.owner_user_id.to_bytes(8, "big"),
        context.source_pixel_sha256,
        context.trace_id.encode("utf-8"),
    )
    return AUTH_MESSAGE_PREFIX + b"".join(_field(value) for value in values)


def _validate_key(key: bytes) -> bytes:
    if type(key) is not bytes:
        raise TypeError("authentication key must be bytes")
    if len(key) < AUTH_KEY_MIN_BYTES:
        raise ValueError("authentication key must contain at least 32 bytes")
    return key


def authentication_tag(context: AuthContext, key: bytes) -> bytes:
    return hmac.new(
        _validate_key(key),
        canonical_auth_message(context),
        hashlib.sha256,
    ).digest()[:AUTH_TAG_BYTES]


def verify_authentication_tag(context: AuthContext, key: bytes, tag: bytes) -> bool:
    if type(tag) is not bytes or len(tag) != AUTH_TAG_BYTES:
        return False
    return hmac.compare_digest(authentication_tag(context, key), tag)


def encode_codeword(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("v4 payload must be bytes")
    if len(payload) != RS_DATA_BYTES:
        raise ValueError("v4 payload must be exactly 8 bytes")
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
    if (
        type(byte_confidences) not in (list, tuple)
        or len(byte_confidences) != RS_CODEWORD_BYTES
        or any(not _is_finite_confidence(value) for value in byte_confidences)
    ):
        return None

    expected_codeword = encode_codeword(expected_payload)
    confidence_order = sorted(
        range(RS_CODEWORD_BYTES),
        key=lambda index: (byte_confidences[index], index),
    )
    for erasure_count in RS_ERASURE_COUNTS:
        try:
            decoded, corrected, errata = _RS_CODEC.decode(
                observed,
                erase_pos=confidence_order[:erasure_count],
            )
        except (ReedSolomonError, ValueError, IndexError, TypeError):
            continue
        decoded_bytes = bytes(decoded)
        corrected_bytes = bytes(corrected)
        if not hmac.compare_digest(decoded_bytes, expected_payload):
            continue
        if not hmac.compare_digest(corrected_bytes, expected_codeword):
            continue
        return CandidateDecode(
            payload=decoded_bytes,
            corrected_codeword=corrected_bytes,
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
    return type(value) is float and isfinite(value) and 0.0 <= value <= 1.0


def bytes_to_bits(codeword: bytes) -> tuple[int, ...]:
    if type(codeword) is not bytes:
        raise TypeError("v4 codeword must be bytes")
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("v4 codeword must be exactly 16 bytes")
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
    values = list(range(CARRIER_BITS))
    random.Random(int.from_bytes(seed, "big")).shuffle(values)
    return tuple(values)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    if type(permutation) is not tuple:
        raise TypeError("permutation must be a tuple")
    if (
        len(permutation) != CARRIER_BITS
        or any(type(value) is not int for value in permutation)
        or sorted(permutation) != list(range(CARRIER_BITS))
    ):
        raise ValueError("permutation must contain every V4 codeword bit index")
    inverse = [0] * CARRIER_BITS
    for logical, physical in enumerate(permutation):
        inverse[physical] = logical
    return tuple(inverse)


def permute_codeword_bits(
    codeword: bytes, phase: int, carrier_class: int
) -> tuple[int, ...]:
    _validate_carrier_class(carrier_class)
    all_bits = bytes_to_bits(codeword)
    start = carrier_class * CARRIER_BITS
    logical_bits = all_bits[start : start + CARRIER_BITS]
    physical_bits = [0] * CARRIER_BITS
    for logical, physical in enumerate(phase_permutation(phase)):
        physical_bits[physical] = logical_bits[logical]
    return tuple(physical_bits)


def phase_for_tile(tile_x: int, tile_y: int) -> int:
    if type(tile_x) is not int or type(tile_y) is not int:
        raise TypeError("tile coordinates must be integers")
    if tile_x < 0 or tile_y < 0:
        raise ValueError("tile coordinates must be nonnegative")
    return (tile_x + 2 * tile_y) % PHASE_COUNT


def carrier_class_for_tile(tile_x: int, tile_y: int) -> int:
    if type(tile_x) is not int or type(tile_y) is not int:
        raise TypeError("tile coordinates must be integers")
    if tile_x < 0 or tile_y < 0:
        raise ValueError("tile coordinates must be nonnegative")
    return (tile_x + tile_y) % 2


def _validate_carrier_class(carrier_class: int) -> None:
    if type(carrier_class) is not int or carrier_class not in (0, 1):
        raise ValueError("carrier class must be zero or one")


def _validate_phase(phase: int) -> None:
    if type(phase) is not int:
        raise TypeError("phase must be an integer")
    if phase not in range(PHASE_COUNT):
        raise ValueError("phase must be between 0 and 3")


__all__ = (
    "AUTH_KEY_MIN_BYTES",
    "AUTH_MESSAGE_PREFIX",
    "AUTH_TAG_BYTES",
    "AuthContext",
    "CODEC_ID",
    "CARRIER_BITS",
    "CODEWORD_BITS",
    "CandidateDecode",
    "PHASE_COUNT",
    "PHASE_PERMUTATION_PREFIX",
    "RS_CODEWORD_BYTES",
    "RS_DATA_BYTES",
    "RS_ERASURE_COUNTS",
    "RS_PARITY_BYTES",
    "authentication_tag",
    "bytes_to_bits",
    "canonical_auth_message",
    "carrier_class_for_tile",
    "decode_candidate_codeword",
    "encode_codeword",
    "inverse_permutation",
    "phase_for_tile",
    "phase_permutation",
    "permute_codeword_bits",
    "verify_authentication_tag",
)
