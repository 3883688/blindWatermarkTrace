import hashlib
import hmac
import math
import random


AUTH_CODE_BYTES = 8
AUTH_CODE_BITS = AUTH_CODE_BYTES * 8
AUTH_KEY_MIN_BYTES = 32
AUTH_MESSAGE_PREFIX = b"robust-v3:"
AUTH_PERMUTATION_PREFIX = b"robust-v3-carrier-permutation:"


def _validated_key(key: str | bytes | None) -> bytes:
    if isinstance(key, str):
        encoded = key.encode("utf-8")
    elif isinstance(key, bytes):
        encoded = key
    else:
        encoded = b""
    if len(encoded) < AUTH_KEY_MIN_BYTES:
        raise ValueError("WATERMARK_AUTH_KEY must contain at least 32 bytes")
    return encoded


def auth_code_from_trace(trace_id: str, key: str | bytes | None) -> bytes:
    validated = _validated_key(key)
    message = AUTH_MESSAGE_PREFIX + str(trace_id).encode("utf-8")
    return hmac.new(validated, message, hashlib.sha256).digest()[:AUTH_CODE_BYTES]


def phase_permutation(phase: int) -> tuple[int, ...]:
    if phase not in range(3):
        raise ValueError("watermark carrier phase must be 0, 1, or 2")
    seed = hashlib.sha256(
        AUTH_PERMUTATION_PREFIX + str(phase).encode("ascii")
    ).digest()
    values = list(range(AUTH_CODE_BITS))
    random.Random(int.from_bytes(seed, "big")).shuffle(values)
    return tuple(values)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    if len(permutation) != AUTH_CODE_BITS or sorted(permutation) != list(range(AUTH_CODE_BITS)):
        raise ValueError("watermark carrier permutation must contain 0 through 63")
    inverse = [0] * AUTH_CODE_BITS
    for logical, physical in enumerate(permutation):
        inverse[physical] = logical
    return tuple(inverse)


def permuted_code_bits(code: bytes, phase: int) -> tuple[int, ...]:
    if len(code) != AUTH_CODE_BYTES:
        raise ValueError("v3 watermark auth code must be exactly 8 bytes")
    logical_bits = tuple(
        (int.from_bytes(code, "big") >> shift) & 1
        for shift in range(AUTH_CODE_BITS - 1, -1, -1)
    )
    physical_bits = [0] * AUTH_CODE_BITS
    for logical, physical in enumerate(phase_permutation(phase)):
        physical_bits[physical] = logical_bits[logical]
    return tuple(physical_bits)


def candidate_radius_probability(max_errors: int, bits: int = AUTH_CODE_BITS) -> float:
    if max_errors < 0 or max_errors > bits:
        raise ValueError("max_errors must be within the code length")
    return sum(math.comb(bits, count) for count in range(max_errors + 1)) / (2**bits)
