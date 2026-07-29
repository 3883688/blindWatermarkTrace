from dataclasses import replace
import hashlib
import hmac

import pytest

from trace_app.v4.keys import KeyRing
from watermark_v4.payload import (
    AUTH_MESSAGE_PREFIX,
    AUTH_TAG_BYTES,
    CODEC_ID,
    RS_CODEWORD_BYTES,
    AuthContext,
    authentication_tag,
    canonical_auth_message,
    decode_candidate_codeword,
    encode_codeword,
    verify_authentication_tag,
)


KEY = b"k" * 32
CONTEXT = AuthContext("v4", "key-2026-07", 9, b"s" * 32, "TR-1")


def _field(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def test_auth_tag_is_64_bit_source_and_owner_bound() -> None:
    expected_message = AUTH_MESSAGE_PREFIX + b"".join(
        _field(value)
        for value in (
            b"v4",
            b"key-2026-07",
            (9).to_bytes(8, "big"),
            b"s" * 32,
            b"TR-1",
        )
    )
    assert canonical_auth_message(CONTEXT) == expected_message
    tag = authentication_tag(CONTEXT, KEY)
    assert tag == hmac.new(KEY, expected_message, hashlib.sha256).digest()[:8]
    assert len(tag) == AUTH_TAG_BYTES == 8
    assert tag != authentication_tag(replace(CONTEXT, owner_user_id=10), KEY)
    assert tag != authentication_tag(
        replace(CONTEXT, source_pixel_sha256=b"t" * 32), KEY
    )
    assert CODEC_ID == "hmac64_rs_16_8_split_repeat_sync_v4"


def test_rs_16_8_corrects_four_unknown_symbols_and_eight_erasures() -> None:
    tag = authentication_tag(CONTEXT, KEY)
    codeword = encode_codeword(tag)
    assert len(codeword) == RS_CODEWORD_BYTES == 16
    assert codeword[:8] == tag

    damaged = bytearray(codeword)
    for index in (0, 3, 7, 12):
        damaged[index] ^= 0x5A
    result = decode_candidate_codeword(bytes(damaged), tag, [1.0] * 16)
    assert result is not None and result.payload == tag

    erased = bytearray(codeword)
    confidence = [1.0] * 16
    for order, index in enumerate(range(0, 16, 2)):
        erased[index] ^= 0xA5
        confidence[index] = order / 100
    result = decode_candidate_codeword(bytes(erased), tag, confidence)
    assert result is not None and result.erasure_count == 8


def test_record_verification_uses_constant_time_compare(monkeypatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    original = hmac.compare_digest

    def compared(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr("watermark_v4.payload.hmac.compare_digest", compared)
    tag = authentication_tag(CONTEXT, KEY)
    assert verify_authentication_tag(CONTEXT, KEY, tag)
    assert calls[-1] == (tag, tag)


def test_key_ring_hides_secrets_and_retries_group_local_collision() -> None:
    ring = KeyRing({"key-old": b"o" * 32, "key-2026-07": KEY}, "key-2026-07")
    assert KEY.hex() not in repr(ring)
    first = ring.sign(CONTEXT)
    contexts = iter((CONTEXT, replace(CONTEXT, trace_id="TR-2")))
    selected, tag = ring.issue_unique(lambda _attempt: next(contexts), {first}.__contains__)
    assert selected.trace_id == "TR-2"
    assert tag != first
    assert ring.verify(selected, tag)


@pytest.mark.parametrize(
    "changes",
    [
        {"owner_user_id": 0},
        {"source_pixel_sha256": b"short"},
        {"trace_id": " TR-1"},
        {"key_id": ""},
    ],
)
def test_auth_context_rejects_noncanonical_fields(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(CONTEXT, **changes)
