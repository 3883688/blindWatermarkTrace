import math

import pytest

from trace_app.watermark.ecc import (
    RS_CODEWORD_BYTES,
    RS_DATA_BYTES,
    codeword_phase,
    decode_expected_codeword,
    encode_codeword,
    tile_phase,
)


PAYLOAD = bytes.fromhex("acd3123456789abc")


def test_rs_24_8_encoding_is_deterministic():
    first = encode_codeword(PAYLOAD)
    second = encode_codeword(PAYLOAD)

    assert RS_DATA_BYTES == 8
    assert RS_CODEWORD_BYTES == 24
    assert first == second
    assert len(first) == 24
    assert first[:8] == PAYLOAD


@pytest.mark.parametrize("error_count", [1, 4, 8])
def test_rs_decoder_corrects_up_to_eight_symbol_errors(error_count):
    encoded = bytearray(encode_codeword(PAYLOAD))
    for index in range(error_count):
        encoded[index * 2] ^= 0x5A

    result = decode_expected_codeword(bytes(encoded), PAYLOAD, [1.0] * 24)

    assert result is not None
    assert result["payload"] == PAYLOAD
    assert result["corrected_codeword"] == encode_codeword(PAYLOAD)
    assert result["corrected_symbols"] == error_count


def test_rs_decoder_rejects_wrong_candidate_payload():
    encoded = encode_codeword(PAYLOAD)

    assert decode_expected_codeword(
        encoded,
        bytes.fromhex("acd3000000000000"),
        [1.0] * 24,
    ) is None


def test_rs_decoder_uses_bounded_low_confidence_erasures():
    encoded = bytearray(encode_codeword(PAYLOAD))
    damaged = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    confidence = [1.0] * 24
    for index in damaged:
        encoded[index] ^= 0xA5
        confidence[index] = 0.01

    result = decode_expected_codeword(bytes(encoded), PAYLOAD, confidence)

    assert result is not None
    assert result["erasure_count"] in {2, 4, 6}


def test_rs_decoder_rejects_corruption_outside_bound():
    encoded = bytearray(encode_codeword(PAYLOAD))
    for index in range(17):
        encoded[index] ^= index + 1

    assert decode_expected_codeword(bytes(encoded), PAYLOAD, [1.0] * 24) is None


def test_candidate_codeword_distance_recovers_bit_sparse_symbol_damage():
    encoded = bytearray(encode_codeword(PAYLOAD))
    for index in range(13):
        encoded[index] ^= 0x01
    for index in range(6):
        encoded[index] ^= 0x02

    result = decode_expected_codeword(bytes(encoded), PAYLOAD, [1.0] * 24)

    assert result is not None
    assert result["recovery_method"] == "expected_codeword_distance"
    assert result["bit_errors"] == 19
    assert result["corrected_symbols"] == 13


def test_candidate_distance_false_accept_bound_is_stricter_than_v1():
    v2_probability = sum(math.comb(192, count) for count in range(33)) / (2**192)
    v1_probability = sum(math.comb(64, count) for count in range(5)) / (2**64)

    assert v2_probability < 1e-20
    assert v2_probability < v1_probability


def test_tile_phase_rotates_horizontally_and_vertically():
    assert [tile_phase(x, 0) for x in range(6)] == [0, 1, 2, 0, 1, 2]
    assert [tile_phase(0, y) for y in range(6)] == [0, 2, 1, 0, 2, 1]


def test_codeword_phase_returns_exact_eight_byte_section():
    codeword = bytes(range(24))

    assert codeword_phase(codeword, 0) == bytes(range(0, 8))
    assert codeword_phase(codeword, 1) == bytes(range(8, 16))
    assert codeword_phase(codeword, 2) == bytes(range(16, 24))
