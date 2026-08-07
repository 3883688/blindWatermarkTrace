import pytest

import watermark_v4
from watermark_v4.payload import (
    CODEWORD_BITS,
    CandidateDecode,
    bytes_to_bits,
    decode_candidate_codeword,
    encode_codeword,
    inverse_permutation,
    phase_for_tile,
    phase_permutation,
    permute_codeword_bits,
)


TAG = bytes.fromhex("1234567890abcdef")


def test_rs_16_8_codeword_is_systematic_and_deterministic() -> None:
    codeword = encode_codeword(TAG)
    assert len(codeword) == 16
    assert codeword[:8] == TAG
    assert codeword == encode_codeword(TAG)


@pytest.mark.parametrize("error_count", [1, 2, 3, 4])
def test_rs_16_8_corrects_up_to_four_unknown_symbols(error_count: int) -> None:
    damaged = bytearray(encode_codeword(TAG))
    for index in range(error_count):
        damaged[index * 3] ^= 0xA5
    result = decode_candidate_codeword(bytes(damaged), TAG, [1.0] * 16)
    assert isinstance(result, CandidateDecode)
    assert result.payload == TAG
    assert result.corrected_symbols == error_count
    assert result.erasure_count == 0


def test_rs_16_8_uses_low_confidence_erasures() -> None:
    damaged = bytearray(encode_codeword(TAG))
    confidence = [1.0] * 16
    for order, index in enumerate((0, 2, 4, 6, 8, 10, 12, 14)):
        damaged[index] ^= 0x33
        confidence[index] = order / 100
    result = decode_candidate_codeword(bytes(damaged), TAG, confidence)
    assert result is not None
    assert result.erasure_count == 8
    assert result.corrected_symbols == 8


def test_rs_rejects_wrong_tag_and_malformed_observations() -> None:
    codeword = encode_codeword(TAG)
    assert decode_candidate_codeword(codeword, b"wrongtag", [1.0] * 16) is None
    assert decode_candidate_codeword(codeword[:-1], TAG, [1.0] * 16) is None
    assert decode_candidate_codeword(codeword, TAG, [1.0] * 15) is None
    assert decode_candidate_codeword(
        codeword, TAG, [1.0] * 15 + [float("nan")]
    ) is None


def test_decode_result_is_immutable() -> None:
    result = decode_candidate_codeword(encode_codeword(TAG), TAG, [1.0] * 16)
    assert result is not None
    with pytest.raises((AttributeError, TypeError)):
        result.erasure_count = 8  # type: ignore[misc]


def test_phase_permutations_cover_each_64_bit_carrier_half() -> None:
    assert CODEWORD_BITS == 128
    permutations = [phase_permutation(phase) for phase in range(4)]
    assert len(set(permutations)) == 4
    for permutation in permutations:
        assert sorted(permutation) == list(range(64))
        inverse = inverse_permutation(permutation)
        assert all(
            inverse[physical] == logical
            for logical, physical in enumerate(permutation)
        )


def test_every_phase_round_trips_the_complete_codeword() -> None:
    codeword = bytes.fromhex("00112233445566778899aabbccddeeff")
    logical = bytes_to_bits(codeword)
    for phase in range(4):
        for carrier_class in (0, 1):
            physical = permute_codeword_bits(codeword, phase, carrier_class)
            recovered = [0] * 64
            for logical_index, physical_index in enumerate(phase_permutation(phase)):
                recovered[logical_index] = physical[physical_index]
            start = carrier_class * 64
            assert tuple(recovered) == logical[start : start + 64]


def test_tile_phase_is_stable() -> None:
    assert [phase_for_tile(x, 0) for x in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [phase_for_tile(0, y) for y in range(4)] == [0, 2, 0, 2]


@pytest.mark.parametrize("value", [b"", bytes(8), bytes(15), bytearray(16)])
def test_bit_mapping_rejects_non_codewords(value) -> None:
    with pytest.raises((TypeError, ValueError), match="codeword"):
        bytes_to_bits(value)


def test_package_no_longer_exports_candidate_probability() -> None:
    assert "candidate_match_probability" not in watermark_v4.__all__
    assert {"AuthContext", "canonical_auth_message", "verify_authentication_tag"} <= set(
        watermark_v4.__all__
    )
