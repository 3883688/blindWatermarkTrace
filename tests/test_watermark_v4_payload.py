import hashlib
import hmac

import pytest

import watermark_v4

from watermark_v4.payload import (
    AUTH_KEY_MIN_BYTES,
    AUTH_MESSAGE_PREFIX,
    AUTH_TAG_BYTES,
    CandidateDecode,
    authentication_tag,
    bytes_to_bits,
    candidate_match_probability,
    decode_candidate_codeword,
    encode_codeword,
    inverse_permutation,
    phase_for_tile,
    phase_permutation,
    permute_codeword_bits,
)


KEY_A = b"a" * 32
KEY_B = b"b" * 32


def test_v4_tag_matches_independent_hmac_vector():
    trace_id = "TR-V4-TEST"
    expected = hmac.new(
        KEY_A,
        b"robust-v4:" + trace_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()[:4]

    assert authentication_tag(trace_id, KEY_A) == expected
    assert AUTH_TAG_BYTES == 4
    assert AUTH_KEY_MIN_BYTES == 32
    assert AUTH_MESSAGE_PREFIX == b"robust-v4:"


def test_v4_tag_is_deterministic_and_separated_by_trace_and_key():
    baseline = authentication_tag("TR-V4-TEST", KEY_A)

    assert baseline == authentication_tag("TR-V4-TEST", KEY_A)
    assert baseline != authentication_tag("TR-V4-OTHER", KEY_A)
    assert baseline != authentication_tag("TR-V4-TEST", KEY_B)


def test_v4_tag_accepts_unicode_trace_and_key_by_utf8_byte_length():
    tag = authentication_tag("TR-V4-用户", "密" * 11)

    assert len(tag) == 4


@pytest.mark.parametrize("key", [b"", b"x" * 31, "short", "密" * 10])
def test_v4_tag_rejects_missing_or_short_keys(key):
    with pytest.raises((TypeError, ValueError), match="at least 32 bytes"):
        authentication_tag("TR-V4-TEST", key)


@pytest.mark.parametrize("key", [None, bytearray(b"x" * 32), memoryview(b"x" * 32), 123])
def test_v4_tag_rejects_unsupported_key_types(key):
    with pytest.raises(TypeError, match="string or bytes"):
        authentication_tag("TR-V4-TEST", key)


@pytest.mark.parametrize("trace_id", [None, "", "   ", " TR-V4", "TR-V4 ", 123])
def test_v4_tag_rejects_invalid_or_noncanonical_trace_ids(trace_id):
    with pytest.raises((TypeError, ValueError), match="trace_id"):
        authentication_tag(trace_id, KEY_A)


def test_v4_tag_errors_do_not_echo_key_material():
    secret = "private-key-material"

    with pytest.raises(ValueError) as captured:
        authentication_tag("TR-V4-TEST", secret)

    assert secret not in str(captured.value)


TAG = bytes.fromhex("12345678")


def test_rs_8_4_codeword_is_deterministic_and_systematic():
    codeword = encode_codeword(TAG)

    assert len(codeword) == 8
    assert codeword[:4] == TAG
    assert codeword == encode_codeword(TAG)


@pytest.mark.parametrize("error_count", [1, 2])
def test_rs_8_4_corrects_up_to_two_unknown_bytes(error_count):
    damaged = bytearray(encode_codeword(TAG))
    for index in range(error_count):
        damaged[index * 3] ^= 0xA5

    result = decode_candidate_codeword(bytes(damaged), TAG, [1.0] * 8)

    assert isinstance(result, CandidateDecode)
    assert result.payload == TAG
    assert result.corrected_codeword == encode_codeword(TAG)
    assert result.corrected_symbols == error_count
    assert result.erasure_count == 0


def test_rs_8_4_uses_low_confidence_erasures_within_bound():
    damaged = bytearray(encode_codeword(TAG))
    confidence = [1.0] * 8
    for index in (0, 2, 5):
        damaged[index] ^= 0x5A
    confidence[0] = 0.01
    confidence[2] = 0.02

    result = decode_candidate_codeword(bytes(damaged), TAG, confidence)

    assert result is not None
    assert result.payload == TAG
    assert result.erasure_count == 2
    assert result.corrected_symbols == 3


@pytest.mark.parametrize(
    "damaged_indices,low_confidence_indices,expected_erasures",
    [
        ((0,), (), 0),
        ((0, 3), (), 0),
        ((0, 2, 5), (0, 2), 2),
        ((0, 2, 4, 6), (0, 2, 4, 6), 4),
    ],
)
def test_rs_8_4_covers_bounded_error_erasure_combinations(
    damaged_indices,
    low_confidence_indices,
    expected_erasures,
):
    damaged = bytearray(encode_codeword(TAG))
    confidence = [1.0] * 8
    for index in damaged_indices:
        damaged[index] ^= 0x33
    for order, index in enumerate(low_confidence_indices):
        confidence[index] = 0.01 + order * 0.01

    result = decode_candidate_codeword(bytes(damaged), TAG, confidence)

    assert result is not None
    assert result.erasure_count == expected_erasures


def test_equal_confidences_use_stable_byte_index_order():
    damaged = bytearray(encode_codeword(TAG))
    for index in (0, 1, 2):
        damaged[index] ^= 0x33

    result = decode_candidate_codeword(bytes(damaged), TAG, [0.5] * 8)

    assert result is not None
    assert result.erasure_count == 2


def test_rs_8_4_rejects_damage_outside_bounded_attempts():
    damaged = bytearray(encode_codeword(TAG))
    for index in (4, 5, 6):
        damaged[index] ^= 0x7F
    confidence = [0.01, 0.02, 0.03, 0.04, 1.0, 1.0, 1.0, 1.0]

    assert decode_candidate_codeword(bytes(damaged), TAG, confidence) is None


def test_rs_8_4_rejects_wrong_candidate_tag():
    assert decode_candidate_codeword(
        encode_codeword(TAG),
        bytes.fromhex("87654321"),
        [1.0] * 8,
    ) is None


@pytest.mark.parametrize(
    "observed,expected,confidence",
    [
        (b"short", TAG, [1.0] * 8),
        (b"12345678", b"bad", [1.0] * 8),
        (b"12345678", TAG, [1.0] * 7),
        (b"12345678", TAG, [1.0] * 7 + [float("nan")]),
        (b"12345678", TAG, [1.0] * 7 + [float("inf")]),
        (encode_codeword(TAG), TAG, [1.0] * 7 + [-0.1]),
        (encode_codeword(TAG), TAG, [1.0] * 7 + [10**1000]),
        (encode_codeword(TAG), TAG, [1.0] * 7 + [True]),
        (encode_codeword(TAG), TAG, [1.0] * 7 + ["high"]),
        (bytearray(b"12345678"), TAG, [1.0] * 8),
    ],
)
def test_rs_8_4_rejects_malformed_inputs(observed, expected, confidence):
    assert decode_candidate_codeword(observed, expected, confidence) is None


def test_rs_8_4_result_is_immutable():
    result = decode_candidate_codeword(encode_codeword(TAG), TAG, [1.0] * 8)

    assert result is not None
    with pytest.raises((AttributeError, TypeError)):
        result.erasure_count = 4


def test_four_phase_permutations_are_distinct_bijections():
    permutations = [phase_permutation(phase) for phase in range(4)]

    assert len(set(permutations)) == 4
    for permutation in permutations:
        assert type(permutation) is tuple
        assert sorted(permutation) == list(range(64))
        inverse = inverse_permutation(permutation)
        assert all(
            inverse[physical] == logical
            for logical, physical in enumerate(permutation)
        )


def test_every_phase_carries_the_complete_codeword():
    codeword = bytes.fromhex("0011223344556677")
    logical = bytes_to_bits(codeword)

    for phase in range(4):
        physical = permute_codeword_bits(codeword, phase)
        recovered = [0] * 64
        for logical_index, physical_index in enumerate(phase_permutation(phase)):
            recovered[logical_index] = physical[physical_index]
        assert tuple(recovered) == logical


def test_tile_phase_is_stable_in_registered_coordinates():
    assert [phase_for_tile(x, 0) for x in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [phase_for_tile(0, y) for y in range(4)] == [0, 2, 0, 2]


@pytest.mark.parametrize("phase", [-1, 4, True, 1.0, "1"])
def test_phase_functions_reject_invalid_phases(phase):
    with pytest.raises((TypeError, ValueError), match="phase"):
        phase_permutation(phase)
    with pytest.raises((TypeError, ValueError), match="phase"):
        permute_codeword_bits(bytes(8), phase)


@pytest.mark.parametrize("coordinates", [(-1, 0), (0, -1), (True, 0), (0, 1.0)])
def test_tile_phase_rejects_invalid_coordinates(coordinates):
    with pytest.raises((TypeError, ValueError), match="tile"):
        phase_for_tile(*coordinates)


def test_inverse_permutation_rejects_malformed_values():
    for malformed in (
        list(range(64)),
        tuple(range(63)),
        tuple(range(63)) + (62,),
        tuple(range(63)) + (True,),
    ):
        with pytest.raises((TypeError, ValueError), match="permutation"):
            inverse_permutation(malformed)


@pytest.mark.parametrize("value", [b"", bytes(7), bytes(9), bytearray(8)])
def test_codeword_bit_mapping_rejects_non_codewords(value):
    with pytest.raises((TypeError, ValueError), match="codeword"):
        bytes_to_bits(value)


def test_eight_candidate_random_match_bound_is_below_internal_limit():
    assert candidate_match_probability(8) == 8 / (2**32)
    assert candidate_match_probability(8) < 1e-8


def test_package_exports_only_stable_v4_foundation_api():
    assert set(watermark_v4.__all__) == {
        "CandidateDecode",
        "SyncEstimate",
        "V4Config",
        "TileScores",
        "authentication_tag",
        "bytes_to_bits",
        "candidate_match_probability",
        "decode_candidate_codeword",
        "detect_pilot",
        "encode_codeword",
        "embed_codeword",
        "embed_pilot",
        "extract_image_tiles",
        "inverse_permutation",
        "phase_for_tile",
        "phase_permutation",
        "permute_codeword_bits",
    }


@pytest.mark.parametrize("candidate_count", [-1, 0, 9, True, 1.0])
def test_candidate_match_probability_rejects_counts_outside_online_bound(candidate_count):
    with pytest.raises((TypeError, ValueError), match="candidate"):
        candidate_match_probability(candidate_count)
