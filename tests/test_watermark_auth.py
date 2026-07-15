import math

import pytest

from watermark_auth import (
    candidate_radius_probability,
    auth_code_from_trace,
    inverse_permutation,
    permuted_code_bits,
    phase_permutation,
)


KEY_A = "a" * 32
KEY_B = "b" * 32


def test_auth_code_is_deterministic_and_exactly_eight_bytes():
    first = auth_code_from_trace("TR-V3-TEST", KEY_A)
    second = auth_code_from_trace("TR-V3-TEST", KEY_A)

    assert first == second
    assert len(first) == 8


def test_auth_code_changes_with_key_or_trace():
    baseline = auth_code_from_trace("TR-V3-TEST", KEY_A)

    assert auth_code_from_trace("TR-V3-TEST", KEY_B) != baseline
    assert auth_code_from_trace("TR-V3-OTHER", KEY_A) != baseline


@pytest.mark.parametrize("key", [None, "", "short", b"x" * 31])
def test_auth_code_rejects_missing_or_short_keys(key):
    with pytest.raises(ValueError, match="at least 32 bytes"):
        auth_code_from_trace("TR-V3-TEST", key)


def test_phase_permutations_are_bijections_with_valid_inverses():
    permutations = [phase_permutation(phase) for phase in range(3)]

    assert len(set(permutations)) == 3
    for permutation in permutations:
        assert sorted(permutation) == list(range(64))
        inverse = inverse_permutation(permutation)
        assert all(inverse[physical] == logical for logical, physical in enumerate(permutation))


def test_phase_permutation_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        phase_permutation(3)


def test_every_phase_carries_the_complete_logical_code():
    code = bytes.fromhex("0123456789abcdef")
    logical_bits = tuple(
        (int.from_bytes(code, "big") >> shift) & 1
        for shift in range(63, -1, -1)
    )

    for phase in range(3):
        physical = permuted_code_bits(code, phase)
        permutation = phase_permutation(phase)
        recovered = [0] * 64
        for logical, physical_index in enumerate(permutation):
            recovered[logical] = physical[physical_index]
        assert tuple(recovered) == logical_bits


def test_eight_error_radius_has_commercially_small_random_probability():
    expected = sum(math.comb(64, count) for count in range(9)) / (2**64)

    assert candidate_radius_probability(8) == pytest.approx(expected)
    assert candidate_radius_probability(8) < 1e-8
