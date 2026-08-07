# V4 Phase 0B Payload Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Establish the immutable v4 codec configuration, HMAC32 candidate tag, RS(8,4) codeword, four carrier permutations, and bounded candidate-specific decoder before image-domain work begins.

**Architecture:** `watermark_v4` is independent from `main.py`, persistence, and OpenCV. `config.py` owns validated versioned constants and online budgets; `payload.py` owns authentication and error correction. The public package exports only stable types/functions needed by later DCT, embed, and detect modules.

**Tech Stack:** Python 3.13, dataclasses, hashlib/hmac, reedsolo 1.7.0, pytest

---

## File Structure

- Create `watermark_v4/__init__.py`: stable public codec exports.
- Create `watermark_v4/config.py`: frozen v4 configuration and validation.
- Create `watermark_v4/payload.py`: HMAC32, RS(8,4), bit conversion, phase permutations, bounded decode.
- Create `tests/test_watermark_v4_config.py`: configuration and budget tests.
- Create `tests/test_watermark_v4_payload.py`: authentication, codec, safety-bound, and phase tests.

## Task 1: Frozen V4 Configuration

**Files:**
- Create: `watermark_v4/__init__.py`
- Create: `watermark_v4/config.py`
- Create: `tests/test_watermark_v4_config.py`

- [x] **Step 1: Write failing configuration tests**

```python
import pytest

from watermark_v4.config import V4Config


def test_default_config_matches_approved_codec():
    config = V4Config()
    assert config.version == 4
    assert config.codec == "hmac32_rs_8_4_full_repeat_sync_v4"
    assert config.tile_size == 128
    assert config.grid_size == 8
    assert config.cell_size == 16
    assert config.online_p95_seconds == 10.0
    assert config.hard_timeout_seconds == 16.0
    assert config.candidate_limit == 3
    assert config.minimum_tiles == 2
    assert config.minimum_phases == 2


@pytest.mark.parametrize("field,value", [
    ("dct_margin", 1.99),
    ("dct_margin", 10.01),
    ("pilot_amplitude", 0.24),
    ("pilot_amplitude", 1.26),
    ("hard_timeout_seconds", 9.99),
])
def test_config_rejects_out_of_contract_values(field, value):
    with pytest.raises(ValueError):
        V4Config(**{field: value})
```

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_watermark_v4_config.py -q`  
Expected: collection failure because `watermark_v4.config` does not exist.

- [x] **Step 3: Implement frozen validated configuration**

Create a frozen dataclass with the approved defaults:

```python
@dataclass(frozen=True, slots=True)
class V4Config:
    version: int = 4
    codec: str = "hmac32_rs_8_4_full_repeat_sync_v4"
    tile_size: int = 128
    grid_size: int = 8
    cell_size: int = 16
    dct_margin: float = 6.0
    dct_margin_min: float = 2.0
    dct_margin_max: float = 10.0
    pilot_amplitude: float = 0.75
    pilot_amplitude_min: float = 0.25
    pilot_amplitude_max: float = 1.25
    analysis_max_side: int = 1024
    tile_minimum_coverage: float = 0.70
    minimum_tiles: int = 2
    minimum_phases: int = 2
    candidate_limit: int = 3
    online_p95_seconds: float = 10.0
    hard_timeout_seconds: float = 16.0
```

Also define immutable coefficient pairs, pilot vectors, calibration sets, and per-stage budgets from the approved design. `__post_init__` rejects inconsistent dimensions, values outside hard ranges, nonpositive budgets, a hard timeout below the P95 target, and candidate/tile/phase counts outside their approved bounds.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_watermark_v4_config.py -q`  
Expected: all configuration tests pass.

## Task 2: HMAC32 Candidate Tag

**Files:**
- Create: `watermark_v4/payload.py`
- Create: `tests/test_watermark_v4_payload.py`

- [x] **Step 1: Write failing authentication tests**

```python
import pytest

from watermark_v4.payload import authentication_tag


KEY_A = b"a" * 32
KEY_B = b"b" * 32


def test_v4_tag_is_deterministic_and_domain_separated():
    tag = authentication_tag("TR-V4-TEST", KEY_A)
    assert tag == authentication_tag("TR-V4-TEST", KEY_A)
    assert len(tag) == 4
    assert tag != authentication_tag("TR-V4-OTHER", KEY_A)
    assert tag != authentication_tag("TR-V4-TEST", KEY_B)


@pytest.mark.parametrize("key", [None, b"", b"x" * 31, "short"])
def test_v4_tag_rejects_short_keys(key):
    with pytest.raises(ValueError, match="at least 32 bytes"):
        authentication_tag("TR-V4-TEST", key)


@pytest.mark.parametrize("trace_id", [None, "", "   ", 123])
def test_v4_tag_rejects_invalid_trace_ids(trace_id):
    with pytest.raises((TypeError, ValueError)):
        authentication_tag(trace_id, KEY_A)
```

- [x] **Step 2: Run authentication tests and verify RED**

Run: `python -m pytest tests/test_watermark_v4_payload.py -q`  
Expected: import failure because `watermark_v4.payload` does not exist.

- [x] **Step 3: Implement minimal authentication**

Use the exact prefix `b"robust-v4:"`, require a nonempty string trace ID, require at least 32 encoded key bytes, and return the first four bytes of HMAC-SHA256. Error messages must not include key or tag values.

- [x] **Step 4: Run authentication tests and verify GREEN**

Run: `python -m pytest tests/test_watermark_v4_payload.py -q`  
Expected: authentication tests pass.

## Task 3: RS(8,4) Encoding And Bounded Decode

**Files:**
- Modify: `watermark_v4/payload.py`
- Modify: `tests/test_watermark_v4_payload.py`

- [x] **Step 1: Add failing codec tests**

```python
from watermark_v4.payload import decode_candidate_codeword, encode_codeword


TAG = bytes.fromhex("12345678")


def test_rs_8_4_codeword_is_deterministic():
    codeword = encode_codeword(TAG)
    assert len(codeword) == 8
    assert codeword[:4] == TAG
    assert codeword == encode_codeword(TAG)


@pytest.mark.parametrize("errors", [1, 2])
def test_rs_8_4_corrects_up_to_two_unknown_bytes(errors):
    damaged = bytearray(encode_codeword(TAG))
    for index in range(errors):
        damaged[index * 3] ^= 0xA5
    result = decode_candidate_codeword(bytes(damaged), TAG, [1.0] * 8)
    assert result is not None
    assert result.payload == TAG
    assert result.corrected_codeword == encode_codeword(TAG)


def test_wrong_candidate_never_authenticates():
    observed = encode_codeword(TAG)
    assert decode_candidate_codeword(
        observed,
        bytes.fromhex("87654321"),
        [1.0] * 8,
    ) is None
```

- [x] **Step 2: Run codec tests and verify RED**

Run: `python -m pytest tests/test_watermark_v4_payload.py -q`  
Expected: failures because codec functions are missing.

- [x] **Step 3: Implement RS codec and immutable result type**

Use `RSCodec(4, nsize=8)`. Define a frozen `CandidateDecode` dataclass containing payload, corrected codeword, corrected symbols, erasure count, and observed bit errors.

Try erasure counts `(0, 1, 2, 3, 4)` in order, using the lowest byte confidences. Accept only when Reed-Solomon returns the exact expected four-byte tag and exact expected corrected codeword. Catch controlled Reed-Solomon/input errors and return `None`. Do not add distance fallback.

- [x] **Step 4: Add boundary tests**

Cover mixed error/erasure cases allowed by `2e+s <= 4`, corruption outside the bound, malformed observed/tag/confidence lengths, NaN/infinite confidences, and deterministic lowest-confidence ordering.

- [x] **Step 5: Run codec tests and verify GREEN**

Run: `python -m pytest tests/test_watermark_v4_payload.py -q`  
Expected: all codec tests pass.

## Task 4: Bit Mapping And Four Phase Permutations

**Files:**
- Modify: `watermark_v4/payload.py`
- Modify: `tests/test_watermark_v4_payload.py`
- Modify: `watermark_v4/__init__.py`

- [x] **Step 1: Add failing phase tests**

```python
from watermark_v4.payload import (
    bytes_to_bits,
    inverse_permutation,
    phase_for_tile,
    phase_permutation,
    permute_codeword_bits,
)


def test_four_phase_permutations_are_distinct_bijections():
    permutations = [phase_permutation(phase) for phase in range(4)]
    assert len(set(permutations)) == 4
    for permutation in permutations:
        assert sorted(permutation) == list(range(64))
        inverse = inverse_permutation(permutation)
        assert all(inverse[physical] == logical for logical, physical in enumerate(permutation))


def test_every_phase_carries_complete_codeword():
    codeword = bytes.fromhex("0011223344556677")
    logical = bytes_to_bits(codeword)
    for phase in range(4):
        physical = permute_codeword_bits(codeword, phase)
        recovered = [0] * 64
        for logical_index, physical_index in enumerate(phase_permutation(phase)):
            recovered[logical_index] = physical[physical_index]
        assert tuple(recovered) == logical


def test_tile_phase_is_stable():
    assert [phase_for_tile(x, 0) for x in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [phase_for_tile(0, y) for y in range(4)] == [0, 2, 0, 2]
```

- [x] **Step 2: Run phase tests and verify RED**

Run: `python -m pytest tests/test_watermark_v4_payload.py -q`  
Expected: failures because phase functions are missing.

- [x] **Step 3: Implement deterministic phase mapping**

Derive each permutation seed with SHA-256 over `b"hmac32_rs_8_4_full_repeat_sync_v4:carrier-permutation:" + phase_ascii`, shuffle `0..63` using a local `random.Random`, validate inverse inputs strictly, and implement `phase_for_tile(x, y) = (x + 2*y) % 4` for nonnegative integer coordinates.

- [x] **Step 4: Add random acceptance-bound test**

Assert `8 / 2**32 < 1e-8`. This is a documented candidate-tag bound, not a substitute for negative-image testing.

- [x] **Step 5: Export stable public API and verify GREEN**

Export `V4Config`, `CandidateDecode`, authentication, codec, and phase helpers from `watermark_v4/__init__.py` through an explicit `__all__`.

Run: `python -m pytest tests/test_watermark_v4_config.py tests/test_watermark_v4_payload.py -q`  
Expected: all v4 foundation tests pass.

## Task 5: Foundation Regression Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-14-v4-phase-0b-payload.md` checkboxes only

- [x] **Step 1: Compile the new package**

Run: `python -m py_compile watermark_v4/__init__.py watermark_v4/config.py watermark_v4/payload.py`  
Expected: exit 0.

- [x] **Step 2: Run focused v4 tests with warnings as errors**

Run: `python -m pytest tests/test_watermark_v4_config.py tests/test_watermark_v4_payload.py -q -W error`  
Expected: all tests pass with no warnings.

- [x] **Step 3: Run the complete existing suite**

Run: `python -m pytest -q`  
Expected: no failures. Existing third-party/Pillow warnings are recorded rather than hidden.

- [x] **Step 4: Verify scope**

Confirm no changes to `main.py`, legacy watermark modules, benchmark gates, production data, uploads, or the protected V3 archive. Record exact files and test evidence.

## Exit Gate

This batch completes only when configuration, authentication, RS correction, malformed-input rejection, four phase permutations, public exports, package compilation, focused warning-free tests, and the full existing suite all pass. It does not claim image-domain embedding or commercial recall improvement.

## Verification Evidence

- Focused V4 foundation suite with warnings as errors: `113 passed`.
- Package compilation: exit `0`.
- Final complete project suite: `585 passed, 2 skipped, 184 warnings in 154.93s`.
- Protected V3 archive SHA-256 remained `abf858cc83e92a0691afd92955abbcc9199a310839b21ee8105cb61e30b0b3c2` and matched `SHA256SUMS`.
- `main.py`, legacy watermark modules, production data, uploads, and the protected V3 archive were not modified by this foundation batch.

