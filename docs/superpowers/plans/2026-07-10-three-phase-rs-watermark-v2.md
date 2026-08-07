# Three-Phase Reed-Solomon Watermark V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned three-phase `RS(24,8)` robust watermark that improves crop/scale recovery without increasing image damage or allowing non-code evidence to attribute a trace.

**Architecture:** A focused `watermark_ecc.py` module owns Reed-Solomon encoding, bounded erasure decoding, and phase layout. `main.py` keeps image carriers and alignment, dispatches v1/v2 by record version, and requires exact candidate payload verification plus a unique passing candidate. Benchmarks explicitly generate v2 until quality, safety, latency, and recall gates justify changing the production default.

**Tech Stack:** Python 3.13, `reedsolo==1.7.0`, NumPy, Pillow, OpenCV, FastAPI, pytest.

**Repository note:** This workspace is not a Git repository. Do not create commits, branches, or worktrees.

---

### Task 1: Add The Isolated RS Codec

**Files:**
- Modify: `requirements.txt`
- Create: `watermark_ecc.py`
- Create: `tests/test_watermark_ecc.py`

- [ ] **Step 1: Pin and install the dependency**

Add exactly this line to `requirements.txt`:

```text
reedsolo==1.7.0
```

Run:

```powershell
python -m pip install reedsolo==1.7.0
```

Expected: installation succeeds from the `py3-none-any` wheel and `python -c "import reedsolo; print(reedsolo.__version__)"` prints `1.7.0`.

- [ ] **Step 2: Write failing deterministic codec tests**

Create `tests/test_watermark_ecc.py` with tests that import these wished-for APIs:

```python
import pytest

from watermark_ecc import (
    RS_CODEWORD_BYTES,
    RS_DATA_BYTES,
    decode_expected_codeword,
    encode_codeword,
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
    assert decode_expected_codeword(encoded, bytes.fromhex("acd3000000000000"), [1.0] * 24) is None


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
        encoded[index] ^= (index + 1)
    assert decode_expected_codeword(bytes(encoded), PAYLOAD, [1.0] * 24) is None
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m pytest tests/test_watermark_ecc.py -q
```

Expected: collection fails because `watermark_ecc` does not exist.

- [ ] **Step 4: Implement the minimal codec module**

Create `watermark_ecc.py` with these public constants and functions:

```python
from typing import Any

from reedsolo import RSCodec, ReedSolomonError

RS_DATA_BYTES = 8
RS_PARITY_BYTES = 16
RS_CODEWORD_BYTES = 24
RS_ERASURE_COUNTS = (0, 2, 4, 6)
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
    confidence_order = sorted(range(RS_CODEWORD_BYTES), key=lambda index: byte_confidences[index])
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
        }
    return None
```

- [ ] **Step 5: Verify GREEN**

Run `python -m pytest tests/test_watermark_ecc.py -q`.

Expected: all codec tests pass.

---

### Task 2: Implement Three-Phase Layout And V2 Embedding

**Files:**
- Modify: `watermark_ecc.py`
- Modify: `main.py:506-516, 1836-1867`
- Modify: `tests/test_watermark_ecc.py`
- Modify: `tests/test_aligned_authenticated_detection.py`

- [ ] **Step 1: Write failing phase-layout tests**

Add to `tests/test_watermark_ecc.py`:

```python
from watermark_ecc import codeword_phase, tile_phase


def test_tile_phase_rotates_horizontally_and_vertically():
    assert [tile_phase(x, 0) for x in range(6)] == [0, 1, 2, 0, 1, 2]
    assert [tile_phase(0, y) for y in range(6)] == [0, 2, 1, 0, 2, 1]


def test_codeword_phase_returns_exact_eight_byte_section():
    codeword = bytes(range(24))
    assert codeword_phase(codeword, 0) == bytes(range(0, 8))
    assert codeword_phase(codeword, 1) == bytes(range(8, 16))
    assert codeword_phase(codeword, 2) == bytes(range(16, 24))
```

Add a test to `tests/test_aligned_authenticated_detection.py` that embeds v2 into a textured `512x512` image and compares the changed-pixel count per robust tile against v1. Require the same tile dimensions and no greater maximum pixel delta at equal strength.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_watermark_ecc.py tests/test_aligned_authenticated_detection.py -q
```

Expected: failure because phase helpers and `embed_robust_watermark_v2` do not exist.

- [ ] **Step 3: Add phase helpers**

Add to `watermark_ecc.py`:

```python
RS_PHASES = 3
RS_PHASE_BYTES = 8


def tile_phase(tile_x: int, tile_y: int) -> int:
    return (int(tile_x) + 2 * int(tile_y)) % RS_PHASES


def codeword_phase(codeword: bytes, phase: int) -> bytes:
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("RS watermark codeword must be exactly 24 bytes")
    if phase not in range(RS_PHASES):
        raise ValueError("RS watermark phase must be 0, 1, or 2")
    start = phase * RS_PHASE_BYTES
    return codeword[start : start + RS_PHASE_BYTES]
```

- [ ] **Step 4: Add v2 payload and embedding functions**

In `main.py`, import `codeword_phase`, `encode_codeword`, and `tile_phase`. Add:

```python
ROBUST_WATERMARK_VERSION_V1 = 1
ROBUST_WATERMARK_VERSION_V2 = 2
ROBUST_WATERMARK_CODEC_V2 = "rs_24_8_three_phase"


def robust_payload_bytes(trace_id: str) -> bytes:
    return robust_code_from_trace(trace_id).to_bytes(8, "big")


def embed_robust_watermark_v2(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    codeword = encode_codeword(robust_payload_bytes(trace_id))
    for x, y in iter_robust_tiles(image.width, image.height):
        phase = tile_phase(x // ROBUST_TILE, y // ROBUST_TILE)
        phase_bytes = codeword_phase(codeword, phase)
        bits = robust_bits_from_code(int.from_bytes(phase_bytes, "big"))
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, ROBUST_GRID)
            y0 = y + row * ROBUST_CELL
            x0 = x + col * ROBUST_CELL
            patch = arr[y0:y0 + ROBUST_CELL, x0:x0 + ROBUST_CELL, ROBUST_CHANNEL]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch + robust_pattern(bit_index, ROBUST_CELL) * ROBUST_DELTA * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")
```

Keep the existing `embed_robust_watermark` unchanged as the v1 writer used by legacy unit tests and compatibility paths.

- [ ] **Step 5: Verify embedding tests**

Run the focused tests again. Expected: all pass and v2 does not increase per-tile modification count or maximum delta relative to v1.

---

### Task 3: Decode Three Aligned Phases With Bounded Erasures

**Files:**
- Modify: `main.py:1917-2019`
- Modify: `tests/test_aligned_authenticated_detection.py`

- [ ] **Step 1: Write failing v2 decoder tests**

Add tests that construct an alignment from a v2-watermarked textured image and assert:

```python
def test_aligned_rs_v2_decoder_recovers_exact_trace():
    # Build a 768x512 textured image so every phase has multiple robust tiles.
    # Embed v2, create full valid mask, then decode against the exact record.
    decoded = main.decode_aligned_robust_trace_v2(alignment, record)
    assert decoded["trace_id"] == record["trace_id"]
    assert decoded["phase_tile_counts"] == [8, 8, 8]
    assert decoded["corrected_symbols"] <= 8


def test_aligned_rs_v2_decoder_rejects_wrong_record():
    assert main.decode_aligned_robust_trace_v2(alignment, wrong_record) is None


def test_aligned_rs_v2_decoder_requires_two_tiles_per_phase():
    restricted_mask = mask_that_leaves_one_phase_with_one_tile()
    assert main.decode_aligned_robust_trace_v2(
        {**alignment, "valid_mask": restricted_mask}, record
    ) is None


def test_aligned_rs_v2_decoder_rejects_unwatermarked_image():
    assert main.decode_aligned_robust_trace_v2(unwatermarked_alignment, record) is None
```

Use deterministic arrays and real carrier functions; do not mock `decode_expected_codeword`.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_aligned_authenticated_detection.py -q`.

Expected: failures because `decode_aligned_robust_trace_v2` does not exist.

- [ ] **Step 3: Implement score-to-codeword conversion**

Add private helpers in `main.py`:

```python
def _scores_to_byte(scores: np.ndarray) -> tuple[int, float]:
    value = 0
    absolute = np.abs(scores.astype(np.float64))
    scale = max(1e-6, float(np.median(absolute)))
    for score in scores:
        value = (value << 1) | int(score > 0)
    return value, float(np.min(absolute / scale))


def _phase_scores_to_codeword(
    phase_scores: np.ndarray,
    phase_counts: list[int],
) -> tuple[bytes, list[float]]:
    observed = bytearray()
    confidences = []
    for phase in range(3):
        average = phase_scores[phase] / max(1, phase_counts[phase])
        for start in range(0, ROBUST_BITS, 8):
            value, confidence = _scores_to_byte(average[start:start + 8])
            observed.append(value)
            confidences.append(confidence)
    return bytes(observed), confidences
```

- [ ] **Step 4: Implement the v2 aligned decoder**

Implement `decode_aligned_robust_trace_v2(alignment, record)` using the same validation and tile normalization as v1, with these differences:

1. Maintain `phase_scores = np.zeros((3, 64), dtype=np.float64)` and `phase_counts = [0, 0, 0]`.
2. Determine each tile phase from original aligned grid coordinates.
3. Reject unless all phase counts are at least two.
4. Convert scores with `_phase_scores_to_codeword`.
5. Call `decode_expected_codeword(observed, robust_payload_bytes(trace_id), confidences)`.
6. Return `None` on decode failure.
7. Return record, trace, corrected-symbol count, erasure count, phase counts, and mean absolute score on success.

- [ ] **Step 5: Verify GREEN**

Run the focused aligned detector tests. Expected: v1 tests remain green and all new v2 decoder tests pass.

---

### Task 4: Add Versioned API Generation And Detection Dispatch

**Files:**
- Modify: `main.py:1836-1850, 2021-2076, 3140-3250`
- Modify: `.env.example`
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `tests/test_false_positive_gate.py`

- [ ] **Step 1: Write failing version and migration tests**

Add tests for these behaviors:

```python
def test_v2_record_uses_rs_embedding_metadata(client, source_file):
    response = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", source_file, "image/png")},
        data={"user_id": "v2-test", "robust_watermark_version": "2"},
    )
    record = response.json()
    assert record["robust_watermark_version"] == 2
    assert record["robust_watermark_codec"] == "rs_24_8_three_phase"


def test_missing_record_version_dispatches_to_v1(monkeypatch):
    # Assert the legacy decoder is called and v2 decoder is not called.


def test_v2_record_never_falls_through_to_fuzzy_v1(monkeypatch):
    # Make v2 decoding fail and assert no v1 robust decoder receives this record.


def test_two_authenticated_v2_candidates_return_none():
    # Two passing candidate records must not produce attribution.
```

The migration tests may spy on dispatch functions, but positive and negative codec tests must continue to use real code.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_aligned_authenticated_detection.py tests/test_false_positive_gate.py -q
```

Expected: version metadata and dispatch tests fail.

- [ ] **Step 3: Add safe version parsing and API input**

Add:

```python
DEFAULT_ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")


def normalize_robust_watermark_version(value: str | int | None) -> int:
    try:
        version = int(value if value is not None else DEFAULT_ROBUST_WATERMARK_VERSION)
    except (TypeError, ValueError):
        version = 1
    return 2 if version == 2 else 1
```

Add `robust_watermark_version: str = Form(DEFAULT_ROBUST_WATERMARK_VERSION)` to the embed endpoint. Select v1 or v2 embedding explicitly, then persist version and codec metadata in the record.

Keep `.env.example` at:

```text
ROBUST_WATERMARK_VERSION=1
```

until Task 7 proves every promotion gate.

- [ ] **Step 4: Dispatch the aligned decoder by record version**

Inside `detect_aligned_authenticated_watermark`, use:

```python
version = normalize_robust_watermark_version(record.get("robust_watermark_version", 1))
decoded = (
    decode_aligned_robust_trace_v2(alignment, record)
    if version == 2
    else decode_aligned_robust_trace(alignment, record)
)
```

Return v2 mode `aligned_robust_rs_v2` and label `几何对齐 RS 认证水印` only for a unique v2 result. Include codec, corrected symbols, erasure count, phase counts, alignment evidence, and elapsed time. Keep the existing v1 response unchanged.

Filter records passed to dense v1 robust functions so records with version 2 cannot be fuzzily interpreted as v1.

- [ ] **Step 5: Verify migration and safety tests**

Run the focused tests. Expected: exact v1 compatibility, v2 unique attribution, and all negative gates pass.

---

### Task 5: Make Commercial Benchmarks Version-Aware

**Files:**
- Modify: `tests/commercial_benchmark_config.py`
- Modify: `tests/commercial_quality_benchmark.py`
- Modify: `tests/commercial_trace_benchmark.py`
- Modify: `tests/commercial_attack_benchmark.py`
- Modify: `tests/commercial_negative_benchmark.py`
- Modify: `tests/test_commercial_benchmark_config.py`
- Modify: `run_commercial_benchmark.ps1`

- [ ] **Step 1: Write failing benchmark configuration tests**

Extend `tests/test_commercial_benchmark_config.py`:

```python
def test_embedding_form_sends_explicit_robust_version(monkeypatch):
    monkeypatch.setenv("ROBUST_WATERMARK_VERSION", "2")
    form = build_embedding_form("benchmark-user", "1.0")
    assert form["robust_watermark_version"] == "2"
```

Add assertions that every benchmark JSON `settings` section records `robust_watermark_version`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_commercial_benchmark_config.py tests/test_commercial_benchmark_gates.py -q
```

Expected: the form lacks `robust_watermark_version`.

- [ ] **Step 3: Propagate and record v2 settings**

Add this entry in `build_embedding_form`:

```python
"robust_watermark_version": os.getenv("ROBUST_WATERMARK_VERSION", "1"),
```

Each benchmark reads `ROBUST_WATERMARK_VERSION`, includes it in reports/settings, and sends it through the shared form. Add the environment name to `$managedEnvironment` in `run_commercial_benchmark.ps1`. Quality runs set it to `2`; reuse checks reject reports from a different robust version.

- [ ] **Step 4: Verify benchmark tests and script syntax**

Run:

```powershell
python -m pytest tests/test_commercial_benchmark_config.py tests/test_commercial_benchmark_gates.py -q
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path 'run_commercial_benchmark.ps1'),
    [ref]$null,
    [ref]$errors
)
if ($errors.Count) { $errors; exit 1 }
```

Expected: tests pass and PowerShell reports no parse errors.

---

### Task 6: Run Code Verification Before Commercial Testing

**Files:**
- No new files

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_watermark_ecc.py tests/test_aligned_authenticated_detection.py tests/test_false_positive_gate.py tests/test_commercial_benchmark_config.py -q
```

Expected: all focused tests pass with zero failures.

- [ ] **Step 2: Run the complete suite**

```powershell
python -m pytest -q
```

Expected: all tests pass. Deprecation warnings may remain, but no test errors or failures are permitted.

- [ ] **Step 3: Compile all changed Python files**

```powershell
python -m py_compile main.py watermark_ecc.py tests/commercial_benchmark_config.py tests/commercial_quality_benchmark.py tests/commercial_trace_benchmark.py tests/commercial_attack_benchmark.py tests/commercial_negative_benchmark.py
```

Expected: exit code 0 with no output.

---

### Task 7: Run Quality And 60-Case Promotion Gates

**Files:**
- Generated: `test_output/commercial_quality_benchmark/*`
- Generated: `test_output/commercial_trace_benchmark/*`
- Modify: `docs/commercial_watermark_upgrade_plan.md`
- Modify: `test_output/commercial_balanced_assessment.md`
- Conditionally modify: `.env.example`

- [ ] **Step 1: Run the v2 quality gate**

```powershell
$env:ROBUST_WATERMARK_VERSION='2'
$env:ROBUST_WATERMARK_STRENGTH='1.0'
$env:FIDELITY_LEVELS='1.0'
$env:SMALL_CROP_TRACE_STRENGTH='0.35'
$env:SMALL_CROP_TRACE_DENSITY='medium'
$env:QUALITY_PROBE_FILTER='intact'
python tests/commercial_quality_benchmark.py
```

Require minimum PSNR `>=38`, minimum SSIM `>=0.95`, wrong trace `0`, and false positives `0`. Stop if any gate fails; do not run the crop matrix.

- [ ] **Step 2: Run the deterministic 60-case matrix**

Only after quality passes:

```powershell
$env:FIDELITY_LEVEL='1.0'
$env:ROBUST_WATERMARK_VERSION='2'
$env:ROBUST_WATERMARK_STRENGTH='1.0'
$env:SCALE_FACTORS='0.5,1.5'
$env:CROP_RATIOS='0.3,0.5,0.8'
$env:NEGATIVE_SCALE_FACTORS='0.5,1.5'
$env:NEGATIVE_CROP_RATIOS='0.3,0.5,0.8'
$env:CROPS_PER_RATIO='1'
$env:BENCHMARK_WORKERS='20'
python tests/commercial_trace_benchmark.py
```

Compare against the exact promotion gates:

- Correct recall `>46.67%`.
- Wrong trace `0`.
- False positives `0/30`.
- P95 detection latency `<=5 seconds`.
- Quality remains above the Step 1 thresholds.

- [ ] **Step 3: Apply the promotion decision**

If every gate passes, change `.env.example` to:

```text
ROBUST_WATERMARK_VERSION=2
```

and change the code fallback for `DEFAULT_ROBUST_WATERMARK_VERSION` to `"2"`.

If any gate fails, keep the default at version 1. Leave v2 available only through explicit `ROBUST_WATERMARK_VERSION=2`, preserve all diagnostics, and do not weaken exact-payload or unique-candidate checks.

- [ ] **Step 4: Update reports with measured results**

Record actual v2 PSNR, SSIM, recall by scale/crop, wrong traces, false positives, P50/P95/max latency, corrected-symbol distribution, erasure distribution, and the promotion decision in:

- `docs/commercial_watermark_upgrade_plan.md`
- `test_output/commercial_balanced_assessment.md`

State explicitly that a 30-negative quick result does not prove a `<0.1%` false-positive rate.

- [ ] **Step 5: Re-run verification after the promotion decision**

Run `python -m pytest -q`, the compile command from Task 6, and the PowerShell parser command from Task 5. All must pass after any default-version change.

- [ ] **Step 6: Stop before long commercial tests**

Report the quick-gate result and estimate the runtime for five trace rounds and 1,000+ negatives. Do not start those tests without a new explicit user confirmation.
