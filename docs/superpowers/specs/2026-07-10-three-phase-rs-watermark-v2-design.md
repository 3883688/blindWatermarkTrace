# Three-Phase Reed-Solomon Watermark V2 Design

## Status

Approved direction: introduce a versioned three-phase `RS(24,8)` robust watermark while preserving legacy v1 extraction and the current image-quality ceiling.

## Goals

1. Improve recovery after scaling and random cropping without increasing robust strength above `1.0`.
2. Preserve the current 64-bit trace payload: 16-bit magic plus a 48-bit BLAKE2b trace hash.
3. Correct substantially more damaged evidence than the v1 hard limit of four bit errors.
4. Never return a trace from visual similarity, residual similarity, short codes, or ECC decoding alone.
5. Keep new and legacy watermarked images traceable during migration.

## Non-Goals

- This version does not claim resistance to an attacker who deliberately forges a known watermark code.
- It does not increase the online candidate limit or the five-second detection budget.
- It does not enable dense legacy scans by default.
- It does not start the 1,000/10,000-negative commercial runs until the quick quality and recall gates pass.
- It does not modify the other LSB, frequency, small-crop, dot-matrix, or visible-watermark formats.

## Constraints And Findings

Each robust tile is `128x128` pixels and contains an `8x8` grid of 64 physical bit cells. Putting strong ECC into a single tile would shorten the trace tag and weaken candidate uniqueness. Increasing the number of cells or strength would change the measured image-quality boundary.

`bchlib 2.1.3` has no binary distribution for the current Windows/Python 3.13 environment. Requiring a local C compiler would make deployment unreliable. `reedsolo 1.7.0` is a small pure-Python package with a platform-independent wheel, so v2 uses shortened Reed-Solomon coding over GF(256).

## Codeword Format

The v2 logical payload remains eight bytes:

| Bytes | Meaning |
| --- | --- |
| `0..1` | Existing 16-bit `ROBUST_MAGIC`, big-endian |
| `2..7` | Existing 48-bit BLAKE2b digest of `trace_id` |

`RSCodec(16, nsize=24)` encodes the eight payload bytes with 16 parity bytes to produce a 24-byte codeword. The code can correct up to eight unknown erroneous bytes, or a mixture of errors and erasures satisfying `2 * errors + erasures <= 16`.

The decoder must not trust a successfully decoded Reed-Solomon message by itself. It must verify all of the following:

1. The decoded eight bytes exactly equal `robust_code_from_trace(record.trace_id)` serialized as eight big-endian bytes.
2. The corrected codeword is consistent with the candidate record's expected v2 codeword.
3. At least two covered tiles exist for every phase used in the final decision.
4. Alignment satisfies the existing inlier, ratio, and coverage gates.
5. Exactly one candidate trace passes. Zero or multiple passing candidates return no result.

## Three-Phase Spatial Layout

The 24-byte codeword is split into three consecutive eight-byte phase payloads:

- Phase 0 embeds codeword bytes `0..7`.
- Phase 1 embeds codeword bytes `8..15`.
- Phase 2 embeds codeword bytes `16..23`.

For a robust tile at grid coordinate `(tile_x, tile_y)`, the phase is:

```python
phase = (tile_x + 2 * tile_y) % 3
```

Each tile still embeds exactly 64 bits using the current carrier, cell size, channel, and strength. The modification count per tile does not increase. Adjacent horizontal and vertical tiles rotate through phases, so a normal rectangular crop containing several tiles supplies all three codeword sections.

The phase formula uses original-image tile coordinates after homography alignment. It never uses query-image coordinates, which would change after cropping or scaling.

## Soft Aggregation And Erasures

The aligned decoder maintains a separate 64-element score vector and tile count for each phase. It normalizes each covered tile to `128x128`, computes the existing carrier correlation for every cell, and sums scores only into that tile's phase.

For each phase:

1. Require at least two tiles with at least 70% valid-mask coverage.
2. Convert every group of eight signed bit scores into one byte.
3. Calculate byte confidence as the minimum normalized absolute bit score in that byte.
4. Rank all 24 bytes by confidence.

The first implementation tries bounded decoding variants in this order:

1. Hard-decision decode without erasures.
2. Decode with the lowest-confidence 2 bytes marked as erasures.
3. Decode with the lowest-confidence 4 bytes marked as erasures.
4. Decode with the lowest-confidence 6 bytes marked as erasures.

No unbounded combination search is permitted. The decoder stops at the first variant that produces the candidate's exact expected payload and a valid corrected codeword.

Implementation evidence showed that splitting spatial repetition across three phases can produce more than eight erroneous byte symbols even when the total number of wrong bits remains low. After bounded RS decoding fails, v2 therefore permits one candidate-specific fallback: compare the observed 192-bit codeword with the current aligned candidate's exact expected RS codeword and accept at no more than 32 bit errors. This is not an arbitrary RS-codeword match and cannot discover a record. The random-code acceptance volume is approximately `5.68e-22` per candidate, compared with approximately `3.68e-14` for the v1 `4/64` radius.

Diagnostic output records the recovery method, bit errors, erasure count, corrected symbol count, phase tile counts, and per-byte confidence.

## Versioning And Migration

Newly generated records receive:

```json
{
  "robust_watermark": true,
  "robust_watermark_version": 2,
  "robust_watermark_codec": "rs_24_8_three_phase"
}
```

New images embed only robust v2 in the robust layer. Other existing watermark layers remain unchanged.

Records missing `robust_watermark_version`, or explicitly set to `1`, continue through the current v1 robust decoder. Version 2 records only use the v2 aligned decoder; they must not fall through to fuzzy v1 robust attribution. Exact LSB and block-LSB extraction remain first in the overall extraction order.

The dependency is pinned as `reedsolo==1.7.0`. If it cannot be imported, application startup or watermark generation must fail with an explicit dependency error. It must not silently generate a v1 image while recording version 2.

## Detection Flow

1. Run exact file fingerprint, full LSB, and registered-original rejection as today.
2. Build the existing bounded candidate list and apply ORB/RANSAC alignment.
3. Dispatch by `robust_watermark_version`:
   - v1: existing 64-bit aligned decoder.
   - v2: three-phase aggregation and bounded RS decode.
4. Collect candidates that pass exact candidate-payload verification.
5. Return a result only when exactly one trace passes.
6. Preserve the existing five-second overall budget. Budget exhaustion returns no result.

The result mode for v2 is `aligned_robust_rs_v2`, with the Chinese label `几何对齐 RS 认证水印`. The response includes `codec`, `corrected_symbols`, `erasure_count`, `phase_tile_counts`, alignment evidence, and elapsed time.

## Safety Model

Reed-Solomon correction and candidate-specific 192-bit distance confirmation are used for recovery, not candidate discovery. Candidate discovery remains content-alignment based and bounded to at most eight records. A decoded payload or nearby codeword cannot identify an arbitrary database record; it can only confirm the record currently being checked.

The expected codeword comparison makes accidental acceptance far narrower than accepting any valid RS codeword. A collision of the existing 48-bit trace hashes is also handled conservatively: if two aligned candidates share a passing payload, the unique-candidate rule returns no result.

Unwatermarked images, registered originals, short-code matches, visual matches, and residual matches must all remain unable to return a v2 trace without exact v2 payload confirmation.

## Test Strategy

### Codec Unit Tests

- Encoding is deterministic and produces exactly 24 bytes.
- Round-trip restores the exact eight-byte payload.
- Up to eight corrupted byte symbols are corrected.
- Mixed errors and erasures pass only when `2e+s <= 16`.
- Corruption beyond the bound raises a controlled decode failure or fails exact expected-payload verification.
- A different trace payload never authenticates the candidate.

### Layout Unit Tests

- Horizontal and vertical neighboring tiles rotate through all three phases.
- Every tile embeds exactly one 64-bit phase payload.
- Phase selection is stable under aligned scaling and cropping.
- Missing any phase or having fewer than two covered tiles in a phase returns no result.

### Detector Tests

- A deterministic scaled crop recovers the exact v2 trace.
- A legacy v1 scaled crop still follows the v1 decoder.
- Unwatermarked, wrong-record, ambiguous-record, blank, and insufficient-feature images return no result.
- Budget exhaustion returns no result.
- Dense and residual-only fallbacks remain disabled for attribution.

### Quality And Commercial Quick Gate

Use the five images in `img/`, `fidelity=1.0`, `small_crop_trace=medium/0.35`, and robust strength `1.0`.

Required quality:

- Minimum PSNR `>= 38 dB`.
- Minimum SSIM `>= 0.95`.
- No increase in the robust layer's configured strength.

Required 60-case quick matrix:

- Scales `0.5` and `1.5`.
- Crop ratios `0.3`, `0.5`, and `0.8`.
- One deterministic crop per combination.
- Correct recall must exceed the v1 safe maximum of `46.67%`.
- Wrong trace count must be zero.
- All 30 matching unwatermarked cases must produce zero false positives.
- P95 detection latency must be no more than five seconds.

If any safety or quality gate fails, v2 is not enabled as the production default. If only recall fails, retain the implementation behind an explicit feature flag and use the diagnostics to decide whether a four-phase code or a revised carrier is justified.

## Long-Test Boundary

Passing the quick gate permits, but does not automatically start, longer testing. Before running five full trace rounds, the complete attack matrix, or 1,000+ negatives, report the quick results and estimated runtime to the user and obtain confirmation.
