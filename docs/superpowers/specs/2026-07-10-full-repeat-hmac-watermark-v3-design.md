# Full-Repeat HMAC Watermark V3 Design

## Goal

Build an experimental robust watermark v3 that preserves one complete 64-bit candidate code in every robust tile, reduces carrier-specific bias through deterministic phase permutations, and never attributes a trace without content alignment plus exact candidate-code evidence.

## Constraints

- Production remains on robust watermark v1 until v3 passes every quick promotion gate.
- Robust strength must not exceed the strongest quality-approved v3 setting.
- Visual similarity, residual similarity, short codes, and dense scans cannot return a v3 trace.
- Existing v1 and experimental v2 records remain readable through version dispatch.
- Online candidate count remains at most eight and the total detection budget remains five seconds.

## Candidate Authentication Code

V3 uses a 64-bit HMAC-derived code instead of a public trace hash:

```python
HMAC-SHA256(WATERMARK_AUTH_KEY, b"robust-v3:" + trace_id.encode("utf-8"))[:8]
```

`WATERMARK_AUTH_KEY` must contain at least 32 UTF-8 bytes before a v3 image can be generated. Missing or short keys produce an explicit HTTP 503 response. The implementation must never silently fall back to v1 while recording v3.

The generated record stores the resulting code as 16 lowercase hexadecimal characters in `robust_auth_code`. Storing the code preserves detection across key rotation; the key remains necessary to create new v3 codes. The record also stores:

```json
{
  "robust_watermark_version": 3,
  "robust_watermark_codec": "hmac64_full_repeat_phase_permutation_v3"
}
```

This design prevents code derivation from a public trace ID alone. It does not claim protection after both database contents and watermarked source images are compromised.

## Full-Repetition Carrier Layout

Every existing `128x128` robust tile embeds all 64 HMAC bits exactly once using the current `8x8` cell grid, blue channel, carrier amplitude, and tile coverage. V3 does not split payload data across phases.

For tile grid coordinates `(tile_x, tile_y)`:

```python
phase = (tile_x + 2 * tile_y) % 3
```

Each phase has a fixed permutation of the 64 physical carrier indices, generated deterministically from `ROBUST_MAGIC`, v3, and the phase number. The permutation maps a logical HMAC bit to a physical cell/carrier. Adjacent tiles therefore observe the same logical bit through different spatial cells and carrier patterns while retaining complete payload repetition.

## Aligned Soft Decoder

After the existing ORB/RANSAC alignment, the decoder:

1. Enumerates covered robust tiles in original-image coordinates.
2. Requires at least two covered tiles and at least two distinct phases.
3. Computes all 64 physical carrier correlations per tile.
4. Applies the inverse phase permutation to accumulate correlations by logical bit.
5. Normalizes each tile score vector by its median absolute score before aggregation so one high-energy tile cannot dominate.
6. Hard-decides the aggregate vector only after all tiles are combined.
7. Compares the 64 observed bits with the candidate record's stored `robust_auth_code`.

V3 accepts at most eight hard bit errors. For a uniformly random 64-bit observation, the Hamming-ball probability is recorded in a unit test and must remain below `1e-8` per candidate. The decoder also requires positive mean signed agreement with the expected bits; this prevents a numerically degenerate zero-score vector from passing.

The code is only a candidate confirmation mechanism. It cannot scan the database by decoded value. Exactly one aligned candidate must pass; multiple passing candidates return no result.

## Version Dispatch

- Missing version or version `1`: existing v1 aligned decoder.
- Version `2`: existing experimental three-phase RS decoder.
- Version `3`: new full-repeat HMAC decoder.

V2 and v3 records are excluded from dense v1 robust detection. Exact fingerprint, full LSB, block LSB, and registered-original rejection retain their existing ordering.

The v3 result uses:

- mode: `aligned_robust_hmac_v3`
- label: `几何对齐 HMAC 认证水印`
- diagnostics: `bit_errors`, `authenticated_tiles`, `phase_tile_counts`, `mean_signed_agreement`, `mean_abs_score`, alignment evidence, and elapsed time.

## Quality And Safety

V3 changes bit-to-carrier assignment but does not increase the number of modified pixels per tile or the configured amplitude. Quality is still measured after all watermark layers are composed.

Required quality gates:

- Minimum PSNR `>=38 dB`.
- Minimum SSIM `>=0.95`.
- Intact recovery `5/5`.
- Intact original false positives `0/5`.

The initial strength scan tests `0.74`, then lower values only if quality fails. It never increases above `0.74` during v3 promotion testing.

## Quick Promotion Gate

Use the established deterministic 60-case matrix:

- Five images.
- Scales `0.5` and `1.5`.
- Crop ratios `0.3`, `0.5`, and `0.8`.
- One deterministic crop per combination.
- Matching 30 unwatermarked negative cases.

Promotion requires all of:

- Correct recall greater than `46.67%`.
- The 30% crop bucket has recall `>=80%`.
- The 50% and 80% crop buckets each have recall `>=95%`.
- Wrong trace count `0`.
- False positives `0/30`.
- P95 latency `<=5s`.
- Quality gates remain satisfied.

If any safety, quality, crop-bucket, or latency gate fails, v3 remains experimental. Retain diagnostics and keep production on v1. Do not increase the eight-bit radius without a statistically independent negative calibration set.

## Long-Test Boundary

Do not start full attack rounds or 1,000+ negative testing until the quick gate passes. After a quick pass, report results and estimated runtime before starting long tests.

## Tests

- Deterministic HMAC code generation, key validation, and key separation.
- Three valid permutations with inverse mapping and complete 0-63 coverage.
- Same per-tile damage budget as v1 at equal strength.
- Identity and scaled-crop positive recovery.
- Wrong candidate, unwatermarked image, one-phase coverage, ambiguous candidates, and missing auth code all reject.
- V1/v2/v3 dispatch and dense-v1 isolation.
- Benchmark configuration records version, strength, and diagnostics.
- Full pytest, Python compilation, and PowerShell parsing before and after promotion testing.
