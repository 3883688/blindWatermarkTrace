# V4 Synchronized DCT Watermark Design

**Date:** 2026-07-14  
**Status:** Approved by user  
**Depends on:** `2026-07-13-commercial-v4-watermark-design.md`

## 1. Objective

Build a new v4 invisible watermark for real social-platform propagation and 30%/50%/80% crop attribution. V4 uses one authenticated format rather than stacking the existing independent DCT, DWT, FFT, short-code, dot-matrix, and legacy robust attribution paths.

The commercial path does not detect v1, v2, or v3 images. Existing formats remain available only in isolated historical tools until removed. V4 qualification still requires the commercial gates defined by the Phase 0A design; Phase 0B itself first establishes a testable implementation and quick regression gate.

## 2. Selected Architecture

V4 assigns one responsibility to each signal-processing component:

```text
content feature index
  -> at most three candidate records
FFT pilot synchronization
  -> initial rotation, scale, translation, and tile phase
DCT differential carriers
  -> repeated 64-bit authenticated RS codeword
multi-tile soft aggregation
  -> bounded RS(8,4) decoding
candidate HMAC verification
  -> exactly one candidate or no attribution
ORB/RANSAC alignment
  -> fallback geometry only when FFT synchronization is insufficient
```

DWT is disabled in the first v4 format. It may be evaluated behind an experimental flag only after the DCT/FFT implementation is measured and only when image-quality headroom remains.

## 3. Payload And Authentication

### 3.1 Candidate Tag

The logical payload is four bytes:

```python
HMAC_SHA256(
    WATERMARK_AUTH_KEY,
    b"robust-v4:" + trace_id.encode("utf-8"),
)[:4]
```

`WATERMARK_AUTH_KEY` must contain at least 32 UTF-8 bytes. A missing or short key prevents v4 generation with an explicit controlled error. There is no fallback to another watermark version.

The four-byte tag is encoded with shortened Reed-Solomon `RS(8,4)` over GF(256), producing an eight-byte/64-bit codeword. `reedsolo==1.7.0` remains the implementation dependency.

The record stores:

```json
{
  "robust_watermark_version": 4,
  "robust_watermark_codec": "hmac32_rs_8_4_full_repeat_sync_v4",
  "robust_auth_code": "8 lowercase hexadecimal characters"
}
```

The key is never stored in records, images, API responses, logs, fixtures, or reports. Storing the tag allows detection after key rotation. A database compromise therefore exposes candidate tags, so the tag is a candidate-confirmation mechanism rather than an anti-forgery claim after database compromise.

### 3.2 Acceptance Bound

RS decoding is accepted only when the recovered four-byte payload exactly matches the current aligned candidate's stored authentication tag. The decoder never searches the database by recovered tag.

With at most eight aligned candidates, a uniformly random recovered payload matches any expected candidate with probability at most `8 / 2^32`, approximately `1.86e-9`. Content alignment, synchronization, tile coverage, and unique-candidate requirements further constrain acceptance.

No Hamming-distance or nearest-codeword attribution fallback is allowed in v4.

## 4. Spatial Layout

### 4.1 Tile And Phase

- Tile size: `128x128` pixels in registered-image coordinates.
- Grid: `8x8` cells per tile.
- Cell size: `16x16` pixels.
- Physical payload: one codeword bit per cell, 64 bits per tile.
- Every valid tile carries the complete RS(8,4) codeword.

Four deterministic carrier permutations reduce fixed-cell bias:

```python
phase = (tile_x + 2 * tile_y) % 4
```

Each phase is a bijection over physical cell indices `0..63`, derived from a versioned SHA-256 seed. The inverse permutation maps decoded cell scores back to logical codeword bits.

Detection requires at least two sufficiently covered tiles and at least two distinct phases. Every accepted tile must have at least 70% valid aligned coverage.

### 4.2 DCT Differential Carrier

The v4 data carrier works in the luminance channel of YCrCb. Each `16x16` cell is converted to `float32` and centered around 128. The batch transform accepts real integer, `float32`, or `float64` blocks and converts them to `float64`; a vectorized orthonormal DCT-II basis then transforms all cells in a tile as one NumPy batch. Unit tests require this batch transform and inverse to match `cv2.dct`/`cv2.idct` applied to the same `float64` values within a fixed numerical tolerance; this avoids 64 Python-to-OpenCV calls per tile while preserving OpenCV-equivalent coefficients.

One bit is represented by the average signed difference from two fixed mid-frequency coefficient pairs. V4 uses these zero-based `(row, column)` coordinates in every `16x16` DCT cell:

```text
pair A: (2, 3) versus (3, 2)
pair B: (2, 4) versus (4, 2)
```

The pairs avoid DC and the highest JPEG-fragile frequencies. Embedding minimally adjusts both coefficients in each pair by equal and opposite half-corrections so the signed difference reaches the configured margin:

```text
bit 1: coefficient_a - coefficient_b >= margin
bit 0: coefficient_b - coefficient_a >= margin
```

The initial margin is `6.0` DCT coefficient units. The Phase 0B calibration set is exactly `4.0`, `6.0`, and `8.0`; values outside `2.0..10.0` are rejected. Promotion selects one value that passes the combined gates and records it as part of the codec configuration.

The inverse transform uses `cv2.idct`, clips to the legal luminance range, and converts back to RGB. Promotion may not select a margin that fails PSNR/SSIM gates.

Detection averages the normalized signed differences from pair A and pair B. Tile phase changes the logical-bit-to-cell permutation, not the coefficient coordinates. This preserves the 64-bit physical layout without doubling the number of modified cells.

Embedding and extraction operate on batched NumPy block views. Python loops may iterate tiles, but not individual pixels. Only tiles intersecting the valid aligned region are decoded.

## 5. FFT Pilot Synchronization

V4 embeds a low-amplitude, versioned pilot composed of four symmetric spatial sinusoidal components in the luminance channel. Their normalized `(x, y)` cycles-per-pixel frequency vectors are:

```text
(0.0703125, 0.1093750)
(0.1015625, 0.1562500)
(0.1406250, 0.0859375)
(0.1718750, 0.1250000)
```

The corresponding negative-frequency conjugates are implicit. The initial per-component luminance amplitude is `0.75`; the Phase 0B calibration set is `0.50`, `0.75`, and `1.00`, with a hard accepted range of `0.25..1.25`. Pilot phase offsets are derived from a SHA-256 seed containing the codec name and component index, not from the authentication key.

Detection performs one query-level synchronization pass:

1. Resize only when the maximum analysis side of `1024` pixels is exceeded.
2. Compute a windowed luminance FFT.
3. Detect the symmetric pilot peak constellation relative to local spectral energy.
4. Use log-polar mapping to estimate rotation and scale.
5. Warp the query into the normalized orientation.
6. Use phase correlation with the versioned pilot template to estimate translation and tile-grid offset.
7. Return geometry, confidence, peak support, and elapsed time.

The initial synchronization gate requires at least three of four conjugate peak pairs, a peak-to-local-median-energy ratio of at least `2.5`, and mutually consistent rotation/scale estimates. These values are reported and calibrated against negative tests before the codec is frozen; they are not attribution evidence by themselves.

The pilot has a separate image-quality contribution measured through layer-ablation tests. If the pilot cannot stay within the combined quality gate, v4 does not ship by increasing its strength.

Generation applies the FFT pilot before the DCT codeword carrier. Phase 0B measurement showed that applying the pilot after DCT could invert a low-margin authenticated bit, while applying DCT last preserves the configured carrier margin without materially changing pilot quality or synchronization strength.

## 6. Candidate Selection And Geometry Fallback

### 6.1 Precomputed Record Index

Generation stores a versioned compressed feature index containing:

- registered-image dimensions;
- ORB keypoint coordinates;
- ORB descriptors;
- thumbnail/global descriptor used for coarse ranking;
- index schema version and OpenCV version.

Detection computes query grayscale, resized analysis image, ORB keypoints/descriptors, and FFT data once. It never repeats query feature extraction per candidate.

Coarse ranking reduces the normal candidate set to two records. One recent-record reserve may be added, for a hard maximum of three candidates in the online v4 path.

### 6.2 Geometry Order

For each candidate:

1. Apply the query-level FFT synchronization estimate.
2. Validate content correspondence against the precomputed candidate index.
3. If FFT synchronization is insufficient but ORB correspondence is strong, compute one ORB/RANSAC homography using the already computed query features and stored candidate features.
4. Reject singular, non-finite, low-inlier, low-ratio, or implausible transforms.

ORB and visual similarity can establish geometry and candidate order only. They can never return a trace without v4 RS/HMAC confirmation.

## 7. Decoder And Bounded Recovery

For each aligned candidate, the decoder:

1. Enumerates covered registered-coordinate tiles.
2. Rejects tiles below 70% valid coverage.
3. Extracts 64 physical signed DCT scores per tile.
4. Applies the inverse phase permutation.
5. Normalizes the tile score vector by robust absolute energy.
6. Aggregates logical-bit scores across tiles.
7. Converts every eight bit scores to one observed byte and one byte confidence.
8. Attempts bounded RS decoding.

RS attempts are fixed and finite:

```text
hard decision, no erasures
lowest-confidence 1 byte as erasure
lowest-confidence 2 bytes as erasures
lowest-confidence 3 bytes as erasures
lowest-confidence 4 bytes as erasures
```

The decoder accepts only combinations allowed by `2 * errors + erasures <= 4` and only when the decoded payload equals the candidate tag exactly.

There is no candidate-codeword distance fallback, short-code fallback, residual-only attribution, visual-only attribution, or legacy dense scan.

Exactly one candidate must pass. Zero or multiple passing candidates return no attribution.

## 8. Fallback Ladder

The commercial detection order is:

```text
1. Exact watermarked-file or decoded-pixel fingerprint
2. FFT synchronization + DCT/RS/HMAC confirmation
3. Precomputed ORB/RANSAC geometry + DCT/RS/HMAC confirmation
4. Bounded synchronization hypotheses + DCT/RS/HMAC confirmation
5. Online budget exhausted -> not detected
```

An exact match to a registered original is explicitly rejected as unwatermarked. Fingerprint attribution requires an exact stored watermarked fingerprint.

Bounded synchronization hypotheses use a fixed configuration list for scale, rotation, and phase. They stop when the time, hypothesis, tile, or candidate budget is reached. An offline deep-detection command may later use a larger explicit budget, but it is not invoked automatically from the online API.

## 9. Performance Design

The online algorithm target is P95 at or below ten seconds and a hard per-case limit of sixteen seconds. The timer starts after the server has received the image bytes and includes decode, query features, candidate retrieval, synchronization, alignment, DCT/RS/HMAC confirmation, persistence, and response construction. Network upload time and future external queue wait are reported separately.

Budget allocation:

| Stage | Target |
| --- | ---: |
| Input decode and validation | 300 ms |
| Query features and candidate ranking | 600 ms |
| FFT synchronization | 1000 ms |
| DCT extraction and RS confirmation | 1600 ms |
| ORB fallback for at most three candidates | 5000 ms |
| Persistence and response | 1000 ms |

Optimization requirements:

- Query ORB, grayscale, FFT, and DCT inputs are computed once.
- Record ORB keypoints/descriptors are precomputed at generation.
- A bounded LRU cache stores decoded record indexes and grayscale thumbnails.
- Cache entries are invalidated when a record is deleted or replaced.
- OpenCV internal threads default to `cv2.setNumThreads(1)` in worker processes to avoid oversubscription.
- CPU-heavy detection runs in bounded worker execution rather than unbounded FastAPI event-loop work.
- Every expensive loop checks the monotonic deadline.
- Diagnostics record elapsed time by stage without exposing secret values.

GPU/CUDA and OpenCL are not required for the first commercial implementation.

## 10. Module Boundaries

```text
watermark_v4/
  __init__.py   public embed/detect interfaces and result types
  config.py     immutable versioned parameters and compute budgets
  payload.py    HMAC32, RS(8,4), permutations, and bounded decode
  dct.py        batched DCT carrier embed/extract
  sync.py       FFT pilot embed and geometry recovery
  features.py   query features, persisted record index, and bounded cache
  embed.py      full v4 image embedding pipeline
  detect.py     candidate, alignment, tile aggregation, and unique attribution
```

`main.py` contains only narrow API integration and persistence calls. V4 modules do not read global application files or databases directly.

## 11. Error Semantics

Internal outcomes are typed as:

- `invalid_input`;
- `not_detected`;
- `budget_exceeded`;
- `system_error`.

Expected decode failures return structured non-match results rather than raising. Configuration, missing dependency, corrupt stored index, and unexpected OpenCV failures raise controlled internal errors and are logged without image content or authentication material.

External API responses preserve the Phase 0A error contract and do not expose candidate lists, tags, thresholds, raw scores, or keys.

## 12. Test Strategy

### 12.1 Payload Tests

- HMAC domain separation, key validation, and deterministic four-byte tag.
- RS(8,4) deterministic encoding and exact eight-byte output.
- Correction of up to two unknown erroneous bytes.
- Mixed errors/erasures only when `2e+s <= 4`.
- Wrong candidate, ambiguous candidate, and invalid payload rejection.
- Four phase permutations are distinct bijections with valid inverses.
- The eight-candidate random-match bound remains below `1e-8`.

### 12.2 DCT Tests

- One cell embeds and recovers both bit values.
- Every phase uses the specified coefficient pairs and complete 64-bit mapping.
- Batched embed/extract is deterministic for a fixed input.
- Constant, low-texture, high-texture, small, grayscale, RGB, and alpha inputs are controlled.
- Embedding does not resize the source image.
- Layer-only quality and clipping metrics are recorded.

### 12.3 Synchronization Tests

- Pilot embedding and detection on a synthetic neutral image.
- Rotation, scale, translation, crop, JPEG, and combined deterministic transforms.
- Unwatermarked, blank, noise, and natural spectral peaks reject.
- Low-confidence and ambiguous pilot constellations reject.
- Deadline and hypothesis limits stop work deterministically.

### 12.4 Feature And Performance Tests

- Query ORB extraction occurs once regardless of candidate count.
- Record alignment uses persisted keypoints/descriptors without reopening the source image.
- Candidate count never exceeds three.
- LRU eviction and record invalidation work.
- Singular and implausible homographies reject.
- OpenCV thread configuration is explicit.

### 12.5 Integrated Quick Gate

Before API promotion, use the five current source images and multiple random trace IDs:

- intact correct attribution `5/5` per trace round;
- intact originals `0` false positives;
- wrong trace `0`;
- minimum PSNR `>= 38 dB`;
- minimum SSIM `>= 0.95`;
- 60-case crop/scale recall must exceed the current v3 baseline;
- P95 detection latency `<= 10s`;
- no case exceeds the sixteen-second hard budget.

Failure keeps v4 isolated from the production API. Parameter tuning may not weaken authentication, unique-candidate, quality, or time-budget rules.

## 13. Implementation Order

1. Implement `config.py` and `payload.py` with exhaustive unit tests.
2. Implement DCT carrier tests and code.
3. Implement FFT pilot synchronization tests and code.
4. Implement persisted feature index and query-once path.
5. Implement candidate-specific decoder and unique-attribution flow.
6. Build the quick benchmark and optimize within fixed safety/quality constraints.
7. Integrate the v4 API only after the quick gate passes.
8. Collect real platform samples and run the 100/300 negative gates before commercial qualification.

## 14. Non-Goals

- Legacy v1/v2/v3 detection compatibility.
- Neural-network watermark training.
- GPU-specific deployment.
- Visual, residual, or short-code attribution.
- Automatic offline deep detection from an online request.
- Production authentication/RBAC deployment work, which remains the later commercial-security phase.

## 15. Completion Definition

Phase 0B completes when the isolated v4 modules, unit/integration tests, quick benchmark, diagnostics, and performance bounds are implemented and the quick gate produces retained evidence. Commercial algorithm qualification still requires real-route samples and the full 300-negative release gate.
