# V4 Phase 0B DCT Carrier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow checkbox steps in order.

**Goal:** Embed and recover the complete 64-bit v4 RS codeword in every eligible `128x128` tile using a vectorized, OpenCV-equivalent DCT differential carrier.

**Architecture:** `watermark_v4/dct.py` is a pure image-domain module. It consumes an already encoded eight-byte codeword and `V4Config`; it does not know trace IDs, keys, records, candidates, databases, or synchronization. It batches all 64 `16x16` cells in a tile through an orthonormal DCT-II basis verified against OpenCV.

**Tech Stack:** NumPy, OpenCV headless, Pillow, pytest

---

## Task 1: Vectorized DCT Transform

**Files:**
- Create: `watermark_v4/dct.py`
- Create: `tests/test_watermark_v4_dct.py`

- [x] Write failing tests for `_dct_basis(16)`, `_forward_dct_blocks`, and `_inverse_dct_blocks` using deterministic random blocks.
- [x] Require vectorized output shape `(N,16,16)`, `float32`/`float64` finite values, input immutability, and rejection of malformed shapes/types.
- [x] Compare every batch block against `cv2.dct` and `cv2.idct` with maximum absolute error `<= 1e-4`.
- [x] Run focused tests and observe RED because `watermark_v4.dct` is missing.
- [x] Implement a cached orthonormal DCT-II matrix and broadcast matrix multiplication over the batch; do not loop over cells or pixels.
- [x] Run focused tests and verify GREEN.

## Task 2: Differential Cell Carrier

**Files:**
- Modify: `watermark_v4/dct.py`
- Modify: `tests/test_watermark_v4_dct.py`

- [x] Add failing tests for embedding all-zero/all-one/alternating 64-bit physical vectors into one luminance tile.
- [x] Assert both approved coefficient pairs reach the requested signed margin in coefficient space.
- [x] Assert extraction returns 64 finite signed scores in physical-cell order and recovers every embedded bit before image quantization.
- [x] Cover constant, gradient, deterministic noise, clipped dark, and clipped bright tiles.
- [x] Implement `embed_tile_bits(luminance_tile, bits, config)` and `extract_tile_scores(luminance_tile, config)` using `8x8` block views, equal/opposite half-corrections, inverse batch DCT, and no input mutation.
- [x] Reject non-`128x128` tiles, non-64-bit tuples, non-binary/bool-confused values, nonfinite arrays, and write-protected input assumptions cleanly.
- [x] Run focused RED/GREEN.

## Task 3: Image-Level Full-Tile Embedding

**Files:**
- Modify: `watermark_v4/dct.py`
- Modify: `watermark_v4/__init__.py`
- Modify: `tests/test_watermark_v4_dct.py`

- [x] Add failing tests for `embed_codeword(image, codeword, config)` and `extract_image_tiles(image, config)`.
- [x] Require at least two complete tiles and two distinct phases; reject smaller images before mutation.
- [x] Convert RGB to YCrCb once, batch eligible tiles, apply `phase_for_tile` and `permute_codeword_bits`, then convert back once.
- [x] Preserve image dimensions; preserve alpha bytes for RGBA; define RGB output for RGB and RGBA output for RGBA; reject unsupported modes rather than silently changing them.
- [x] Ignore incomplete right/bottom edge regions and verify they remain pixel-identical except for unavoidable whole-image color conversion. To avoid edge drift, replace only modified tile RGB regions in the original array.
- [x] `extract_image_tiles` returns immutable records containing tile coordinates, phase, and 64 logical signed scores after inverse permutation.
- [x] Export only stable `embed_codeword`, `extract_image_tiles`, and tile-score result type from the package.
- [x] Run focused RED/GREEN with warnings as errors.

## Task 4: DCT Quality And Performance Gate

**Files:**
- Modify: `tests/test_watermark_v4_dct.py`

- [x] Add deterministic PSNR/SSIM-compatible damage assertions for DCT-only embedding at margins `4.0`, `6.0`, and `8.0` on representative current `img/` samples.
- [x] Require default margin `6.0` DCT-only minimum PSNR `>= 42 dB`; record SSIM through the existing quality helper and require `>= 0.98` for this layer alone.
- [x] Require quantized-image soft aggregation to recover the exact codeword on every intact image at default margin. At least 90% of complete tiles must also recover the full codeword individually; remaining tile errors are diagnostic evidence handled by aggregation and RS rather than repeated color-conversion passes.
- [x] Add a bounded performance test over a synthetic `1024x1024` image: query extraction computes batched DCT and completes under a generous regression ceiling derived from the current machine; record timing without using it as the commercial SLA.
- [x] Verify no OpenCV per-cell loop by monkeypatching `cv2.dct`/`cv2.idct` to fail during production batch operations; reference-equivalence tests may call them separately.

## Task 5: DCT Batch Exit Gate

**Files:**
- Update plan checkboxes/evidence only after verification.

- [x] Run `python -m py_compile watermark_v4/dct.py watermark_v4/__init__.py`.
- [x] Run `python -m pytest tests/test_watermark_v4_config.py tests/test_watermark_v4_payload.py tests/test_watermark_v4_dct.py -q -W error`.
- [x] Run `python -m pytest -q` and record exact pass/skip/warning counts.
- [x] Verify `main.py`, legacy modules, production data/uploads, benchmark gates, and protected V3 archive were not modified.

## Exit Gate

The DCT batch completes only when vectorized transforms match OpenCV, intact quantized images recover the exact aggregate codeword with at least 90% exact individual-tile recovery, layer-only quality gates pass, alpha/dimensions are preserved, malformed inputs reject, performance remains bounded, and the full existing suite stays green. FFT synchronization and API integration remain out of scope.

## Verification Evidence

- Focused DCT suite with warnings as errors: `122 passed`.
- Full V4 config/payload/DCT verification before the final project gate: `224 passed`.
- Final complete project suite: `707 passed, 2 skipped, 184 warnings in 164.51s`.
- Default margin `6.0`: minimum measured PSNR `44.344482 dB`, minimum SSIM `0.988879`, exact aggregate recovery on all three representative images, and minimum individual exact-tile recovery `92.00%`.
- Synthetic `1024x1024` extraction: 64 tiles in `0.017625s` in the recorded focused run.
- Production DCT paths do not call scalar `cv2.dct` or `cv2.idct` per cell.
- `main.py`, legacy watermark modules, production data/uploads, benchmark gates, and the protected V3 archive were not modified by this DCT batch.

