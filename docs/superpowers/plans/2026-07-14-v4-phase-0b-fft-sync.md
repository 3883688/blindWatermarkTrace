# V4 Phase 0B FFT Pilot Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow strict TDD.

**Goal:** Embed the approved four-component FFT pilot and recover bounded rotation, scale, and `128px` tile-grid offset without using visual similarity as attribution evidence.

**Architecture:** `watermark_v4/sync.py` is query-level and candidate-independent. It embeds deterministic luminance sinusoids, scores transformed conjugate peak constellations, and uses pilot coefficient phases to estimate tile offset modulo the tile size. It returns geometry evidence only; it never returns a trace.

**Tech Stack:** NumPy, OpenCV headless, Pillow, pytest

---

## Task 1: Pilot Generation And Embedding

- [x] Create `watermark_v4/sync.py` and `tests/test_watermark_v4_sync.py` with missing-module RED.
- [x] Implement cached deterministic pilot phase offsets from SHA-256 of codec/component index.
- [x] Implement vectorized `pilot_signal(height,width,config)` with the four approved frequencies/amplitude; exact finite float64 output and strict dimension/config validation.
- [x] Implement `embed_pilot(image,config)` for RGB/RGBA, one RGB→YCrCb and one reverse conversion, alpha/dimensions/input preserved, unsupported modes rejected.
- [x] Add neutral/gradient/noise tests, deterministic output, no clipping exceptions, and pilot-only quality checks.

## Task 2: Spectral Peak Scoring

- [x] Define frozen `PilotPeakEvidence` and `SyncEstimate` result types.
- [x] Implement windowed centered FFT magnitude/complex spectrum with maximum analysis side `1024` and monotonic deadline checks.
- [x] Implement transformed frequency prediction for bounded rotation/scale hypotheses and symmetric 3x3 peak sampling.
- [x] Score peak/local-median ratios without exposing raw image content; require at least three of four conjugate pairs and ratio `>=2.5`.
- [x] Test intact pilot acceptance and original/blank/noise/natural-image rejection across current fixtures.

## Task 3: Rotation And Scale Search

- [x] Use a fixed coarse-to-fine hypothesis grid: scale `0.50..2.00`, rotation `-12..12 degrees`; refine around the best coarse result.
- [x] Keep hypothesis count configuration-bounded and stop at deadline.
- [x] Return normalized source-to-query rotation/scale plus confidence, supported peaks, evaluated hypothesis count, and elapsed time.
- [x] Add deterministic resize, rotation, crop, JPEG, and combined-transform tests; ambiguous ties and insufficient support reject.

## Task 4: Tile Offset From Pilot Phase

- [x] For the winning geometry, sample complex pilot coefficients and compare with a generated transformed-pilot reference.
- [x] Search integer offsets modulo `128` using vectorized circular phase residual; no per-pixel Python loop.
- [x] Return `offset_x/offset_y` in `0..127` only when phase residual and peak support pass fixed gates.
- [x] Test known translations/crop origins, modulo equivalence, phase corruption rejection, and deadline behavior.

## Task 5: Synchronization Quality And Performance Gate

- [x] Combine FFT pilot then DCT codeword on representative images and require total minimum PSNR `>=38`, SSIM `>=0.95`. This measured order preserves the final authenticated DCT margin; applying the pilot last inverted a carrier bit at the default margin.
- [x] Require intact, scale, rotation, crop, JPEG, and deterministic combined cases to recover geometry within declared tolerances.
- [x] Require unwatermarked current images and synthetic negatives to return no estimate.
- [x] Bound `1024x1024` synchronization below the `1.0s` stage target on the current machine with a non-SLA regression ceiling.
- [x] Export only stable `SyncEstimate`, `embed_pilot`, and `detect_pilot` APIs.
- [x] Run focused tests with warnings as errors, package compilation, then the complete project suite.

## Exit Gate

The FFT batch completes only when pilot quality, positive geometry recovery, negative rejection, deadline behavior, and performance tests pass without weakening the DCT/RS/HMAC attribution conditions. ORB fallback, candidate records, and API integration remain out of scope.
