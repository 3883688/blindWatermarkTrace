# V4 Candidate Detector And API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Follow strict TDD. This workspace is not a Git repository; use focused and full-suite verification checkpoints instead of commits.

**Goal:** Complete the v4 commercial path from one-time query analysis through at-most-three candidate geometry checks, DCT/RS/HMAC unique attribution, generation persistence, and API detection.

**Architecture:** `watermark_v4/features.py` owns versioned ORB indexes and query-once extraction. `watermark_v4/detector.py` owns bounded candidate ranking, homography validation, registered-tile extraction, soft aggregation, candidate-specific RS/HMAC confirmation, and unique-result enforcement. `main.py` adapts records and HTTP responses but cannot attribute from FFT, ORB, residual, or visual evidence alone.

**Tech Stack:** Python 3.13, NumPy, OpenCV headless, Pillow, FastAPI, pytest

---

### Task 1: Versioned Feature Index And Query-Once Extraction

**Files:**
- Create: `watermark_v4/features.py`
- Create: `tests/test_watermark_v4_features.py`
- Preserve: `candidate_feature_index.py` as the isolated historical index path.

- [x] Write RED tests for a frozen `FeatureIndex` containing schema version, OpenCV version, original dimensions, original-coordinate keypoints, `(n,32)` descriptors, and a bounded thumbnail descriptor.
- [x] Write RED tests proving save/load round trips without pickle, malformed or incompatible indexes reject, and legacy descriptor-only helpers remain controlled.
- [x] Write RED tests proving one query image performs grayscale conversion and ORB extraction exactly once while multiple candidate indexes reuse the result.
- [x] Implement strict bounded extraction (`max_side=640`, `3072` geometry descriptors, first `256` for coarse ranking) and safe compressed persistence.
- [x] Run focused feature tests with warnings as errors and the complete project suite.

### Task 2: Candidate Ranking And ORB/RANSAC Geometry

**Files:**
- Modify: `watermark_v4/features.py`
- Modify: `tests/test_watermark_v4_features.py`

- [x] Write RED tests for ratio-test matching, query-to-record homography in original coordinates, minimum inliers/ratio, finite nonsingular matrices, and implausible transform rejection.
- [x] Write RED tests that normal ranking returns two feature candidates plus at most one recent reserve, never exceeding `V4Config.candidate_limit == 3`.
- [x] Implement deterministic matching and ranking with stable record IDs as tie breakers.
- [x] Verify crop/scale/rotation recovery and unrelated/blank rejection.

### Task 3: Registered Tile Extraction And Candidate Decode

**Files:**
- Create: `watermark_v4/detector.py`
- Create: `tests/test_watermark_v4_detector.py`
- Modify: `watermark_v4/__init__.py`

- [x] Write RED tests that warp query pixels and a validity mask into registered coordinates, accept only tiles with coverage `>=0.70`, and require at least two tiles and two phases.
- [x] Implement batched registered-tile DCT scoring using the existing carrier and phase permutation; do not run scalar `cv2.dct/idct`.
- [x] Write RED tests for robust per-tile normalization, 64-bit aggregation, eight byte confidences, bounded RS erasures, and exact expected HMAC32 payload comparison.
- [x] Reject malformed v4 records, wrong tags, wrong candidates, insufficient coverage, and deadline expiration.

### Task 4: Unique V4 Attribution Orchestrator

**Files:**
- Modify: `watermark_v4/detector.py`
- Modify: `tests/test_watermark_v4_detector.py`

- [x] Write RED tests proving query features and FFT synchronization run once, candidate count is at most three, and per-candidate work stops at the monotonic deadline.
- [x] Implement FFT-first geometry evidence with precomputed ORB/RANSAC fallback; neither geometry source may return a trace.
- [x] Accept exactly one candidate-specific RS/HMAC decode; zero or multiple passing candidates return no attribution.
- [x] Return a frozen result with trace ID, record ID, codec, tile/phase counts, geometry method, confidence evidence, hypothesis counts, and elapsed time, excluding keys and raw image content.

### Task 5: V4 Generation And Persistence API

**Files:**
- Modify: `main.py`
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `index.html` only if the existing version control cannot select v4

- [x] Write RED endpoint tests for explicit v4 selection, missing/short auth key controlled failure, and no fallback to v1/v2/v3.
- [x] Generate `authentication_tag -> RS(8,4) -> FFT pilot -> DCT carrier`, then persist version `4`, exact codec, eight-lowercase-hex auth tag, dimensions, hashes, and v4 feature index.
- [x] Ensure v4 does not stack legacy DWT/FFT/short-code/dot-matrix/residual attribution layers.
- [x] Keep historical versions isolated; commercial v4 detection considers only exact v4 records.

### Task 6: Detection API, Quick Matrix, And Exit Gates

**Files:**
- Modify: `main.py`
- Modify: `tests/test_watermark_v4_api.py`
- Create: `tests/test_watermark_v4_quick_matrix.py`
- Modify: `docs/commercial/phase-0b-v4-results.md`

- [x] Write RED API tests for exact watermarked fingerprint success, exact original rejection, v4 transformed success, wrong-candidate rejection, and controlled 404 on insufficient evidence.
- [x] Run deterministic intact, resize, rotation, crop, JPEG, and combined positives plus unwatermarked natural/synthetic negatives.
- [x] Enforce P95 `<=10s`, hard timeout `16s`, candidate limit `3`, and no visual/residual/short-code attribution fallback.
- [x] Run the 100-negative development gate; retain the 300-negative and real-route collection as commercial promotion gates.
- [x] Run V4 tests with warnings as errors, `compileall`, then `python -m pytest -q`.

## Exit Gate

Phase 0B detector/API work completes only when a transformed image can be attributed by exactly one candidate's DCT/RS/HMAC evidence, every non-authenticated path returns no trace, query analysis is reused, online bounds are enforced, and all quick/full tests pass. The 300-negative and real social-route evidence remain promotion gates, not implementation assumptions.
