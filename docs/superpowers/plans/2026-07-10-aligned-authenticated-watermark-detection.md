# Aligned Authenticated Watermark Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the existing authenticated 48-bit watermark from geometrically transformed images without increasing image damage or allowing short-code, visual, or residual-only attribution.

**Architecture:** Use ORB/RANSAC to align a query image into each registered watermarked image coordinate system. Decode only known embedded tile locations with sufficient valid coverage, aggregate full-code evidence, and return a trace only after magic, CRC, unique-record, and multi-tile checks pass. Keep dense legacy scans behind an explicit offline flag.

**Tech Stack:** Python, FastAPI, Pillow, NumPy, OpenCV, pytest.

**Repository note:** The workspace is not a Git repository, so commit and worktree steps are omitted.

---

### Task 1: Compute Safe Query-To-Record Alignment

**Files:**
- Modify: `main.py:2303-2334`
- Create: `tests/test_aligned_authenticated_detection.py`

- [ ] Write tests using a synthetic textured image transformed by scale, crop, and small rotation. Assert `align_query_to_record` returns a target-sized RGB image, target-sized mask, at least 18 inliers, ratio at least 0.32, and valid coverage between 5% and 100%.
- [ ] Add tests for a blank unrelated image and singular/insufficient-feature cases; assert alignment returns `None`.
- [ ] Run `python -m pytest tests\test_aligned_authenticated_detection.py -q` and verify RED because the helper does not exist.
- [ ] Implement `align_query_to_record(image, record)` using stored `download_url`, bounded image resizing, `feature_match_homography`, matrix inversion, `cv2.warpPerspective`, and projected mask validation.
- [ ] Run the focused tests and verify GREEN.

### Task 2: Enumerate Known Covered Tiles

**Files:**
- Modify: `main.py:1018-1056`
- Test: `tests/test_aligned_authenticated_detection.py`

- [ ] Write tests that construct aligned arrays and masks for `low`, `medium`, and `high` density records. Assert only tile positions written by `small_crop_density_offsets` are returned and tiles below 70% mask coverage are excluded.
- [ ] Verify RED because `iter_aligned_small_trace_tiles` does not exist.
- [ ] Implement `iter_aligned_small_trace_tiles(aligned, valid_mask, record)` using the record density and the target scale metadata returned by alignment. Yield normalized 96x96 RGB tiles plus source position and coverage.
- [ ] Verify all tile enumeration tests pass.

### Task 3: Decode Full Authenticated Payload Evidence

**Files:**
- Modify: `main.py:1341-1411`
- Test: `tests/test_aligned_authenticated_detection.py`

- [ ] Write tests that embed a known trace into multiple synthetic tiles using `apply_small_crop_trace_layer`, then assert `decode_authenticated_aligned_trace` recovers the exact record from at least two spatially separate tiles.
- [ ] Add negative tests for short-code-only synthetic carriers, one valid tile, wrong CRC, and an unrelated unwatermarked image; all must return `None`.
- [ ] Verify RED because the decoder does not exist.
- [ ] Implement per-tile full score decoding with `decode_small_trace_code_scores`, `code_from_score_vector`, `recover_payload_from_code`, magic distance, CRC distance, expected payload distance, marker score, trace score, and spatial separation checks.
- [ ] Do not use `record_from_short_code_match` in the authenticated decoder.
- [ ] Verify focused decoder tests pass.

### Task 4: Integrate A Budgeted Online Detector

**Files:**
- Modify: `main.py:2949-2994`
- Modify: `.env.example`
- Test: `tests/test_aligned_authenticated_detection.py`
- Test: `tests/test_false_positive_gate.py`

- [ ] Write an integration test that seeds one record, transforms the watermarked image, and asserts either the aligned detector returns the exact trace or safely returns `None`; it must never return another record.
- [ ] Write a deterministic positive case using a transform proven by Task 3 and require the exact trace.
- [ ] Verify RED because `detect_aligned_authenticated_watermark` is not connected.
- [ ] Implement candidate limit and deadline settings: `ALIGNED_CANDIDATE_LIMIT=8`, `WATERMARK_DETECTION_BUDGET_SECONDS=5`, and `ENABLE_DENSE_WATERMARK_FALLBACK=false`.
- [ ] Insert aligned authenticated detection after exact LSB/block extraction and before dense fallbacks. When the deadline expires return no result.
- [ ] Keep dense `detect_small_crop_trace` and `detect_watermark_code` available only when the explicit offline flag is enabled.
- [ ] Run integration and existing safety tests.

### Task 5: Benchmark And Document The Result

**Files:**
- Modify: `docs/commercial_watermark_upgrade_plan.md`
- Generated: `test_output/commercial_trace_benchmark/*`
- Generated: `test_output/commercial_balanced_assessment.md`

- [ ] Run `python -m pytest -q` and require all tests to pass.
- [ ] Run the 2-scale x 3-crop positive and negative quick matrix with 20 workers and candidate configuration `fidelity=1.0`, `medium/0.35`.
- [ ] Compare correct recall against 46.67%, wrong trace against 0, false positives against 0, and P95 detection latency against 5 seconds.
- [ ] If recall does not improve without weakening authentication, keep the aligned detector disabled by default and document that the existing embedding format requires a new ECC watermark version.
- [ ] If recall improves and all safety gates remain zero, enable the aligned detector by default but do not run the 1,000-negative long test until the user approves the measured estimate.
- [ ] Update the commercial report with actual metrics and limitations.
