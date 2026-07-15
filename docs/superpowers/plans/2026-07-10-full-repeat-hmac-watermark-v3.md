# Full-Repeat HMAC Watermark V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and test-driven development. Steps use checkbox syntax for tracking.

**Goal:** Implement, benchmark, and safely gate a full-repeat HMAC64 robust watermark v3.

**Architecture:** Add HMAC and carrier-permutation helpers in a focused module, keep image embedding/alignment in `main.py`, dispatch by record version, and make benchmarks explicitly generate v3. Keep production default v1 unless every quick gate passes.

**Tech Stack:** Python, standard-library `hmac/hashlib`, NumPy, OpenCV, Pillow, FastAPI, pytest.

**Repository note:** The workspace is not a Git repository; commit and worktree steps are omitted.

---

### Task 1: HMAC Code And Permutations

**Files:** `watermark_auth.py`, `tests/test_watermark_auth.py`

- [ ] Write failing tests for 32-byte key validation, deterministic 8-byte HMAC output, different-key separation, three permutation bijections, inverse mappings, and the eight-error random acceptance bound.
- [ ] Run `python -m pytest tests/test_watermark_auth.py -q` and verify RED because the module does not exist.
- [ ] Implement `auth_code_from_trace`, `phase_permutation`, `inverse_permutation`, and `candidate_radius_probability` using only standard-library cryptography plus NumPy-free tuple mappings.
- [ ] Run the focused tests and verify GREEN.

### Task 2: Full-Repeat V3 Embedding

**Files:** `main.py`, `tests/test_aligned_authenticated_detection.py`

- [ ] Write a failing test that compares v1/v3 changed-pixel ratio and maximum delta at equal strength.
- [ ] Write a failing test that verifies every v3 tile contains all 64 logical bits under its phase permutation.
- [ ] Implement `embed_robust_watermark_v3(image, auth_code, strength_scale)` while leaving v1/v2 writers unchanged.
- [ ] Verify focused embedding tests pass.

### Task 3: Aligned V3 Decoder

**Files:** `main.py`, `tests/test_aligned_authenticated_detection.py`

- [ ] Write failing identity, scaled-crop, wrong-candidate, unwatermarked, missing-code, one-phase, and ambiguous-candidate tests.
- [ ] Implement per-tile normalization, inverse-permutation soft aggregation, maximum eight hard errors, positive signed agreement, and diagnostics.
- [ ] Dispatch versions 1/2/3 without cross-version fuzzy fallback.
- [ ] Verify all aligned and false-positive tests pass.

### Task 4: API And Benchmark Versioning

**Files:** `main.py`, `.env.example`, `tests/commercial_benchmark_config.py`, commercial benchmark scripts, `run_commercial_benchmark.ps1`, related tests.

- [ ] Write failing tests for v3 API metadata, missing/short key rejection, and benchmark form propagation.
- [ ] Add `WATERMARK_AUTH_KEY`, version 3 metadata, stored auth code, and v3 benchmark settings.
- [ ] Set benchmark-only v3 key and version explicitly; keep `.env.example` production version at `1`.
- [ ] Add v3 diagnostics to JSON/CSV output and verify PowerShell syntax.

### Task 5: Verification And Quick Gates

- [ ] Run focused tests, full `python -m pytest -q`, compilation, and PowerShell parsing.
- [ ] Run v3 quality at fidelity `1.0`, small crop `medium/0.35`, robust strength `0.74`.
- [ ] Only if quality passes, run the established 60-case matrix with 20 workers.
- [ ] Promote the default to v3 only if recall, wrong trace, false-positive, quality, and latency gates all pass; otherwise keep v1.
- [ ] Update the commercial upgrade plan and balanced assessment with actual metrics and the promotion decision.
- [ ] Re-run full verification and stop before long tests.
