# Commercial Watermark Quality And Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable commercial benchmark that selects the least visually damaging watermark configuration meeting strict trace recall and zero-attribution-error gates, then harden the current residual fallback against false positives.

**Architecture:** Keep production watermark code in `main.py`, add focused benchmark utilities under `tests/`, and orchestrate existing benchmark stages from one PowerShell entry point. Quality scanning first narrows the fidelity configurations; only Pareto candidates enter expensive crop, attack, stability, and negative-sample runs. Residual and visual similarity remain candidate evidence and cannot independently attribute a trace.

**Tech Stack:** Python 3, FastAPI TestClient, Pillow, NumPy, OpenCV, PyWavelets, pytest, PowerShell.

**Repository note:** `D:\WWW\python\trace` is not a Git repository. Commit steps are intentionally omitted; do not initialize Git or rewrite existing files outside this plan without user approval.

---

## File Map

- Create `tests/commercial_quality_metrics.py`: pure image-quality metric and percentile helpers.
- Create `tests/test_commercial_quality_metrics.py`: fast unit tests for metric correctness and threshold decisions.
- Create `tests/commercial_quality_benchmark.py`: fidelity scan, representative trace probes, Pareto selection, JSON/CSV/Markdown outputs.
- Create `tests/test_residual_attribution_gate.py`: deterministic regression for synthetic residual false positives and positive attacked-watermark recovery.
- Modify `main.py`: expose residual evidence and prevent residual-only trace attribution.
- Modify `tests/commercial_trace_benchmark.py`: configurable fidelity, expanded scale/crop matrix, per-case timing, threshold verdict.
- Modify `tests/commercial_attack_benchmark.py`: configurable fidelity, per-round metrics, elapsed time, gate verdict.
- Modify `tests/commercial_negative_benchmark.py`: 1,000+ diverse and source-similar negatives, method evidence, threshold verdict.
- Create `run_commercial_benchmark.ps1`: stage orchestration, 20-worker default, reuse/skip controls, fail-fast gate handling.
- Modify `docs/commercial_watermark_upgrade_plan.md`: actual before/after results, selected configuration, limitations.

### Task 1: Add Deterministic Image Quality Metrics

**Files:**
- Create: `tests/commercial_quality_metrics.py`
- Create: `tests/test_commercial_quality_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create tests that assert identical images produce infinite PSNR, SSIM 1.0, and zero error; a one-channel perturbation produces finite metrics; and any image under either hard threshold fails.

```python
import numpy as np
from PIL import Image

from tests.commercial_quality_metrics import quality_gate, quality_metrics


def test_identical_images_have_perfect_quality():
    image = Image.new("RGB", (64, 64), (100, 120, 140))
    metrics = quality_metrics(image, image)
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0


def test_quality_gate_requires_both_psnr_and_ssim():
    assert quality_gate({"psnr": 40.0, "ssim": 0.99}) is True
    assert quality_gate({"psnr": 37.99, "ssim": 0.99}) is False
    assert quality_gate({"psnr": 40.0, "ssim": 0.9799}) is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_commercial_quality_metrics.py -q`

Expected: collection fails because `tests.commercial_quality_metrics` does not exist.

- [ ] **Step 3: Implement pure metric helpers**

Implement `quality_metrics(original, watermarked)` with RGB shape validation, NumPy error metrics, and OpenCV SSIM using an 11x11 Gaussian window. Implement `quality_gate(metrics, min_psnr=38.0, min_ssim=0.98)` and `metric_distribution(rows, field)` returning `min`, `p5`, `p50`, `p95`, and `mean`.

```python
def quality_gate(metrics: dict[str, float], min_psnr: float = 38.0, min_ssim: float = 0.98) -> bool:
    return metrics["psnr"] >= min_psnr and metrics["ssim"] >= min_ssim


def metric_distribution(rows: list[dict], field: str) -> dict[str, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "min": round(float(values.min()), 6),
        "p5": round(float(np.percentile(values, 5)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "mean": round(float(values.mean()), 6),
    }
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_commercial_quality_metrics.py -q`

Expected: all tests pass.

### Task 2: Build The Fidelity Quality Scanner

**Files:**
- Create: `tests/commercial_quality_benchmark.py`
- Test: `tests/test_commercial_quality_metrics.py`

- [ ] **Step 1: Add failing Pareto-selection tests**

Add tests for `select_recommended_config(configs)`: reject any config with false positives, wrong traces, failed quality, or recall below the probe gate; among valid rows select the highest minimum SSIM then highest minimum PSNR.

```python
def test_selector_prefers_least_damage_among_configs_that_pass_trace_gates():
    configs = [
        {"fidelity": 0.75, "quality_pass": True, "wrong": 0, "false_positive": 0, "probe_recall": 1.0, "min_ssim": 0.985, "min_psnr": 39.0},
        {"fidelity": 0.90, "quality_pass": True, "wrong": 0, "false_positive": 0, "probe_recall": 0.90, "min_ssim": 0.993, "min_psnr": 42.0},
        {"fidelity": 0.85, "quality_pass": True, "wrong": 0, "false_positive": 0, "probe_recall": 1.0, "min_ssim": 0.990, "min_psnr": 41.0},
    ]
    assert select_recommended_config(configs)["fidelity"] == 0.85
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_commercial_quality_metrics.py -q`

Expected: import or assertion failure because the selector is not implemented.

- [ ] **Step 3: Implement scanner configuration and isolated state**

Use environment variables with explicit defaults:

```python
FIDELITY_LEVELS = [float(item) for item in os.getenv("FIDELITY_LEVELS", "0.70,0.75,0.80,0.85,0.90").split(",")]
QUALITY_MIN_PSNR = float(os.getenv("QUALITY_MIN_PSNR", "38.0"))
QUALITY_MIN_SSIM = float(os.getenv("QUALITY_MIN_SSIM", "0.98"))
PROBE_MIN_RECALL = float(os.getenv("PROBE_MIN_RECALL", "0.95"))
```

For each fidelity, reset `test_output/quality_benchmark/runtime`, embed the five `img/*.png` sources with copyright and dot matrix disabled, compute quality metrics, and save watermarked files under `test_output/commercial_quality_benchmark/fidelity-<value>/`.

- [ ] **Step 4: Add representative trace probes**

For each generated image run: intact PNG, 0.5 scale with 50% crop, 1.5 scale with 30% crop, JPEG 30, WeChat simulation, and screen-photo simulation. Run the same probes on the five originals and count every HTTP 200 as a false positive.

- [ ] **Step 5: Write machine-readable and Markdown outputs**

Write `commercial_quality_results.json`, `commercial_quality_results.csv`, and `commercial_quality_test_report.md`. Include per-image metrics, distributions, trace probes, rejected reasons, Pareto rows, selected fidelity, elapsed time, and `PASS/FAIL`.

- [ ] **Step 6: Verify scanner in a reduced matrix**

Run: `$env:FIDELITY_LEVELS='0.75,0.85'; python tests\commercial_quality_benchmark.py`

Expected: exit 0, three output files exist, and JSON contains two configurations plus `recommended_fidelity` or an explicit no-candidate failure reason.

### Task 3: Make Crop Benchmark Configurable And Commercially Gated

**Files:**
- Modify: `tests/commercial_trace_benchmark.py`
- Test: `tests/test_commercial_quality_metrics.py`

- [ ] **Step 1: Add parsing and verdict tests**

Test comma-separated float parsing and a `crop_verdict(summary)` function. The verdict must fail on any wrong trace or false positive, overall recall below 95%, 30% crop below 80%, or 50%/80% crop below 95%.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_commercial_quality_metrics.py -q`

Expected: failure because crop verdict helpers do not exist.

- [ ] **Step 3: Replace fixed constants with environment-backed values**

```python
SCALE_FACTORS = parse_float_list("SCALE_FACTORS", "0.5,0.75,1.0,1.25,1.5,2.0")
CROP_RATIOS = parse_float_list("CROP_RATIOS", "0.3,0.5,0.8,1.0")
CROPS_PER_RATIO = int(os.getenv("CROPS_PER_RATIO", "3"))
FIDELITY_LEVEL = os.getenv("FIDELITY_LEVEL", "0.75")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260707"))
```

Ensure negative originals use the same full matrix and deterministic crop coordinates derived from `(seed, source, scale, crop, crop_index)`.

- [ ] **Step 4: Record timings and threshold verdict**

Add `detection_ms` to every result. Add `verdict`, `failed_gates`, and executed settings to JSON and Markdown. Exit with code 2 for a completed benchmark whose commercial gates fail; reserve code 1 for execution errors.

- [ ] **Step 5: Run a reduced crop smoke test**

Run: `$env:SCALE_FACTORS='0.5,1.5'; $env:CROP_RATIOS='0.3,0.8'; $env:CROPS_PER_RATIO='1'; python tests\commercial_trace_benchmark.py`

Expected: outputs are written even when the gate verdict is `FAIL`; no execution exception occurs.

### Task 4: Reproduce And Block Residual-Only Attribution

**Files:**
- Create: `tests/test_residual_attribution_gate.py`
- Modify: `main.py:2199-2270`
- Test: `tests/test_false_positive_gate.py`

- [ ] **Step 1: Write a deterministic failing regression**

Seed five watermark records, generate the known synthetic indices `76`, `82`, `94`, and `172` using `synthetic_image`, and assert `extract_watermark_from_image` raises HTTP 404 instead of returning `mode == "residual_verified"`.

```python
@pytest.mark.parametrize("index", [76, 82, 94, 172])
def test_synthetic_residual_matches_do_not_attribute_trace(seeded_records, index):
    with pytest.raises(HTTPException) as exc:
        main.extract_watermark_from_image(synthetic_image(index))
    assert exc.value.status_code == 404
```

- [ ] **Step 2: Verify RED against current code**

Run: `pytest tests/test_residual_attribution_gate.py -q`

Expected: at least one case returns a trace, reproducing the previous 1.57% synthetic false-positive path.

- [ ] **Step 3: Convert residual matching to non-attributing evidence**

Change `detect_by_residual_match` so it returns no trace unless a separately decoded watermark code confirms the same record. Add a helper with this contract:

```python
def residual_candidate_evidence(image: Image.Image) -> dict[str, Any] | None:
    """Return best visual/residual candidate metrics without authorizing trace attribution."""
```

The extraction fallback must not return a record from residual evidence alone. Keep evidence available to strengthen `detect_small_crop_trace`, `detect_watermark_code`, or `detect_robust_watermark` only when their decoded candidate trace equals the residual candidate trace.

- [ ] **Step 4: Preserve positive multi-evidence recovery**

Update the existing positive scaled-crop assertions so accepted modes are code-backed modes. If a previous case was recovered only by residual matching, it may become a 404; record this as recall loss for the later parameter scan rather than reintroducing unsafe attribution.

- [ ] **Step 5: Run focused regression tests**

Run: `pytest tests/test_residual_attribution_gate.py tests/test_false_positive_gate.py -q`

Expected: all negative cases return 404, no wrong trace occurs, and positive code-backed cases pass.

### Task 5: Expand Negative Samples To 1,000+

**Files:**
- Modify: `tests/commercial_negative_benchmark.py`
- Test: `tests/test_residual_attribution_gate.py`

- [ ] **Step 1: Add deterministic generator tests**

Assert the generator creates exactly the requested number, names are unique, repeated runs with the same seed have identical pixel hashes, and all six synthetic families plus source-similar variants appear.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_residual_attribution_gate.py -q`

Expected: generator coverage assertions fail before expansion.

- [ ] **Step 3: Expand sample families without storing all source arrays in memory**

Set `SYNTHETIC_VARIANTS` default to `1000`. Extend families to include low texture, gradients, correlated noise, grids, UI blocks, periodic patterns, text-like edges, color-shifted sources, brightness/contrast variants, source crops, and slight source geometry changes. Generate one file per case and pass only paths to workers.

- [ ] **Step 4: Improve false-positive diagnostics**

Add `mode`, `code_recovery`, `layer_scores`, `match_inliers`, `match_ratio`, and `detection_ms` fields. Reports group false positives by category and mode. Gate rules are `original/attacked-source false_positive == 0`, total rate `<0.001`, and wrong trace attribution count `==0`.

- [ ] **Step 5: Run a 120-sample smoke test**

Run: `$env:SYNTHETIC_VARIANTS='120'; $env:BENCHMARK_WORKERS='20'; python tests\commercial_negative_benchmark.py`

Expected after Task 4: zero residual-only attributions; outputs include diagnostic fields and gate verdict.

### Task 6: Add Per-Round Stability Gates

**Files:**
- Modify: `tests/commercial_attack_benchmark.py`

- [ ] **Step 1: Add failing summary tests**

Test that a five-round aggregate fails when one round drops below 95% even if total recall remains at least 95%, and fails on any wrong trace or negative false positive.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_commercial_quality_metrics.py -q`

Expected: failure because per-round gate data is absent.

- [ ] **Step 3: Add configurable fidelity and per-round summaries**

Read `FIDELITY_LEVEL` from the environment in `seed_watermarks`. Extend `summarize` with `by_round`, including positive total, correct, wrong, recall, negative total, and false positives for each round.

- [ ] **Step 4: Record performance and verdict**

Record per-case `detection_ms`, total elapsed seconds, and cases per second. Add `PASS/FAIL` with explicit failed gates to JSON and Markdown. A round fails if recall is below 95%, wrong trace is nonzero, or negative false positives are nonzero.

- [ ] **Step 5: Run one-round representative smoke test**

Run: `powershell -ExecutionPolicy Bypass -File .\run_simple_stability.ps1 -Workers 20 -TraceRounds 1`

Expected: report contains a per-round table and performance statistics.

### Task 7: Add The Unified Balanced Runner

**Files:**
- Create: `run_commercial_benchmark.ps1`

- [ ] **Step 1: Implement validated parameters**

```powershell
param(
    [ValidateSet('quality','crop','attack','negative','balanced')]
    [string]$Stage = 'balanced',
    [ValidateRange(1, 24)][int]$Workers = 20,
    [ValidateRange(1, 20)][int]$TraceRounds = 5,
    [ValidateRange(1, 10000)][int]$NegativeVariants = 1000,
    [switch]$Reuse
)
```

Save and restore every environment variable in `finally`. Treat benchmark exit code 2 as a completed commercial gate failure, print the report path, and stop later stages that rely on a failed candidate selection.

- [ ] **Step 2: Orchestrate quality-first candidate selection**

Run quality scan first. Parse its JSON with `ConvertFrom-Json`; set `FIDELITY_LEVEL` to `recommended_fidelity`. If no candidate exists, exit 2 without running expensive stages.

- [ ] **Step 3: Orchestrate crop, attack, and negative stages**

For `balanced`, run crop with 3 random regions, attack with 5 trace rounds and all 16 attacks, then negative with at least 1,000 variants. Print elapsed time after each stage and the expected next-stage cost before launching it.

- [ ] **Step 4: Add result reuse**

When `-Reuse` is present, reuse a stage only if its JSON records the same fidelity and matrix settings. Otherwise rerun it. Never infer reuse from file existence alone.

- [ ] **Step 5: Run quality-only orchestration smoke test**

Run: `powershell -ExecutionPolicy Bypass -File .\run_commercial_benchmark.ps1 -Stage quality -Workers 20`

Expected: quality report path and recommended fidelity are printed; process exits 0 for a candidate or 2 with an explicit no-candidate reason.

### Task 8: Run Baseline, Optimize, And Publish The Upgrade Report

**Files:**
- Modify: `docs/commercial_watermark_upgrade_plan.md`
- Generated: `test_output/commercial_quality_benchmark/*`
- Generated: `test_output/commercial_trace_benchmark/*`
- Generated: `test_output/commercial_attack_benchmark/*`
- Generated: `test_output/commercial_negative_benchmark/*`

- [ ] **Step 1: Run fast verification before expensive benchmarks**

Run:

```powershell
python -m py_compile main.py tests\commercial_quality_metrics.py tests\commercial_quality_benchmark.py tests\commercial_trace_benchmark.py tests\commercial_attack_benchmark.py tests\commercial_negative_benchmark.py
pytest tests\test_commercial_quality_metrics.py tests\test_residual_attribution_gate.py tests\test_false_positive_gate.py -q
```

Expected: compile succeeds and all focused tests pass.

- [ ] **Step 2: Run the quality scan and crop matrix**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_commercial_benchmark.ps1 -Stage quality -Workers 20
powershell -ExecutionPolicy Bypass -File .\run_commercial_benchmark.ps1 -Stage crop -Workers 20
```

Expected: a recommended fidelity exists, every image meets PSNR >= 38 dB and SSIM >= 0.98, wrong trace is zero, and crop report is written regardless of gate verdict.

- [ ] **Step 3: Report estimated cost and obtain user approval for the long run**

Before starting the full five-round attack and 1,000+ negative stages, report quality/crop results and measured cases per second. Estimate remaining duration from actual timings. Do not start the long run until the user confirms.

- [ ] **Step 4: Run the approved balanced long stages**

Run: `powershell -ExecutionPolicy Bypass -File .\run_commercial_benchmark.ps1 -Stage balanced -Workers 20 -TraceRounds 5 -NegativeVariants 1000 -Reuse`

Expected: all stage reports are written; commercial gate failures use exit code 2 and include explicit failed gates.

- [ ] **Step 5: Update the commercial upgrade document with actual evidence**

Add the selected fidelity, worst-image PSNR/SSIM, Pareto alternatives, crop recall, attack recall by round, negative sample composition, false-positive rate, throughput, failures, and `PASS/CONDITIONAL/FAIL` conclusion. Do not claim full commercial readiness because only five source images are available.

- [ ] **Step 6: Final verification**

Run:

```powershell
pytest -q
python -m py_compile main.py tests\*.py
```

Expected: pytest passes. If PowerShell wildcard expansion is not accepted by `py_compile`, enumerate files with `Get-ChildItem tests -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }` and require every invocation to exit 0.

