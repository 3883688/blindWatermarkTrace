# Commercial Watermark False Positive Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce false positives from unwatermarked images while preserving trace recovery on scaled/cropped watermarked images as much as possible.

**Architecture:** Add a conservative verification gate around weak fallback detections, especially small-crop frequency-code matches. The benchmark remains the acceptance harness and records positive recall separately from unwatermarked false positives.

**Tech Stack:** Python, FastAPI TestClient, Pillow, OpenCV, NumPy, existing `main.py` watermark functions.

---

### Task 1: Reproduce And Attribute False Positives

**Files:**
- Read: `test_output/commercial_trace_benchmark/commercial_trace_results.json`
- Read: `main.py`

- [ ] **Step 1: Inspect false-positive methods**

Run:

```powershell
@'
import json
from collections import Counter
payload = json.load(open("test_output/commercial_trace_benchmark/commercial_trace_results.json", encoding="utf-8"))
fps = [r for r in payload["results"] if r["case_type"] == "unwatermarked" and r["success"]]
print(len(fps))
print(Counter((r["mode_label"], r["status"]) for r in fps))
for r in fps:
    print(r["source"], r["scale_factor"], r["crop_ratio"], r["detected_trace_id"], r["confidence"], r["status"])
'@ | python -
```

Expected: Output identifies the dominant false-positive fallback.

### Task 2: Add A Fast Regression Test

**Files:**
- Create: `tests/test_false_positive_gate.py`

- [ ] **Step 1: Write failing test for unwatermarked attacked crops**

Create a pytest test that embeds the five `img/*.png` samples, then checks selected unwatermarked scaled/cropped samples return 404 from `/api/watermark/extract`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_false_positive_gate.py -q
```

Expected before implementation: FAIL because at least one unwatermarked sample is detected.

### Task 3: Add Conservative Small-Crop Verification

**Files:**
- Modify: `main.py`
- Test: `tests/test_false_positive_gate.py`

- [ ] **Step 1: Raise evidence requirements for persistent candidate mode**

In `detect_small_crop_trace`, require stronger evidence when candidates come from persisted records instead of recent in-memory generated trace IDs:

```python
if persistent_candidate_mode:
    if matched_by_short_code:
        continue
    if marker_score < 0.070 or trace_score < 0.030 or max(code_strength, short_strength) < 0.006:
        continue
```

- [ ] **Step 2: Require stronger final votes**

Increase final thresholds for persistent candidate mode so weak single-source patterns cannot become a trace attribution.

- [ ] **Step 3: Verify GREEN on regression**

Run:

```powershell
pytest tests/test_false_positive_gate.py -q
```

Expected: PASS.

### Task 4: Run Commercial Benchmark And Update Reports

**Files:**
- Update generated: `test_output/commercial_trace_benchmark/commercial_trace_test_report.md`
- Update generated: `test_output/commercial_trace_benchmark/commercial_trace_results.json`
- Update generated: `test_output/commercial_trace_benchmark/commercial_trace_results.csv`
- Modify: `docs/commercial_watermark_upgrade_plan.md`

- [ ] **Step 1: Run benchmark**

Run:

```powershell
python tests/commercial_trace_benchmark.py
```

Expected: completes and reports `negative_total`, `false_positive`, and `false_positive_rate`.

- [ ] **Step 2: Update improvement document**

Write the new benchmark numbers into `docs/commercial_watermark_upgrade_plan.md`.

### Task 5: Final Verification

**Files:**
- Read: benchmark outputs

- [ ] **Step 1: Compile and test**

Run:

```powershell
python -m py_compile main.py tests/commercial_trace_benchmark.py tests/test_false_positive_gate.py
pytest tests/test_false_positive_gate.py -q
```

- [ ] **Step 2: Report actual status**

Report whether false positives reached `0/20`, whether wrong trace hits decreased, and what recall changed to. If the first gate reduces recall too much, document the tradeoff and next algorithmic step.
