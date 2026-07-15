# Commercial V4 Phase 0A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Preserve the current V3 source, remove baseline configuration drift, standardize commercial benchmark evidence, establish release dataset manifests, and capture a reproducible pre-V4 baseline.

**Architecture:** Phase 0A does not implement V4 signal processing. It adds small, testable tooling under `tools/` and `tests/`, keeps benchmark contracts independent from `main.py`, and records immutable evidence under `backups/` and `docs/commercial/`. Because this workspace has no Git metadata, each task ends with a local verification checkpoint rather than a commit.

**Tech Stack:** PowerShell 7/Windows PowerShell, Python 3.13, pytest, JSON/CSV/Markdown, SHA-256

---

## File Structure

- Create `tools/create_source_backup.ps1`: verified SENSITIVE rollback source archive with immutable-by-default publication.
- Create `tests/test_source_backup_contract.py`: verifies backup inclusion/exclusion rules without creating the production archive.
- Modify `main.py`: make `None` normalization deterministic while preserving explicit environment-selected API defaults.
- Modify `tests/test_aligned_authenticated_detection.py`: cover deterministic normalization and explicit versions.
- Create `tests/commercial_report_contract.py`: common report metadata and schema validation.
- Create `tests/test_commercial_report_contract.py`: report-contract unit tests.
- Modify `tests/commercial_trace_benchmark.py`: attach common metadata and validate output before writing.
- Modify `tests/commercial_attack_benchmark.py`: attach common metadata and validate output before writing.
- Modify `tests/commercial_negative_benchmark.py`: attach common metadata and validate output before writing.
- Modify `tests/commercial_quality_benchmark.py`: attach common metadata and validate output before writing.
- Create `tests/commercial_dataset_manifest.py`: load and validate release dataset manifests.
- Create `tests/test_commercial_dataset_manifest.py`: manifest category, uniqueness, path, and count tests.
- Create `tests/fixtures/commercial/manifests/negative-development.json`: 100-slot development manifest.
- Create `tests/fixtures/commercial/manifests/negative-release.json`: 300-slot release manifest.
- Create `tests/fixtures/commercial/manifests/real-platform-routes.json`: real-route intake manifest.
- Create `docs/commercial/real-sample-intake.md`: operator procedure for collecting real propagation samples.
- Create `docs/commercial/phase-0a-baseline.md`: commands, environment fingerprint, test outcome, and benchmark references.

## Task 1: Create and Verify the V3 Source Backup

The result is a **SENSITIVE rollback source archive**, not a secret-free artifact. `.env`, `.env.*`, runtime data, logs, caches, explicit credential files, and private-key files are excluded, but exact rollback necessarily retains legacy source-embedded or documented default credentials. Restrict archive access and rotate/remove those credentials before production.

**Files:**
- Create: `tools/create_source_backup.ps1`
- Create: `tests/test_source_backup_contract.py`
- Create at execution: `backups/trace-v3-source-20260713.zip`
- Create at execution: `backups/trace-v3-source-20260713.manifest.txt`
- Create at execution: `backups/SHA256SUMS`

- [x] **Step 1: Write the failing backup-contract tests**

```python
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "create_source_backup.ps1"


def test_backup_script_declares_required_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "trace-v3-source-20260713.zip" in text
    assert "trace-v3-source-20260713.manifest.txt" in text
    assert "SHA256SUMS" in text


def test_backup_script_excludes_secrets_and_runtime_data():
    text = SCRIPT.read_text(encoding="utf-8")
    for excluded in (".env", "data", "uploads", "output", "test_output", "__pycache__"):
        assert excluded in text
    assert "Expand-Archive" in text
```

- [x] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_source_backup_contract.py -q`  
Expected: FAIL because `tools/create_source_backup.ps1` does not exist.

- [x] **Step 3: Implement the backup script**

The script must resolve the repository root from `$PSScriptRoot`, enumerate files with `Get-ChildItem -File -Recurse`, apply the tested nested runtime/cache/credential exclusions, sort normalized relative paths, stage only those paths, call `Compress-Archive`, compute `Get-FileHash -Algorithm SHA256`, extract with `Expand-Archive` into a temporary directory, and compare the extracted relative file list byte-for-byte with the manifest. It must delete only guarded temporary directories it created. Existing final artifacts are immutable by default: replacement fails before staging unless the operator explicitly supplies `-Force`; forced replacement retains verified-before-publish rollback protection.

- [x] **Step 4: Run the contract test and verify GREEN**

Run: `python -m pytest tests/test_source_backup_contract.py -q`  
Expected: `2 passed`.

- [x] **Step 5: Create and verify the actual V3 archive**

Run once when no artifact exists: `powershell -ExecutionPolicy Bypass -File tools/create_source_backup.ps1`  
Expected: exit code 0 and output naming the archive, manifest, SHA-256, and successful extraction verification. Once any final artifact exists, the same command must fail without changing it. `-Force` is reserved for an explicitly authorized intentional replacement.

- [x] **Step 6: Independently verify artifacts**

Run: `Get-FileHash backups/trace-v3-source-20260713.zip -Algorithm SHA256; Get-Content backups/SHA256SUMS; Get-Item backups/trace-v3-source-20260713.zip`  
Expected: both hashes match and the ZIP has non-zero length.

## Task 2: Remove Default-Version Test Drift

**Files:**
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `main.py:740`

- [x] **Step 1: Extend the failing test to state the complete contract**

```python
def test_robust_watermark_version_normalization_is_environment_independent(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_ROBUST_WATERMARK_VERSION", "3")

    assert main.normalize_robust_watermark_version(None) == 1
    assert main.normalize_robust_watermark_version("1") == 1
    assert main.normalize_robust_watermark_version("2") == 2
    assert main.normalize_robust_watermark_version("3") == 3
    assert main.normalize_robust_watermark_version("invalid") == 1
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_aligned_authenticated_detection.py::test_robust_watermark_version_normalization_is_environment_independent -q`  
Expected: FAIL because `None` currently reads `DEFAULT_ROBUST_WATERMARK_VERSION` and returns 3.

- [x] **Step 3: Implement deterministic normalization**

Change `normalize_robust_watermark_version` so `None` maps to `ROBUST_WATERMARK_VERSION_V1`. Explicit API defaults continue to work because FastAPI passes `DEFAULT_ROBUST_WATERMARK_VERSION` as the form default rather than passing `None`.

```python
def normalize_robust_watermark_version(value: str | int | None) -> int:
    try:
        version = int(value if value is not None else ROBUST_WATERMARK_VERSION_V1)
    except (TypeError, ValueError):
        version = ROBUST_WATERMARK_VERSION_V1
    if version == ROBUST_WATERMARK_VERSION_V3:
        return ROBUST_WATERMARK_VERSION_V3
    if version == ROBUST_WATERMARK_VERSION_V2:
        return ROBUST_WATERMARK_VERSION_V2
    return ROBUST_WATERMARK_VERSION_V1
```

- [x] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_aligned_authenticated_detection.py::test_robust_watermark_version_normalization_is_environment_independent -q`  
Expected: PASS.

Run: `python -m pytest -q`  
Expected: all tests pass; record the exact warning count rather than hiding warnings.

## Task 3: Add a Common Commercial Report Contract

**Files:**
- Create: `tests/commercial_report_contract.py`
- Create: `tests/test_commercial_report_contract.py`

- [x] **Step 1: Write failing report-contract tests**

```python
from tests.commercial_report_contract import build_report_metadata, validate_report


def test_report_metadata_is_reproducible_and_secret_free(monkeypatch):
    monkeypatch.setenv("WATERMARK_AUTH_KEY", "must-not-leak")
    metadata = build_report_metadata("trace", seed=20260713, algorithm_version="v3-baseline")
    assert metadata["schema_version"] == 1
    assert metadata["benchmark"] == "trace"
    assert metadata["seed"] == 20260713
    assert metadata["algorithm_version"] == "v3-baseline"
    assert "must-not-leak" not in repr(metadata)


def test_validate_report_rejects_missing_result_sections():
    report = {"metadata": build_report_metadata("trace", 1, "v3-baseline")}
    errors = validate_report(report)
    assert errors == ["missing summary", "missing cases"]
```

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_commercial_report_contract.py -q`  
Expected: collection error because the module does not exist.

- [x] **Step 3: Implement the minimal contract**

`build_report_metadata` returns `schema_version`, `benchmark`, `algorithm_version`, `seed`, `generated_at`, Python version, platform, and a filtered configuration dictionary containing only an explicit allowlist of non-secret benchmark environment keys. `validate_report` returns ordered errors for missing `metadata`, `summary`, or `cases`, invalid schema version, and a non-list `cases` value.

- [x] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_commercial_report_contract.py -q`  
Expected: all report-contract tests pass.

## Task 4: Apply the Report Contract to Existing Benchmarks

**Files:**
- Modify: `tests/commercial_trace_benchmark.py`
- Modify: `tests/commercial_attack_benchmark.py`
- Modify: `tests/commercial_negative_benchmark.py`
- Modify: `tests/commercial_quality_benchmark.py`
- Modify: `tests/test_commercial_benchmark_gates.py`

- [x] **Step 1: Write one failing adapter test per benchmark**

Each benchmark exposes a pure `build_report(summary, cases, seed, algorithm_version)` function. Tests assert that the result contains the common metadata, preserves the existing summary/verdict fields, stores cases as a list, and passes `validate_report`.

- [x] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/test_commercial_benchmark_gates.py -q`  
Expected: FAIL because the four `build_report` functions do not exist.

- [x] **Step 3: Implement minimal adapters and validate before file writes**

Import `build_report_metadata` and `validate_report`. Construct the standardized top-level structure without changing detection logic. Immediately before JSON/CSV/Markdown output, call `validate_report`; raise `ValueError("invalid commercial report: ...")` if errors exist.

- [x] **Step 4: Run benchmark unit tests**

Run: `python -m pytest tests/test_commercial_benchmark_gates.py tests/test_commercial_quality_metrics.py tests/test_commercial_benchmark_config.py -q`  
Expected: all selected tests pass.

## Task 5: Establish Dataset Manifests and Validation

**Files:**
- Create: `tests/commercial_dataset_manifest.py`
- Create: `tests/test_commercial_dataset_manifest.py`
- Create: `tests/fixtures/commercial/manifests/negative-development.json`
- Create: `tests/fixtures/commercial/manifests/negative-release.json`
- Create: `tests/fixtures/commercial/manifests/real-platform-routes.json`

- [x] **Step 1: Write failing manifest tests**

Tests require unique sample IDs, allowed categories (`photo`, `illustration`, `ui`, `low_texture`, `high_texture`, `similar_composition`), relative paths only, no path traversal, exact target counts of 100 and 300 slots, and required real routes (`wechat`, `browser`, `target_platform`). Empty slots use `status: "pending_collection"` and may not be counted as collected samples.

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_commercial_dataset_manifest.py -q`  
Expected: collection error because the validator and manifests do not exist.

- [x] **Step 3: Implement the manifest validator**

Expose `load_manifest(path)`, `validate_negative_manifest(data, expected_slots)`, and `validate_route_manifest(data)`. Return ordered validation errors; do not silently repair input.

- [x] **Step 4: Create deterministic slot manifests**

Use IDs `negative-0001` through `negative-0300`. The 100-slot development manifest is the first 100 IDs from the release manifest. Distribute categories deterministically and mark uncollected entries as pending. Real-route records specify source ID, route, sent timestamp, received timestamp, output relative path, and collection status.

- [x] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_commercial_dataset_manifest.py -q`  
Expected: all manifest tests pass and collected counts are reported separately from slot counts.

## Task 6: Document the Real Sample Intake Procedure

**Files:**
- Create: `docs/commercial/real-sample-intake.md`

- [x] **Step 1: Add a documentation contract test**

Add a test that requires the intake document to name all three routes, preserve original and received files, forbid image editing after receipt, record timestamps and operator, use relative paths, and calculate SHA-256 for both files.

- [x] **Step 2: Run the documentation test and verify RED**

Run: `python -m pytest tests/test_commercial_dataset_manifest.py -q`  
Expected: FAIL because the intake document is absent.

- [x] **Step 3: Write the operator procedure**

Document preparation, route-specific send/receive steps, naming, hashes, metadata entry, rejected-sample handling, and the rule that simulated samples cannot be marked as real-route evidence.

- [x] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_commercial_dataset_manifest.py -q`  
Expected: all tests pass.

## Task 7: Capture the Pre-V4 Baseline

**Files:**
- Create: `docs/commercial/phase-0a-baseline.md`
- Modify: `run_commercial_benchmark.ps1` to run schema validation after each selected benchmark stage

- [x] **Step 1: Record environment fingerprint**

Run: `python --version; python -m pip freeze; Get-FileHash requirements.txt -Algorithm SHA256`  
Record Python version, dependency versions, requirements hash, OS, date, explicit benchmark environment variables, and the fact that the directory has no Git metadata. Never record `.env` values or authentication keys.

- [x] **Step 2: Run the complete unit suite**

Run: `python -m pytest -q`  
Expected: all tests pass. Record duration and warning count.

- [x] **Step 3: Run the bounded existing benchmark baseline**

Run: `powershell -ExecutionPolicy Bypass -File run_commercial_benchmark.ps1 -Stage crop`  
Expected: command completes and produces a schema-valid report. A `FAIL` commercial verdict is acceptable and must be recorded as the truthful V3 baseline.

- [x] **Step 4: Validate report artifacts**

Run a small Python command that loads each newly generated JSON report and calls `validate_report`.  
Expected: an empty error list for every report.

- [x] **Step 5: Write the baseline summary**

Record exact recall by route/crop, wrong trace count, false-positive count, PSNR/SSIM where available, P50/P95/max latency, command duration, and links to retained report files. State gaps where real-route or collected-negative data is not yet available.

- [x] **Step 6: Final Phase 0A verification**

Run: `python -m pytest -q`  
Run: `Get-FileHash backups/trace-v3-source-20260713.zip -Algorithm SHA256`  
Run: `python -m pytest tests/test_source_backup_contract.py tests/test_commercial_report_contract.py tests/test_commercial_dataset_manifest.py -q`  
Expected: all tests pass, archive hash matches `backups/SHA256SUMS`, and no implementation file outside the Phase 0A file list changed except the approved `main.py` normalization fix and benchmark adapters.

## Phase 0A Exit Gate

Phase 0A is complete only when:

- the V3 archive, manifest, checksum, and extraction verification exist;
- the full unit suite passes without the current default-version failure;
- all commercial reports follow the common contract;
- 100- and 300-slot manifests validate and clearly distinguish pending from collected data;
- the real-route intake procedure is usable by an operator;
- the truthful pre-V4 baseline is retained with commands and environment fingerprint.

After this gate, write a separate Phase 0B implementation plan for `watermark_v4/` payload, synchronization, embedding, detection, and API integration. Do not implement Phase 0B from this plan.

