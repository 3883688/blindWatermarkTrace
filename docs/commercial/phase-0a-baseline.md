# Phase 0A Baseline Evidence

Status: `PHASE_0A_COMPLETE`. The commercial algorithm gate remains truthfully `FAIL`; Phase 0A completion means the preservation, evidence, and baseline infrastructure is complete, not that V4 is commercially qualified.

## Scope and interpretation

The current unit regression suite is green, but the current commercial algorithm gate is truthfully `FAIL`. These are separate results: unit tests show no detected regression in the tested code paths, while the commercial crop matrix shows that recall is below the required gate.

No Git commands were run and no Git metadata is recorded. No `.env` values, authentication keys, database URLs, passwords, or user-specific absolute paths are included here.

## Environment fingerprint

Captured at `2026-07-13T13:49:14.083+08:00` in `China Standard Time` (`UTC+08:00`).

- OS: Microsoft Windows 11 Pro, version `10.0.26200`, build `26200`.
- Python: `3.13.7`.
- [`requirements.txt`](../../requirements.txt) SHA-256: `6d4263675325cd88af77b9f21eec672698c68e6ce043135ef9010f4e8f364598`.
- `fastapi`: `0.138.1`.
- `uvicorn[standard]`: `0.49.0`.
- `python-multipart`: `0.0.32`.
- `pillow`: `11.3.0`.
- `python-dotenv`: `1.2.2`.
- `sqlalchemy`: `2.0.51`.
- `pymysql`: `1.2.0`.
- `pytest`: `9.1.1`.
- `httpx`: `0.28.1`.
- `opencv-python-headless`: not installed under that distribution name. `opencv-python==4.13.0.92` supplies the installed OpenCV runtime.
- `PyWavelets`: `1.9.0`.
- `reedsolo`: `1.7.0` (matches the `==1.7.0` pin).

Fingerprint commands:

```powershell
Get-Date
Get-TimeZone
Get-CimInstance Win32_OperatingSystem
python --version
python -m pip list --format=freeze
Get-FileHash -Algorithm SHA256 requirements.txt
```

## Unit regression baseline

Command:

```powershell
python -m pytest -q
```

Fresh uninterrupted result: exit `0`; `380 passed, 2 skipped, 184 warnings in 165.46s (0:02:45)`; measured wall time `166.062s`.

The two skips are shown by pytest as `ss`; pytest did not emit skip-reason details under `-q`. The 184 warnings comprise one `StarletteDeprecationWarning` from FastAPI's TestClient import and 183 Pillow `DeprecationWarning` instances from image construction paths in the application and tests.

After the two original bounded runner integration corrections, the full command was run again: exit `0`; `380 passed, 2 skipped, 184 warnings in 184.36s (0:03:04)`; measured wall time `185.046s`.

After all Phase 0A review fixes, the final uninterrupted verification was run on the completed code: exit `0`; `472 passed, 2 skipped, 184 warnings in 153.88s (0:02:33)`. This is the authoritative Phase 0A unit-test result.

## Bounded standardized commercial baseline

No benchmark environment variables were manually exported. The runner applied the nonsecret settings recorded in each report. Attack and long negative stages were not run.

Quality command:

```powershell
powershell -ExecutionPolicy Bypass -File run_commercial_benchmark.ps1 -Stage quality
```

Result: exit `0`; wall time `187.306s`; recommended fidelity `1.0`. The fresh report is [`commercial_quality_results.json`](../../test_output/commercial_quality_benchmark/commercial_quality_results.json).

Crop command used to generate the current report, without reuse:

```powershell
powershell -ExecutionPolicy Bypass -File run_commercial_benchmark.ps1 -Stage crop
```

The current generation took `149.625s`; the benchmark itself returned `2` for a truthful gate failure. During evidence collection this exposed a second orchestration issue, described below, so the wrapper returned `0` on that particular generation. After the fix, the existing fresh report was used only to verify final exit propagation:

```powershell
powershell -ExecutionPolicy Bypass -File run_commercial_benchmark.ps1 -Stage crop -Reuse
```

Final wrapper verification result: exit `2`. The fresh report is [`commercial_trace_results.json`](../../test_output/commercial_trace_benchmark/commercial_trace_results.json).

Both reports were parsed with Python's JSON parser and passed `tests.commercial_report_contract.validate_report` with `[]` errors. Both contain `metadata`, `summary`, and list-valued `cases`:

- Quality: schema `1`, benchmark `quality`, algorithm `3`, seed `20260707`, 4 cases, verdict `PASS`, no failed gates.
- Crop: schema `1`, benchmark `trace`, algorithm `3`, seed `20260707`, 720 cases, verdict `FAIL`.

### Quality result

Configured gates were minimum PSNR `38.0`, minimum SSIM `0.95`, and intact-probe recall `1.0`. At the recommended fidelity `1.0`, observed minimum PSNR was `40.783512`, observed minimum SSIM was `0.957177`, intact-probe recall was `1.0`, wrong traces were `0`, and false positives were `0` across 5 negatives.

Detection latency for the recommended-fidelity probe rows, using nearest-rank percentiles:

| Cases | Count | P50 ms | P95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| All | 10 | 423.906 | 3251.729 | 3251.729 |
| Watermarked | 5 | 2749.919 | 3251.729 | 3251.729 |
| Unwatermarked | 5 | 367.799 | 423.906 | 423.906 |

### Crop result

Overall: recall `0.6139` (`221/360` correct), wrong traces `0`, wrong-trace rate `0.0`, false positives `0/360`, and false-positive rate `0.0`. Failed gates: `overall_recall`, `crop_0.3_recall`, `crop_0.5_recall`, `crop_0.8_recall`, and `crop_1.0_recall`.

Recall by scale:

| Scale | Correct / total | Recall |
| --- | ---: | ---: |
| 0.5 | 45 / 60 | 0.7500 |
| 0.75 | 45 / 60 | 0.7500 |
| 1.0 | 58 / 60 | 0.9667 |
| 1.25 | 35 / 60 | 0.5833 |
| 1.5 | 21 / 60 | 0.3500 |
| 2.0 | 17 / 60 | 0.2833 |

Recall by crop ratio:

| Crop ratio | Correct / total | Recall |
| --- | ---: | ---: |
| 0.3 | 37 / 90 | 0.4111 |
| 0.5 | 82 / 90 | 0.9111 |
| 0.8 | 57 / 90 | 0.6333 |
| 1.0 | 45 / 90 | 0.5000 |

Detection latency across the 720 crop-report cases, using nearest-rank percentiles:

| Cases | Count | P50 ms | P95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| All | 720 | 2999.451 | 5490.369 | 7160.723 |
| Watermarked | 360 | 2980.392 | 5399.854 | 6056.650 |
| Unwatermarked | 360 | 3004.901 | 5512.275 | 7160.723 |

## Runner integration corrections

The first fresh crop generation produced valid standardized UTF-8 JSON and benchmark exit `2`, but Windows PowerShell 5.1 returned wrapper exit `1`. Root cause: `Get-Content` used the Windows PowerShell legacy default encoding for the BOM-less UTF-8 report, corrupting non-ASCII strings before `ConvertFrom-Json`. The permanent regression test creates a sizable 720-case BOM-less UTF-8 report with exact Chinese text, exercises the extracted production `Read-Json` under Windows PowerShell, and confirms that a controlled variant with `-Encoding UTF8` stripped fails. The minimal fix added explicit `-Encoding UTF8` in [`run_commercial_benchmark.ps1`](../../run_commercial_benchmark.ps1).

The next fresh generation exposed that native Python stdout was also entering the PowerShell function success pipeline, turning scalar exit `2` into an array and causing wrapper exit `0`. Permanent bounded regression fixtures now cover printed-marker exits `0`, `2`, and `1`, asserting a single CLR integer return, exact value, host-stream marker visibility, and caller nonzero semantics. A controlled extracted variant that restores native stdout to the success pipeline is required to fail. The runner captures and hosts benchmark stdout while returning only the scalar exit code; stage callers retain execution-error handling for code `1`. TDD evidence was RED at `1 failed, 63 passed` for the pre-refinement exit-1 throw, then GREEN at `64 passed, 1 warning`; `python -m py_compile tests\test_commercial_benchmark_gates.py` also passed. These tests launch only temporary bounded Python fixtures, not commercial benchmarks. The earlier final `-Reuse` verification returned exit `2`.

These are orchestration fixes only. No watermark algorithm behavior was changed.

## V3 archive verification

[`trace-v3-source-20260713.zip`](../../backups/trace-v3-source-20260713.zip) has SHA-256 `abf858cc83e92a0691afd92955abbcc9199a310839b21ee8105cb61e30b0b3c2`, exactly matching [`SHA256SUMS`](../../backups/SHA256SUMS). The ZIP contains 63 entries; [`trace-v3-source-20260713.manifest.txt`](../../backups/trace-v3-source-20260713.manifest.txt) contains 62 non-empty lines.

This artifact is a **SENSITIVE rollback source archive**, not a secret-free backup. The capture excludes `.env`, `.env.*`, runtime data, logs, caches, explicit credential files, and private-key files. Exact legacy rollback still retains source-embedded or documented default credentials; their values are intentionally not repeated here. Access is restricted, the archive must not be deployed as application content, and all retained/default credentials must be removed or rotated before production.

The archive set is immutable by default. `tools/create_source_backup.ps1` exits nonzero before staging if any final ZIP, manifest, or checksum already exists. Intentional replacement requires explicit `-Force` and uses verified-before-publish rollback protection.

Windows ACLs on the ZIP, manifest, and checksum were inspected and explicitly protected after capture. Inheritance is disabled on each artifact; full control is limited to the current user, SYSTEM, and the local Administrators group. The account name and SID are omitted from this evidence document for privacy. The containing directory retains its existing ACL so unrelated workspace behavior is not changed.

Verification commands:

```powershell
Get-FileHash -Algorithm SHA256 backups\trace-v3-source-20260713.zip
Get-Content backups\SHA256SUMS
# ZIP entry count read with System.IO.Compression.ZipFile::OpenRead(...).Entries.Count
```

## Limitations and pending evidence

- No real-route collected samples are present; 3 samples remain pending.
- The 300 negative slots remain pending.
- The attack suite and long negative suite were not run in this bounded baseline.
- The commercial crop/trace verdict is truthfully `FAIL`; this baseline does not satisfy the commercial algorithm gate.
- Phase 0B V4 signal-processing implementation has not started; it requires its own implementation plan and acceptance evidence.
