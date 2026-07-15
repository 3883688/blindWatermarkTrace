import json
import os
import datetime as dt
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.commercial_trace_benchmark import (
    build_report as build_trace_report,
    crop_verdict,
    detection_diagnostics,
    parse_float_list,
)
from tests.commercial_negative_benchmark import (
    build_report as build_negative_report,
    negative_verdict,
)
from tests.commercial_attack_benchmark import (
    attack_verdict,
    build_report as build_attack_report,
    build_report_settings as build_attack_report_settings,
)
from tests.commercial_quality_benchmark import build_report as build_quality_report
from tests.commercial_report_contract import validate_report


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOW = "2026-07-13T12:00:00Z"


def run_powershell_invocation(
    invocation: str,
    *,
    cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
    source_replacements: tuple[tuple[str, str], ...] = (),
) -> subprocess.CompletedProcess:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    assert executable is not None, "PowerShell is required for benchmark orchestration tests"
    escaped_script = str(ROOT / "run_commercial_benchmark.ps1").replace("'", "''")
    escaped_root = str(ROOT).replace("'", "''")
    escaped_invocation = invocation.replace("`", "``").replace("$", "`$").replace('"', '`"')
    replacement_commands = "".join(
        "$source = $source.Replace('" + old.replace("'", "''") + "','"
        + new.replace("'", "''") + "'); "
        for old, new in source_replacements
    )
    command = (
        f"$source = Get-Content -Raw -LiteralPath '{escaped_script}'; "
        + replacement_commands
        + "$prefix = [regex]::Split($source, '(?m)^try \\{', 2)[0]; "
        + f"$prefix = $prefix.Replace('$root = $PSScriptRoot', '$root = ''{escaped_root}'''); "
        + f"$script = $prefix + \"`n{escaped_invocation}\"; "
        + "& ([scriptblock]::Create($script))"
    )
    return subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env={**os.environ, **(environment or {})},
        check=False,
    )


def run_powershell_report_validation(
    report_path: Path, expected_benchmark: str | None = None
) -> subprocess.CompletedProcess:
    escaped_report = str(report_path).replace("'", "''")
    expected = f" '{expected_benchmark}'" if expected_benchmark else ""
    return run_powershell_invocation(
        f"Assert-CommercialReport 'fixture' '{escaped_report}'{expected} | Out-Null"
    )


@pytest.mark.parametrize("cases", [[], [{"case": 1}]])
def test_powershell_report_validation_accepts_contract_json(tmp_path, cases):
    report_path = tmp_path / "valid.json"
    report_path.write_text(
        json.dumps({**stage_report("trace", {}), "cases": cases}),
        encoding="utf-8",
    )

    result = run_powershell_report_validation(report_path)

    assert result.returncode == 0, result.stderr


def test_powershell_read_json_preserves_bomless_utf8_report(tmp_path):
    report_path = tmp_path / "utf8-report.json"
    report_path.write_bytes(
        json.dumps(
            {
                "metadata": {"schema_version": 1},
                "summary": {"label": "中文基线验证"},
                "cases": [
                    {"index": index, "detail": "未检测到可识别的隐式水印"}
                    for index in range(720)
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )
    escaped_report = str(report_path).replace("'", "''")
    invocation = (
        f"$report=Read-Json '{escaped_report}'; "
        "if ($report.metadata.schema_version -ne 1) { throw 'schema' }; "
        "if ($report.summary.label -cne '中文基线验证') { throw 'summary text' }; "
        "if ($report.cases.Count -ne 720) { throw 'case count' }; "
        "if ($report.cases[719].detail -cne '未检测到可识别的隐式水印') { throw 'case text' }"
    )

    result = run_powershell_invocation(invocation)
    broken = run_powershell_invocation(
        invocation,
        source_replacements=((" -Encoding UTF8", ""),),
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert broken.returncode != 0, "encoding-stripped Read-Json unexpectedly preserved UTF-8"


@pytest.mark.parametrize("exit_code", [0, 2, 1])
def test_invoke_benchmark_returns_only_scalar_exit_code_and_hosts_stdout(
    tmp_path, exit_code
):
    marker = f"bounded-marker-{exit_code}"
    fixture = tmp_path / f"benchmark-exit-{exit_code}.py"
    fixture.write_text(
        f"print({marker!r}, flush=True)\nraise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    escaped_fixture = str(fixture).replace("'", "''")
    invocation = (
        f"$actual=@(Invoke-Benchmark 'fixture' '{escaped_fixture}'); "
        "if ($actual.Count -ne 1 -or $actual[0] -isnot [int]) { "
        "throw ('expected single CLR integer, got '+($actual -join '|')) }; "
        f"if ($actual[0] -ne {exit_code}) {{ throw 'wrong exit code' }}; "
        "Write-Host 'ASSERTED_SINGLE_INT'; $codeValue=$actual[0]; exit $codeValue"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == exit_code, result.stderr + result.stdout
    assert marker in result.stdout
    assert "ASSERTED_SINGLE_INT" in result.stdout


def test_invoke_benchmark_test_detects_stdout_pipeline_regression(tmp_path):
    marker = "bounded-pipeline-regression-marker"
    fixture = tmp_path / "benchmark-pipeline-regression.py"
    fixture.write_text(f"print({marker!r}, flush=True)\n", encoding="utf-8")
    escaped_fixture = str(fixture).replace("'", "''")
    invocation = (
        f"$actual=@(Invoke-Benchmark 'fixture' '{escaped_fixture}'); "
        "if ($actual.Count -ne 1 -or $actual[0] -isnot [int]) { "
        "throw ('expected single CLR integer, got '+($actual -join '|')) }"
    )

    broken = run_powershell_invocation(
        invocation,
        source_replacements=(
            ("$benchmarkOutput = & python $ScriptPath", "& python $ScriptPath"),
            ("$benchmarkOutput | ForEach-Object { Write-Host $_ }", ""),
        ),
    )

    assert broken.returncode != 0
    assert "expected single CLR integer" in broken.stderr


@pytest.mark.parametrize(
    "report",
    [
        {"metadata": {"schema_version": "1"}, "summary": {}, "cases": []},
        {"metadata": {"schema_version": 1.0}, "summary": {}, "cases": []},
        {"metadata": {"schema_version": True}, "summary": {}, "cases": []},
        {"metadata": {"schema_version": 1}, "summary": {}, "cases": {}},
        {"metadata": {"schema_version": 1}, "summary": {}, "cases": "cases"},
        {"metadata": {"schema_version": 1}, "summary": {}, "cases": 1},
        {"metadata": {"schema_version": 1}, "summary": {}, "cases": None},
        {"metadata": [], "summary": {}, "cases": []},
        {"metadata": {"schema_version": 1}, "summary": [], "cases": []},
        {"metadata": {"schema_version": 1}, "summary": "summary", "cases": []},
        {"metadata": {"schema_version": 1}, "summary": 1, "cases": []},
        {"metadata": {"schema_version": 1}, "summary": None, "cases": []},
    ],
)
def test_powershell_report_validation_rejects_wrong_json_types(tmp_path, report):
    report_path = tmp_path / "invalid.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_powershell_report_validation(report_path)

    assert result.returncode != 0
    assert "[fixture] commercial report is invalid" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda report: report.pop("settings"), "missing settings"),
        (lambda report: report["metadata"].pop("platform"), "missing metadata.platform"),
        (lambda report: report["metadata"].update(seed=True), "metadata.seed must be an integer"),
        (
            lambda report: report["metadata"].update(generated_at="2026-07-13T12:00:00+00:00"),
            "metadata.generated_at must be a canonical UTC timestamp",
        ),
        (lambda report: report.update(failed_gates=["unexpected"]), "PASS verdict requires no failed gates"),
    ],
)
def test_powershell_report_validation_enforces_strict_contract(
    tmp_path, mutation, expected_error
):
    report = stage_report("trace", {})
    mutation(report)
    report_path = tmp_path / "strict-invalid.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_powershell_report_validation(report_path, "trace")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_powershell_report_validation_rejects_unknown_configuration_without_secret_value(tmp_path):
    secret = "must-not-echo-auth-value"
    report = stage_report("trace", {})
    report["metadata"]["configuration"] = {"WATERMARK_AUTH_KEY": secret}
    report_path = tmp_path / "secret-config.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_powershell_report_validation(report_path, "trace")

    assert result.returncode != 0
    assert "metadata.configuration contains unknown key" in result.stderr
    assert secret not in result.stderr


def stage_report(
    benchmark: str,
    settings: dict,
    verdict: str | None = "PASS",
    *,
    generated_at: str = CANONICAL_NOW,
) -> dict:
    report = {
        "metadata": {
            "schema_version": 1,
            "benchmark": benchmark,
            "algorithm_version": "3",
            "seed": 20260707,
            "generated_at": generated_at,
            "python_version": "3.13.5",
            "platform": "Windows-11",
            "configuration": {},
        },
        "summary": {},
        "cases": [],
        "settings": settings,
        "failed_gates": [] if verdict == "PASS" else ["fixture_gate"],
    }
    if verdict is not None:
        report["verdict"] = verdict
    else:
        report["failed_gates"] = []
    return report


STAGE_REUSE_CASES = {
    "quality": (
        "quality",
        {
            "fidelity_levels": [0.85, 0.9, 0.95, 1.0],
            "quality_min_psnr": 38.0,
            "quality_min_ssim": 0.95,
            "probe_min_recall": 1.0,
            "small_crop_trace_strength": "0.35",
            "small_crop_trace_density": "medium",
            "robust_watermark_strength": "0.74",
            "robust_watermark_version": "3",
            "probes": ["intact"],
        },
        "probe_min_recall",
        0.5,
    ),
    "crop": (
        "trace",
        {
            "fidelity_level": "0.90",
            "scale_factors": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            "crop_ratios": [0.3, 0.5, 0.8, 1.0],
            "negative_scale_factors": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            "negative_crop_ratios": [0.3, 0.5, 0.8, 1.0],
            "crops_per_ratio": 3,
            "workers": 20,
            "small_crop_trace_strength": "0.35",
            "small_crop_trace_density": "medium",
            "robust_watermark_strength": "0.74",
            "robust_watermark_version": "3",
        },
        "crop_ratios",
        [0.5],
    ),
    "attack": (
        "attack",
        {
            "trace_rounds": 5,
            "workers": 20,
            "fidelity_level": "0.90",
            "robust_watermark_strength": "0.74",
            "robust_watermark_version": "3",
            "attack_filter": [],
            "attacks": [
                "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "double_jpeg_70_50",
                "rotate_1deg", "rotate_3deg", "rotate_5deg", "rotate_10deg",
                "gaussian_blur_1_2", "unsharp_mask", "median_denoise",
                "browser_screenshot_sim", "wechat_screenshot_sim", "additive_noise",
                "screen_photo_sim",
            ],
            "negative_sources": ["1.png", "2.png", "3.png", "4.png", "5.png"],
        },
        "trace_rounds",
        4,
    ),
    "negative": (
        "negative",
        {
            "synthetic_variants": 1000,
            "negative_attacks": [
                "jpeg_q90", "jpeg_q50", "jpeg_q30", "rotate_3deg", "rotate_10deg",
                "browser_screenshot_sim", "wechat_screenshot_sim", "screen_photo_sim",
                "gaussian_blur_1_2", "median_denoise",
            ],
            "fidelity_level": "0.90",
            "small_crop_trace_strength": "0.35",
            "small_crop_trace_density": "medium",
            "robust_watermark_strength": "0.74",
            "robust_watermark_version": "3",
        },
        "synthetic_variants",
        999,
    ),
}


def run_fresh_stage_validation(
    report_path: Path,
    subprocess_exit: int,
    *,
    touch_after_start: bool,
    replacement_path: Path | None = None,
) -> subprocess.CompletedProcess:
    escaped_report = str(report_path).replace("'", "''")
    touch = (
        f"[IO.File]::SetLastWriteTimeUtc('{escaped_report}', [DateTime]::UtcNow); "
        if touch_after_start
        else ""
    )
    if replacement_path is None:
        hook = "$hook=$null; "
    else:
        escaped_replacement = str(replacement_path).replace("'", "''")
        hook = (
            "$hook={ "
            + f"[IO.File]::Copy('{escaped_replacement}','{escaped_report}',$true) "
            + "}; "
        )
    invocation = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; "
        "$env:SMALL_CROP_TRACE_STRENGTH='0.35'; $env:SMALL_CROP_TRACE_DENSITY='medium'; "
        f"$before=Get-StageReportState '{escaped_report}'; $started=[DateTime]::UtcNow; {touch}{hook}"
        + f"Assert-FreshStageReport 'crop' '{escaped_report}' 'trace' 'crop' '0.90' {subprocess_exit} $started $before $hook | Out-Null"
    )
    return run_powershell_invocation(invocation)


@pytest.mark.parametrize(
    ("verdict", "subprocess_exit"),
    [("PASS", 0), ("FAIL", 2)],
)
def test_powershell_fresh_stage_accepts_matching_verdict_and_exit(
    tmp_path, verdict, subprocess_exit
):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = tmp_path / "fresh.json"
    report_path.write_text(
        json.dumps(stage_report("trace", settings, verdict, generated_at=generated_at)),
        encoding="utf-8",
    )

    result = run_fresh_stage_validation(report_path, subprocess_exit, touch_after_start=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("verdict", "subprocess_exit"),
    [("PASS", 2), ("FAIL", 0)],
)
def test_powershell_fresh_stage_rejects_verdict_exit_mismatch(
    tmp_path, verdict, subprocess_exit
):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = tmp_path / "mismatch.json"
    report_path.write_text(
        json.dumps(stage_report("trace", settings, verdict, generated_at=generated_at)),
        encoding="utf-8",
    )

    result = run_fresh_stage_validation(report_path, subprocess_exit, touch_after_start=True)

    assert result.returncode != 0
    assert "execution/evidence consistency error" in result.stderr


def test_powershell_fresh_stage_rejects_stale_untouched_report(tmp_path):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    report_path = tmp_path / "stale.json"
    report_path.write_text(
        json.dumps(stage_report("trace", settings, "FAIL", generated_at="2020-01-01T00:00:00Z")),
        encoding="utf-8",
    )
    old_timestamp = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    os.utime(report_path, (old_timestamp, old_timestamp))

    result = run_fresh_stage_validation(report_path, 0, touch_after_start=False)

    assert result.returncode != 0
    assert "report evidence is stale" in result.stderr


@pytest.mark.parametrize("missing", ["verdict", "settings", "benchmark"])
def test_powershell_fresh_stage_rejects_missing_contract_identity(tmp_path, missing):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = stage_report("trace", settings, generated_at=generated_at)
    if missing == "benchmark":
        report["metadata"].pop("benchmark")
    else:
        report.pop(missing)
    report_path = tmp_path / f"missing-{missing}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_fresh_stage_validation(report_path, 0, touch_after_start=True)

    assert result.returncode != 0


@pytest.mark.parametrize("mismatch", ["seed", "config"])
def test_powershell_fresh_stage_rejects_current_run_identity_mismatch(tmp_path, mismatch):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = stage_report("trace", settings, generated_at=generated_at)
    if mismatch == "seed":
        report["metadata"]["seed"] = 7
    else:
        report["settings"] = {**settings, "crop_ratios": [0.5]}
    report_path = tmp_path / f"wrong-{mismatch}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_fresh_stage_validation(report_path, 0, touch_after_start=True)

    assert result.returncode != 0
    assert "does not match current stage configuration" in result.stderr


def test_powershell_fresh_stage_rejects_extra_setting_key(tmp_path):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = stage_report(
        "trace",
        {**settings, "unexpected_behavior": True},
        generated_at=generated_at,
    )
    report_path = tmp_path / "fresh-extra-setting.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_fresh_stage_validation(report_path, 0, touch_after_start=True)

    assert result.returncode != 0
    assert "does not match current stage configuration" in result.stderr


def test_powershell_stage_snapshot_hashes_and_parses_same_utf8_bytes(tmp_path):
    report_path = tmp_path / "snapshot.json"
    payload = json.dumps(
        {**stage_report("trace", {}), "summary": {"label": "same-byte-中文"}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    report_path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest().upper()
    escaped_report = str(report_path).replace("'", "''")
    invocation = (
        f"$snapshot=Get-StageReportSnapshot '{escaped_report}'; "
        f"if ($snapshot.Hash -cne '{expected_hash}') {{ throw 'hash mismatch' }}; "
        f"if ($snapshot.Length -ne {len(payload)}) {{ throw 'length mismatch' }}; "
        "if ($snapshot.Report.summary.label -cne 'same-byte-中文') { throw 'parse mismatch' }"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr + result.stdout


def test_powershell_fresh_stage_rejects_path_replacement_after_snapshot(tmp_path):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = tmp_path / "report-a.json"
    replacement_path = tmp_path / "report-b.json"
    report_path.write_text(
        json.dumps(stage_report("trace", settings, generated_at=generated_at)),
        encoding="utf-8",
    )
    replacement_path.write_text(
        json.dumps(
            stage_report(
                "trace",
                {**settings, "crop_ratios": [0.5]},
                generated_at=generated_at,
            )
        ),
        encoding="utf-8",
    )

    result = run_fresh_stage_validation(
        report_path,
        0,
        touch_after_start=True,
        replacement_path=replacement_path,
    )

    assert result.returncode != 0
    assert "report changed during evidence validation" in result.stderr


def test_powershell_invoke_fresh_benchmark_binds_stub_exit_to_new_report(tmp_path):
    _, settings, _, _ = STAGE_REUSE_CASES["crop"]
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = tmp_path / "stub-fresh.json"
    report_path.write_text(
        json.dumps(stage_report("trace", settings, generated_at=generated_at)),
        encoding="utf-8",
    )
    old_timestamp = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    os.utime(report_path, (old_timestamp, old_timestamp))
    escaped_report = str(report_path).replace("'", "''")
    invocation = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; "
        "$env:SMALL_CROP_TRACE_STRENGTH='0.35'; $env:SMALL_CROP_TRACE_DENSITY='medium'; "
        "function Invoke-Benchmark([string]$Name,[string]$ScriptPath) { "
        + f"[IO.File]::SetLastWriteTimeUtc('{escaped_report}',[DateTime]::UtcNow); return 0 }}; "
        + f"$actual=@(Invoke-FreshBenchmark 'crop' 'stub.py' '{escaped_report}' 'trace' 'crop' '0.90'); "
        + "if ($actual.Count -ne 1 -or $actual[0] -ne 0) { throw 'fresh wrapper exit mismatch' }"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr + result.stdout


def run_stage_reuse(report_path: Path, stage: str) -> subprocess.CompletedProcess:
    escaped_report = str(report_path).replace("'", "''")
    setup = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; "
        "$env:SMALL_CROP_TRACE_STRENGTH='0.35'; $env:SMALL_CROP_TRACE_DENSITY='medium'; "
    )
    fidelity = "0.90" if stage != "quality" else ""
    return run_powershell_invocation(
        setup
        + f"$report = Read-Json '{escaped_report}'; "
        + f"if (-not (Test-StageReportReuse $report '{stage}' '{fidelity}')) {{ throw 'not reusable' }}"
    )


@pytest.mark.parametrize("stage", STAGE_REUSE_CASES)
def test_powershell_stage_reuse_accepts_exact_report(tmp_path, stage):
    benchmark, settings, _, _ = STAGE_REUSE_CASES[stage]
    report = stage_report(benchmark, settings)
    if stage == "negative":
        report["workers"] = 20
    report_path = tmp_path / f"{stage}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_stage_reuse(report_path, stage)

    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize("stage", STAGE_REUSE_CASES)
def test_powershell_stage_reuse_rejects_changed_stage_setting(tmp_path, stage):
    benchmark, settings, changed_key, changed_value = STAGE_REUSE_CASES[stage]
    settings = {**settings, changed_key: changed_value}
    report = stage_report(benchmark, settings)
    if stage == "negative":
        report["workers"] = 20
    report_path = tmp_path / f"{stage}-changed.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_stage_reuse(report_path, stage)

    assert result.returncode != 0


@pytest.mark.parametrize("stage", STAGE_REUSE_CASES)
def test_powershell_stage_reuse_rejects_extra_setting_key(tmp_path, stage):
    benchmark, settings, _, _ = STAGE_REUSE_CASES[stage]
    report = stage_report(benchmark, {**settings, "unexpected_behavior": True})
    if stage == "negative":
        report["workers"] = 20
    report_path = tmp_path / f"{stage}-extra-setting.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_stage_reuse(report_path, stage)

    assert result.returncode != 0


@pytest.mark.parametrize("stage", ["crop", "negative"])
def test_powershell_stage_reuse_rejects_changed_worker_config(tmp_path, stage):
    benchmark, settings, _, _ = STAGE_REUSE_CASES[stage]
    report = stage_report(benchmark, settings)
    if stage == "crop":
        report["settings"] = {**settings, "workers": 19}
    else:
        report["workers"] = 19
    report_path = tmp_path / f"{stage}-workers.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_stage_reuse(report_path, stage)

    assert result.returncode != 0


@pytest.mark.parametrize("stage", STAGE_REUSE_CASES)
@pytest.mark.parametrize("verdict", ["PASS", "FAIL", None])
def test_powershell_stage_reuse_preserves_verdict_semantics(tmp_path, stage, verdict):
    benchmark, settings, _, _ = STAGE_REUSE_CASES[stage]
    report = stage_report(benchmark, settings, verdict)
    if stage == "negative":
        report["workers"] = 20
    report_path = tmp_path / f"{stage}-{verdict}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    escaped_report = str(report_path).replace("'", "''")
    fidelity = "0.90" if stage != "quality" else ""
    expected_reuse = verdict == "PASS" or (verdict == "FAIL" and stage != "quality")
    expected_exit = "0" if verdict == "PASS" else "2" if verdict == "FAIL" else "$null"
    setup = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; "
        "$env:SMALL_CROP_TRACE_STRENGTH='0.35'; $env:SMALL_CROP_TRACE_DENSITY='medium'; "
    )
    invocation = (
        setup
        + f"$report=Read-Json '{escaped_report}'; "
        + f"$reusable=Test-StageReportReuse $report '{stage}' '{fidelity}'; "
        + f"if ($reusable -ne ${str(expected_reuse).lower()}) {{ throw 'reuse mismatch' }}; "
        + "$actualExit=Get-ReusedReportExitCode $report; "
        + f"if ($actualExit -ne {expected_exit}) {{ throw 'exit mismatch' }}"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr + result.stdout


def test_powershell_number_list_always_returns_array():
    invocation = (
        "$single=ConvertTo-NumberList '0.85'; $empty=ConvertTo-NumberList ''; "
        "if ($single -isnot [System.Array] -or $single.Count -ne 1 -or $single[0] -ne 0.85) { throw 'single' }; "
        "if ($empty -isnot [System.Array] -or $empty.Count -ne 0) { throw 'empty' }"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr


def test_powershell_single_number_setting_matches_json_array(tmp_path):
    benchmark, settings, _, _ = STAGE_REUSE_CASES["quality"]
    settings = {**settings, "fidelity_levels": [0.85]}
    report_path = tmp_path / "quality-single-fidelity.json"
    report_path.write_text(json.dumps(stage_report(benchmark, settings)), encoding="utf-8")
    escaped_report = str(report_path).replace("'", "''")
    invocation = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; $env:FIDELITY_LEVELS='0.85'; "
        f"$report=Read-Json '{escaped_report}'; "
        "if (-not (Test-StageReportReuse $report 'quality' '')) { throw 'not reusable' }"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "changed"),
    [("attack_filter", ["jpeg_q90"]), ("negative_sources", ["1.png"])],
)
def test_powershell_attack_reuse_rejects_changed_matrix_identity(tmp_path, field, changed):
    benchmark, settings, _, _ = STAGE_REUSE_CASES["attack"]
    report_path = tmp_path / f"attack-{field}.json"
    report_path.write_text(
        json.dumps(stage_report(benchmark, {**settings, field: changed})), encoding="utf-8"
    )

    result = run_stage_reuse(report_path, "attack")

    assert result.returncode != 0


def test_attack_report_settings_records_effective_matrix():
    result = build_attack_report_settings(
        trace_rounds=2,
        workers=4,
        fidelity_level="0.90",
        robust_watermark_strength="0.74",
        robust_watermark_version="3",
        attack_filter=["jpeg_q90"],
        attacks=["jpeg_q90"],
        negative_sources=["2.png", "1.png"],
    )

    assert result["attack_filter"] == ["jpeg_q90"]
    assert result["attacks"] == ["jpeg_q90"]
    assert result["negative_sources"] == ["1.png", "2.png"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("benchmark", "attack"), ("algorithm_version", "2"), ("seed", 7)],
)
def test_powershell_stage_reuse_rejects_wrong_report_identity(tmp_path, field, value):
    benchmark, settings, _, _ = STAGE_REUSE_CASES["quality"]
    report = stage_report(benchmark, settings)
    report["metadata"][field] = value
    report_path = tmp_path / f"wrong-{field}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_stage_reuse(report_path, "quality")

    assert result.returncode != 0


def test_powershell_reuse_reader_rejects_report_missing_runtime_metadata(tmp_path):
    benchmark, settings, _, _ = STAGE_REUSE_CASES["crop"]
    report = stage_report(benchmark, settings)
    report["metadata"].pop("python_version")
    report_path = tmp_path / "reuse-missing-runtime.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    escaped_report = str(report_path).replace("'", "''")
    invocation = (
        "$env:RANDOM_SEED='20260707'; $env:ROBUST_WATERMARK_VERSION='3'; "
        "$env:ROBUST_WATERMARK_STRENGTH='0.74'; "
        "$env:SMALL_CROP_TRACE_STRENGTH='0.35'; $env:SMALL_CROP_TRACE_DENSITY='medium'; "
        f"$report=Get-ReusableStageReport '{escaped_report}' 'crop' 'trace' 'crop' '0.90'; "
        "if ($null -ne $report) { throw 'invalid report was reusable' }"
    )

    result = run_powershell_invocation(invocation)

    assert result.returncode == 0, result.stderr


def test_powershell_assert_report_rejects_wrong_expected_benchmark(tmp_path):
    report_path = tmp_path / "wrong-benchmark.json"
    report_path.write_text(json.dumps(stage_report("attack", {})), encoding="utf-8")

    result = run_powershell_report_validation(report_path, "trace")

    assert result.returncode != 0
    assert "benchmark must be trace" in result.stderr


def test_powershell_finally_restores_caller_location_and_attack_filter(tmp_path):
    executable = shutil.which("powershell") or shutil.which("pwsh")
    assert executable is not None
    script = str(ROOT / "run_commercial_benchmark.ps1").replace("'", "''")
    root = str(ROOT).replace("'", "''")
    expected_location = str(tmp_path)
    command = (
        f"$source=Get-Content -Raw -LiteralPath '{script}'; "
        "$prefix=[regex]::Split($source,'(?m)^try \\{',2)[0]; "
        f"$prefix=$prefix.Replace('$root = $PSScriptRoot','$root = ''{root}'''); "
        "$body=[regex]::Match($source,'(?ms)^finally \\{(?<body>.*)\\}\\s*$').Groups['body'].Value; "
        "$bounded=$prefix+\"`ntry { Set-Location `$root; Remove-Item Env:\\ATTACK_FILTER; Remove-Item Env:\\NEGATIVE_SOURCES; throw 'bounded' } finally {\"+$body+'}'; "
        "try { & ([scriptblock]::Create($bounded)) } catch { }; "
        "Write-Output ('cwd='+(Get-Location).Path); Write-Output ('attack='+$env:ATTACK_FILTER); Write-Output ('negative='+$env:NEGATIVE_SOURCES)"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=tmp_path,
        env={**os.environ, "ATTACK_FILTER": "keep-me", "NEGATIVE_SOURCES": "2.png"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"cwd={expected_location}" in result.stdout
    assert "attack=keep-me" in result.stdout
    assert "negative=2.png" in result.stdout


def test_commercial_runner_preserves_explicit_v4_and_reads_auth_key_from_dotenv():
    source = (ROOT / "run_commercial_benchmark.ps1").read_text(encoding="utf-8")

    assert (
        "$env:ROBUST_WATERMARK_VERSION = "
        "Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '3'"
    ) in source
    assert "$env:ROBUST_WATERMARK_VERSION = '3'" not in source
    assert "$env:WATERMARK_AUTH_KEY =" not in source


def passing_crop_summary():
    return {
        "recall": 0.96,
        "wrong": 0,
        "false_positive": 0,
        "by_crop_ratio": {
            "0.3": {"recall": 0.80},
            "0.5": {"recall": 0.95},
            "0.8": {"recall": 1.0},
            "1.0": {"recall": 1.0},
        },
    }


def assert_report_adapter(build_report, benchmark):
    summary = {"passed": 3}
    cases = [{"case": "one"}]
    settings = {"fidelity_level": "0.90"}

    result = build_report(
        summary,
        cases,
        seed=17,
        algorithm_version="v4-baseline",
        settings=settings,
        verdict="PASS",
        failed_gates=[],
    )

    assert result["metadata"]["benchmark"] == benchmark
    assert result["metadata"]["seed"] == 17
    assert result["metadata"]["algorithm_version"] == "v4-baseline"
    assert result["summary"] is summary
    assert result["cases"] is cases
    assert result["settings"] is settings
    assert result["verdict"] == "PASS"
    assert result["failed_gates"] == []
    assert validate_report(result) == []


def test_trace_report_adapter_uses_common_contract():
    assert_report_adapter(build_trace_report, "trace")


def test_attack_report_adapter_uses_common_contract():
    assert_report_adapter(build_attack_report, "attack")


def test_negative_report_adapter_uses_common_contract():
    assert_report_adapter(build_negative_report, "negative")


def test_quality_report_adapter_uses_common_contract():
    assert_report_adapter(build_quality_report, "quality")


def test_parse_float_list_reads_environment(monkeypatch):
    monkeypatch.setenv("TEST_FACTORS", "0.5, 1.0,2")
    assert parse_float_list("TEST_FACTORS", "0.1") == [0.5, 1.0, 2.0]


def test_crop_verdict_passes_only_when_all_gates_pass():
    assert crop_verdict(passing_crop_summary()) == {"verdict": "PASS", "failed_gates": []}


def test_crop_verdict_fails_on_any_false_attribution():
    summary = passing_crop_summary()
    summary["false_positive"] = 1
    summary["wrong"] = 1

    result = crop_verdict(summary)

    assert result["verdict"] == "FAIL"
    assert result["failed_gates"] == ["wrong_trace", "false_positive"]


def test_crop_verdict_checks_small_and_large_crop_buckets():
    summary = passing_crop_summary()
    summary["by_crop_ratio"]["0.3"]["recall"] = 0.79
    summary["by_crop_ratio"]["0.5"]["recall"] = 0.94

    result = crop_verdict(summary)

    assert result["verdict"] == "FAIL"
    assert result["failed_gates"] == ["crop_0.3_recall", "crop_0.5_recall"]


def test_detection_diagnostics_extracts_rs_recovery_fields():
    detected = {
        "code_recovery": {
            "bit_errors": 19,
            "corrected_symbols": 13,
            "erasure_count": 2,
            "recovery_method": "expected_codeword_distance",
            "phase_tile_counts": [4, 3, 3],
        }
    }

    assert detection_diagnostics(detected) == {
        "bit_errors": 19,
        "corrected_symbols": 13,
        "erasure_count": 2,
        "recovery_method": "expected_codeword_distance",
        "phase_tile_counts": "4/3/3",
        "authenticated_tiles": "",
        "mean_signed_agreement": "",
    }


def test_detection_diagnostics_extracts_hmac_v3_fields():
    detected = {
        "code_recovery": {
            "bit_errors": 5,
            "authenticated_tiles": 18,
            "phase_tile_counts": [6, 6, 6],
            "mean_signed_agreement": 0.42,
        }
    }

    result = detection_diagnostics(detected)

    assert result["bit_errors"] == 5
    assert result["authenticated_tiles"] == 18
    assert result["phase_tile_counts"] == "6/6/6"
    assert result["mean_signed_agreement"] == 0.42


def test_negative_verdict_requires_zero_source_false_positives_and_rate_below_point_one_percent():
    passing = {"total": 1001, "false_positive": 0, "false_positive_rate": 0.0, "source_false_positive": 0}
    assert negative_verdict(passing) == {"verdict": "PASS", "failed_gates": []}

    source_failure = {**passing, "false_positive": 1, "false_positive_rate": 0.0009, "source_false_positive": 1}
    assert negative_verdict(source_failure)["failed_gates"] == ["source_false_positive"]

    rate_failure = {**passing, "false_positive": 2, "false_positive_rate": 0.0015}
    assert negative_verdict(rate_failure)["failed_gates"] == ["false_positive_rate"]


def test_negative_verdict_uses_raw_rate_below_boundary():
    summary = {
        "total": 1001,
        "false_positive": 1,
        "false_positive_rate": 0.001,
        "source_false_positive": 0,
    }

    assert negative_verdict(summary) == {"verdict": "PASS", "failed_gates": []}


def test_negative_verdict_fails_at_exact_rate_boundary():
    summary = {
        "total": 1000,
        "false_positive": 1,
        "false_positive_rate": 0.001,
        "source_false_positive": 0,
    }

    assert negative_verdict(summary) == {
        "verdict": "FAIL",
        "failed_gates": ["false_positive_rate"],
    }


def test_attack_verdict_checks_every_trace_round():
    summary = {
        "wrong": 0,
        "false_positive": 0,
        "by_round": {
            "1": {"recall": 1.0, "wrong": 0, "false_positive": 0},
            "2": {"recall": 0.94, "wrong": 0, "false_positive": 0},
        },
    }

    result = attack_verdict(summary)

    assert result == {"verdict": "FAIL", "failed_gates": ["round_2_recall"]}


def test_attack_verdict_fails_on_any_wrong_trace_or_negative_hit():
    summary = {
        "wrong": 1,
        "false_positive": 1,
        "by_round": {"1": {"recall": 1.0, "wrong": 1, "false_positive": 1}},
    }

    result = attack_verdict(summary)

    assert result["failed_gates"] == [
        "wrong_trace",
        "false_positive",
        "round_1_wrong_trace",
        "round_1_false_positive",
    ]
