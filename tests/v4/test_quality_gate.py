import pytest

from tests.v4.benchmark_manifest import GateFailure, QUALITY_STRATA, evaluate_release_report, passing_report


def test_quality_gate_requires_every_stratum_at_psnr_38_and_ssim_095() -> None:
    report = passing_report(); report["suites"]["quality"][next(iter(QUALITY_STRATA))]["psnr"] = 37.99
    with pytest.raises(GateFailure, match="PSNR"):
        evaluate_release_report(report)
    report = passing_report(); report["suites"]["quality"]["ui"]["ssim"] = 0.949
    with pytest.raises(GateFailure, match="SSIM"):
        evaluate_release_report(report)
