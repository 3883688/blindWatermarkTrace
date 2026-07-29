import pytest

from tests.v4.benchmark_manifest import (
    GateFailure, evaluate_release_report, passing_report, sign_report, verify_signed_report,
)


def test_negative_gate_requires_3000_independent_samples_and_zero_attribution() -> None:
    report = passing_report()
    report["suites"]["negative"] = {"independent_count": 2999, "wrong_attributions": 0}
    with pytest.raises(GateFailure, match="3000"):
        evaluate_release_report(report)
    report["suites"]["negative"] = {"independent_count": 3000, "wrong_attributions": 1}
    with pytest.raises(GateFailure, match="wrong attribution"):
        evaluate_release_report(report)


def test_missing_manifest_hardware_or_model_evidence_fails_closed() -> None:
    for field in ("manifest_hashes", "hardware", "model_health"):
        report = passing_report(); report.pop(field)
        with pytest.raises(GateFailure, match=field):
            evaluate_release_report(report)


def test_signed_report_is_bound_to_current_commit() -> None:
    report = passing_report(); signed = sign_report(report, b"release-key")
    verify_signed_report(signed, b"release-key", git_commit=report["git_commit"])
    with pytest.raises(GateFailure, match="stale"):
        verify_signed_report(signed, b"release-key", git_commit="b" * 40)
