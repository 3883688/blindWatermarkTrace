import pytest

from tests.v4.benchmark_manifest import GateFailure, evaluate_release_report, passing_report


def test_performance_gate_enforces_p95_120_hard_300_and_deep_1000() -> None:
    for key, value, message in (
        ("standard_p95_seconds", 120.01, "P95"),
        ("standard_max_seconds", 300.01, "300"),
        ("deep_max_seconds", 1000.01, "1000"),
    ):
        report = passing_report(); report["suites"]["performance"][key] = value
        with pytest.raises(GateFailure, match=message):
            evaluate_release_report(report)


def test_recall_and_final_attribution_thresholds_are_enforced() -> None:
    report = passing_report(); report["suites"]["recall"]["jpeg"] = 0.989
    with pytest.raises(GateFailure, match="recall"):
        evaluate_release_report(report)
    report = passing_report(); report["suites"]["attribution"]["rate"] = 0.949
    with pytest.raises(GateFailure, match="attribution"):
        evaluate_release_report(report)
