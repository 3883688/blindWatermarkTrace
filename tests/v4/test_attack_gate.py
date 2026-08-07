import pytest

from tests.v4.benchmark_manifest import ATTACKS, GateFailure, evaluate_release_report, passing_report


def test_every_required_attack_has_evidence_and_zero_wrong_trace() -> None:
    report = passing_report(); del report["suites"]["attacks"][next(iter(ATTACKS))]
    with pytest.raises(GateFailure, match="attack"):
        evaluate_release_report(report)
    report = passing_report(); report["suites"]["attacks"]["crop"]["wrong_traces"] = 1
    with pytest.raises(GateFailure, match="wrong trace"):
        evaluate_release_report(report)
