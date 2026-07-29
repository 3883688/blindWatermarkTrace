import pytest

from tests.v4.benchmark_manifest import GateFailure, evaluate_release_report, passing_report


def test_scale_gate_requires_1000_same_source_versions_with_stable_indexed_lookup() -> None:
    report = passing_report(); report["suites"]["scale"]["same_source_versions"] = 999
    with pytest.raises(GateFailure, match="1000"):
        evaluate_release_report(report)
    report = passing_report(); report["suites"]["scale"]["indexed_lookup"] = False
    with pytest.raises(GateFailure, match="indexed"):
        evaluate_release_report(report)
