import datetime as dt

import pytest

from tests.commercial_report_contract import (
    CONFIGURATION_ENV_ALLOWLIST,
    build_report_metadata,
    validate_report,
)


EXPECTED_CONFIGURATION_ENV_ALLOWLIST = (
    "FIDELITY_LEVEL",
    "SMALL_CROP_TRACE_STRENGTH",
    "SMALL_CROP_TRACE_DENSITY",
    "ROBUST_WATERMARK_STRENGTH",
    "ROBUST_WATERMARK_VERSION",
    "TRACE_ROUNDS",
    "SCALE_FACTORS",
    "CROP_RATIOS",
    "DETECTION_WORKERS",
    "WATERMARK_DETECTION_BUDGET_SECONDS",
)


def test_build_report_metadata_has_required_identity_without_secret(monkeypatch):
    monkeypatch.setenv("WATERMARK_AUTH_KEY", "must-not-leak")

    metadata = build_report_metadata("trace", seed=20260713, algorithm_version="v3-baseline")

    assert metadata["schema_version"] == 1
    assert metadata["benchmark"] == "trace"
    assert metadata["seed"] == 20260713
    assert metadata["algorithm_version"] == "v3-baseline"
    assert "must-not-leak" not in repr(metadata)


def test_build_report_metadata_has_stable_utc_timestamp_and_runtime_fields():
    metadata = build_report_metadata("quality", seed=7, algorithm_version="v1")

    assert set(metadata) == {
        "schema_version",
        "benchmark",
        "algorithm_version",
        "seed",
        "generated_at",
        "python_version",
        "platform",
        "configuration",
    }
    generated_at = metadata["generated_at"]
    assert generated_at.endswith("Z")
    parsed = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    assert parsed.tzinfo == dt.timezone.utc
    assert parsed.microsecond == 0
    assert metadata["python_version"]
    assert metadata["platform"]


@pytest.mark.parametrize(
    ("benchmark", "algorithm_version", "message"),
    [
        ("", "v1", "benchmark must be a nonempty string"),
        ("trace", "", "algorithm_version must be a nonempty string"),
    ],
)
def test_build_report_metadata_rejects_empty_names(benchmark, algorithm_version, message):
    with pytest.raises(ValueError, match=message):
        build_report_metadata(benchmark, seed=1, algorithm_version=algorithm_version)


@pytest.mark.parametrize(
    ("benchmark", "algorithm_version", "message"),
    [
        (None, "v1", "benchmark must be a string"),
        ("trace", None, "algorithm_version must be a string"),
    ],
)
def test_build_report_metadata_rejects_nonstring_names(
    benchmark, algorithm_version, message
):
    with pytest.raises(TypeError, match=message):
        build_report_metadata(benchmark, seed=1, algorithm_version=algorithm_version)


@pytest.mark.parametrize(
    ("benchmark", "algorithm_version", "message"),
    [
        (" \t", "v1", "benchmark must be a nonempty string"),
        ("trace", " \t", "algorithm_version must be a nonempty string"),
    ],
)
def test_build_report_metadata_rejects_whitespace_names(
    benchmark, algorithm_version, message
):
    with pytest.raises(ValueError, match=message):
        build_report_metadata(benchmark, seed=1, algorithm_version=algorithm_version)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_build_report_metadata_requires_integer_seed(seed):
    with pytest.raises(TypeError, match="seed must be an integer"):
        build_report_metadata("trace", seed=seed, algorithm_version="v1")


def test_configuration_environment_allowlist_is_exact():
    assert CONFIGURATION_ENV_ALLOWLIST == EXPECTED_CONFIGURATION_ENV_ALLOWLIST


@pytest.mark.parametrize("name", EXPECTED_CONFIGURATION_ENV_ALLOWLIST)
def test_build_report_metadata_captures_each_allowlisted_configuration(name, monkeypatch):
    for allowed_name in CONFIGURATION_ENV_ALLOWLIST:
        monkeypatch.delenv(allowed_name, raising=False)
    monkeypatch.setenv(name, f"value-for-{name}")

    metadata = build_report_metadata("trace", seed=1, algorithm_version="v1")

    assert metadata["configuration"] == {name: f"value-for-{name}"}


def test_build_report_metadata_excludes_planted_secrets(monkeypatch):
    for name in EXPECTED_CONFIGURATION_ENV_ALLOWLIST:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WATERMARK_AUTH_KEY", "auth-secret")
    monkeypatch.setenv("ADMIN_PASS", "password-secret")
    monkeypatch.setenv("DB_URL", "db-secret")

    metadata = build_report_metadata("trace", seed=1, algorithm_version="v1")

    assert metadata["configuration"] == {}
    rendered = repr(metadata)
    assert "auth-secret" not in rendered
    assert "password-secret" not in rendered
    assert "db-secret" not in rendered


VALID_METADATA = {
    "schema_version": 1,
    "benchmark": "quality",
    "algorithm_version": "3",
    "seed": 20260713,
    "generated_at": "2026-07-13T00:00:00Z",
    "python_version": "3.13.5",
    "platform": "Windows-11",
    "configuration": {},
}


def valid_report(**updates):
    report = {
        "metadata": dict(VALID_METADATA),
        "summary": {},
        "cases": [],
        "settings": {},
        "verdict": "PASS",
        "failed_gates": [],
    }
    report.update(updates)
    return report


def test_validate_report_lists_missing_top_level_fields_in_schema_order():
    assert validate_report({}) == [
        "missing metadata",
        "missing summary",
        "missing cases",
        "missing settings",
        "missing verdict",
        "missing failed_gates",
    ]


@pytest.mark.parametrize("report", [None, [], True, 1, 1.0, "report"])
def test_validate_report_rejects_non_object_roots(report):
    assert validate_report(report) == ["report must be an object"]


def test_validate_report_lists_missing_metadata_fields_in_schema_order():
    assert validate_report(valid_report(metadata={})) == [
        "missing metadata.schema_version",
        "missing metadata.benchmark",
        "missing metadata.algorithm_version",
        "missing metadata.seed",
        "missing metadata.generated_at",
        "missing metadata.python_version",
        "missing metadata.platform",
        "missing metadata.configuration",
    ]


@pytest.mark.parametrize("schema_version", [None, 2, True, 1.0, "1"])
def test_validate_report_requires_exact_integer_schema_version_one(schema_version):
    metadata = {**VALID_METADATA, "schema_version": schema_version}
    assert validate_report(valid_report(metadata=metadata)) == ["invalid schema_version"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark", None),
        ("benchmark", " \t"),
        ("algorithm_version", 3),
        ("algorithm_version", ""),
        ("generated_at", []),
        ("generated_at", ""),
        ("python_version", False),
        ("python_version", ""),
        ("platform", {}),
        ("platform", " \t"),
    ],
)
def test_validate_report_requires_nonempty_metadata_strings(field, value):
    metadata = {**VALID_METADATA, field: value}
    assert validate_report(valid_report(metadata=metadata)) == [
        f"metadata.{field} must be a nonempty string"
    ]


@pytest.mark.parametrize("seed", [None, True, False, 1.0, "1"])
def test_validate_report_requires_exact_integer_metadata_seed(seed):
    metadata = {**VALID_METADATA, "seed": seed}
    assert validate_report(valid_report(metadata=metadata)) == [
        "metadata.seed must be an integer"
    ]


@pytest.mark.parametrize(
    "generated_at",
    [
        "2026-07-13T00:00:00+00:00",
        "2026-07-13T00:00:00.000Z",
        "2026-7-13T00:00:00Z",
        "2026-07-13 00:00:00Z",
        "2026-02-29T00:00:00Z",
        "2026-07-13T24:00:00Z",
        "not-a-timestamp",
    ],
)
def test_validate_report_requires_canonical_calendar_valid_utc_timestamp(generated_at):
    metadata = {**VALID_METADATA, "generated_at": generated_at}

    assert validate_report(valid_report(metadata=metadata)) == [
        "metadata.generated_at must be a canonical UTC timestamp"
    ]


def test_validate_report_accepts_canonical_utc_timestamp_on_leap_day():
    metadata = {**VALID_METADATA, "generated_at": "2024-02-29T23:59:59Z"}

    assert validate_report(valid_report(metadata=metadata)) == []


def test_validate_report_requires_allowlisted_string_configuration_entries():
    metadata = {
        **VALID_METADATA,
        "configuration": {
            "WATERMARK_AUTH_KEY": "must-not-leak",
            7: ["numeric-key-value"],
            "FIDELITY_LEVEL": 0.9,
            "TRACE_ROUNDS": "6",
        },
    }

    errors = validate_report(valid_report(metadata=metadata))

    assert errors == [
        "metadata.configuration contains unknown key 'WATERMARK_AUTH_KEY'",
        "metadata.configuration key must be a string",
        "metadata.configuration value must be a string",
        "metadata.configuration.FIDELITY_LEVEL must be a string",
    ]
    assert "must-not-leak" not in repr(errors)
    assert "numeric-key-value" not in repr(errors)


def test_validate_report_accepts_all_allowlisted_string_configuration_entries():
    metadata = {
        **VALID_METADATA,
        "configuration": {
            name: f"value-for-{name}" for name in CONFIGURATION_ENV_ALLOWLIST
        },
    }

    assert validate_report(valid_report(metadata=metadata)) == []


def test_validate_report_reports_nested_wrong_types_in_field_order():
    report = valid_report(
        metadata=[],
        summary=[],
        cases={},
        settings=None,
        verdict=1,
        failed_gates="gate",
    )

    assert validate_report(report) == [
        "metadata must be an object",
        "summary must be an object",
        "cases must be a list",
        "settings must be an object",
        "verdict must be PASS or FAIL",
        "failed_gates must be a list of strings",
    ]


def test_validate_report_reports_metadata_content_errors_in_field_order():
    metadata = {
        "schema_version": True,
        "benchmark": "",
        "algorithm_version": None,
        "seed": False,
        "generated_at": [],
        "python_version": {},
        "platform": 1,
        "configuration": [],
    }

    assert validate_report(valid_report(metadata=metadata)) == [
        "invalid schema_version",
        "metadata.benchmark must be a nonempty string",
        "metadata.algorithm_version must be a nonempty string",
        "metadata.seed must be an integer",
        "metadata.generated_at must be a nonempty string",
        "metadata.python_version must be a nonempty string",
        "metadata.platform must be a nonempty string",
        "metadata.configuration must be an object",
    ]


def test_validate_report_interleaves_missing_and_invalid_metadata_fields_in_order():
    metadata = {
        "schema_version": 2,
        "algorithm_version": "",
        "seed": 1,
        "generated_at": "2026-07-13T00:00:00Z",
        "python_version": "3.13.5",
        "platform": "Windows-11",
        "configuration": {},
    }

    assert validate_report(valid_report(metadata=metadata)) == [
        "invalid schema_version",
        "missing metadata.benchmark",
        "metadata.algorithm_version must be a nonempty string",
    ]


@pytest.mark.parametrize("failed_gates", [["gate", 1], [None], [True], {}])
def test_validate_report_requires_failed_gates_to_be_a_list_of_strings(failed_gates):
    assert validate_report(valid_report(failed_gates=failed_gates)) == [
        "failed_gates must be a list of strings"
    ]


@pytest.mark.parametrize("verdict", [None, True, "pass", "UNKNOWN"])
def test_validate_report_rejects_invalid_verdict(verdict):
    assert validate_report(valid_report(verdict=verdict)) == [
        "verdict must be PASS or FAIL"
    ]


def test_validate_report_requires_pass_to_have_zero_failed_gates():
    assert validate_report(valid_report(failed_gates=["quality_min_psnr"])) == [
        "PASS verdict requires no failed gates"
    ]


def test_validate_report_requires_fail_to_have_at_least_one_failed_gate():
    assert validate_report(valid_report(verdict="FAIL")) == [
        "FAIL verdict requires at least one failed gate"
    ]


@pytest.mark.parametrize(
    "report",
    [
        valid_report(),
        valid_report(verdict="FAIL", failed_gates=["crop_0.3_recall"]),
    ],
)
def test_validate_report_accepts_valid_pass_and_fail_reports(report):
    assert validate_report(report) == []
