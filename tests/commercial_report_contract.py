import datetime as dt
import os
import platform as runtime_platform
import re


CONFIGURATION_ENV_ALLOWLIST = (
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

CANONICAL_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)


def _is_canonical_utc_timestamp(value: str) -> bool:
    if CANONICAL_UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def build_report_metadata(benchmark: str, seed: int, algorithm_version: str) -> dict:
    if not isinstance(benchmark, str):
        raise TypeError("benchmark must be a string")
    if not benchmark.strip():
        raise ValueError("benchmark must be a nonempty string")
    if not isinstance(algorithm_version, str):
        raise TypeError("algorithm_version must be a string")
    if not algorithm_version.strip():
        raise ValueError("algorithm_version must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    configuration = {
        name: os.environ[name]
        for name in CONFIGURATION_ENV_ALLOWLIST
        if name in os.environ
    }
    generated_at = (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "algorithm_version": algorithm_version,
        "seed": seed,
        "generated_at": generated_at,
        "python_version": runtime_platform.python_version(),
        "platform": runtime_platform.platform(),
        "configuration": configuration,
    }


def validate_report(report) -> list[str]:
    if not isinstance(report, dict):
        return ["report must be an object"]

    errors = []
    required_fields = (
        "metadata",
        "summary",
        "cases",
        "settings",
        "verdict",
        "failed_gates",
    )

    for field in required_fields:
        if field not in report:
            errors.append(f"missing {field}")

    if "metadata" in report:
        metadata = report["metadata"]
        if not isinstance(metadata, dict):
            errors.append("metadata must be an object")
        else:
            metadata_fields = (
                "schema_version",
                "benchmark",
                "algorithm_version",
                "seed",
                "generated_at",
                "python_version",
                "platform",
                "configuration",
            )
            for field in metadata_fields:
                if field not in metadata:
                    errors.append(f"missing metadata.{field}")
                elif field == "schema_version" and (
                    type(metadata[field]) is not int or metadata[field] != 1
                ):
                    errors.append("invalid schema_version")
                elif field in (
                    "benchmark",
                    "algorithm_version",
                    "generated_at",
                    "python_version",
                    "platform",
                ) and (
                    not isinstance(metadata[field], str) or not metadata[field].strip()
                ):
                    errors.append(f"metadata.{field} must be a nonempty string")
                elif field == "generated_at" and not _is_canonical_utc_timestamp(
                    metadata[field]
                ):
                    errors.append(
                        "metadata.generated_at must be a canonical UTC timestamp"
                    )
                elif field == "seed" and type(metadata[field]) is not int:
                    errors.append("metadata.seed must be an integer")
                elif field == "configuration" and not isinstance(
                    metadata[field], dict
                ):
                    errors.append("metadata.configuration must be an object")
                elif field == "configuration":
                    for name, value in metadata[field].items():
                        if not isinstance(name, str):
                            errors.append(
                                "metadata.configuration key must be a string"
                            )
                            if not isinstance(value, str):
                                errors.append(
                                    "metadata.configuration value must be a string"
                                )
                            continue
                        if name not in CONFIGURATION_ENV_ALLOWLIST:
                            errors.append(
                                f"metadata.configuration contains unknown key {name!r}"
                            )
                        if not isinstance(value, str):
                            errors.append(
                                f"metadata.configuration.{name} must be a string"
                            )

    if "summary" in report and not isinstance(report["summary"], dict):
        errors.append("summary must be an object")

    if "cases" in report and not isinstance(report["cases"], list):
        errors.append("cases must be a list")

    if "settings" in report and not isinstance(report["settings"], dict):
        errors.append("settings must be an object")

    verdict_is_valid = "verdict" in report and report["verdict"] in (
        "PASS",
        "FAIL",
    )
    if "verdict" in report and not verdict_is_valid:
        errors.append("verdict must be PASS or FAIL")

    gates_are_valid = "failed_gates" in report and isinstance(
        report["failed_gates"], list
    ) and all(isinstance(gate, str) for gate in report["failed_gates"])
    if "failed_gates" in report and not gates_are_valid:
        errors.append("failed_gates must be a list of strings")

    if verdict_is_valid and gates_are_valid:
        if report["verdict"] == "PASS" and report["failed_gates"]:
            errors.append("PASS verdict requires no failed gates")
        elif report["verdict"] == "FAIL" and not report["failed_gates"]:
            errors.append("FAIL verdict requires at least one failed gate")

    return errors
