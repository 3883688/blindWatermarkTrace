"""Deterministic V4 release evidence contract and thresholds."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from typing import Any, Mapping


ATTACKS = frozenset(
    {"jpeg", "recompression", "crop", "rotation", "screenshot", "screen_photo", "denoise",
     "sharpen", "noise", "pilot_notch", "dct_attenuation", "same_source_collusion", "overwrite"}
)
QUALITY_STRATA = frozenset({"low_texture", "high_texture", "photo", "text", "ui", "synthetic"})


class GateFailure(RuntimeError):
    pass


def passing_report() -> dict[str, Any]:
    return {
        "git_commit": "a" * 40,
        "schema_id": "v4",
        "codec": "hmac64_rs_16_8_split_repeat_sync_v4",
        "model_versions": {"dinov2": "vits14", "superpoint": "pinned", "lightglue": "pinned"},
        "model_health": {"dinov2": True, "superpoint": True, "lightglue": True},
        "manifest_hashes": {"development": ["1" * 64], "release": ["2" * 64], "blind": ["3" * 64]},
        "random_seed": 20260729,
        "hardware": {"cpu": "reference", "gpu": "reference", "ram_bytes": 32_000_000_000},
        "raw_artifact_hashes": ["4" * 64],
        "query_plan_ok": True,
        "sensitive_log_leaks": 0,
        "suites": {
            "recall": {"full": .99, "resize": .99, "jpeg": .99, "crop": .95},
            "attribution": {"rate": .95, "wrong_traces": 0},
            "negative": {"independent_count": 3000, "wrong_attributions": 0},
            "scale": {"same_source_versions": 1000, "indexed_lookup": True},
            "attacks": {name: {"samples": 1, "wrong_traces": 0} for name in ATTACKS},
            "quality": {name: {"psnr": 38.0, "ssim": .95} for name in QUALITY_STRATA},
            "performance": {
                "standard_p95_seconds": 120.0, "standard_max_seconds": 300.0,
                "deep_max_seconds": 1000.0,
                "stage_timings": {"decode": [1.0], "recall": [1.0], "geometry": [1.0], "authentication": [1.0]},
            },
        },
    }


def _require(report: Mapping[str, Any], name: str) -> Any:
    value = report.get(name)
    if value in (None, {}, []):
        raise GateFailure(f"missing {name}")
    return value


def evaluate_release_report(report: Mapping[str, Any]) -> None:
    for field in (
        "git_commit", "schema_id", "codec", "model_versions", "model_health",
        "manifest_hashes", "random_seed", "hardware", "raw_artifact_hashes", "suites",
    ):
        _require(report, field)
    if not all(report["model_health"].get(name) is True for name in ("dinov2", "superpoint", "lightglue")):
        raise GateFailure("model_health failed")
    if not all(report["hardware"].get(name) for name in ("cpu", "gpu", "ram_bytes")):
        raise GateFailure("hardware metadata incomplete")
    manifests = report["manifest_hashes"]
    if set(manifests) != {"development", "release", "blind"} or any(not manifests[name] for name in manifests):
        raise GateFailure("manifest_hashes incomplete")
    flattened = [item for values in manifests.values() for item in values]
    if len(flattened) != len(set(flattened)):
        raise GateFailure("dataset manifests overlap")
    if report.get("query_plan_ok") is not True:
        raise GateFailure("indexed query plan failed")
    if report.get("sensitive_log_leaks") != 0:
        raise GateFailure("sensitive log leak")

    suites = report["suites"]
    recall = _require(suites, "recall")
    for name, threshold in {"full": .99, "resize": .99, "jpeg": .99, "crop": .95}.items():
        if recall.get(name, -1) < threshold:
            raise GateFailure(f"{name} recall below threshold")
    attribution = _require(suites, "attribution")
    if attribution.get("rate", -1) < .95:
        raise GateFailure("final attribution below 95%")
    if attribution.get("wrong_traces") != 0:
        raise GateFailure("wrong trace detected")
    negative = _require(suites, "negative")
    if negative.get("independent_count", 0) < 3000:
        raise GateFailure("negative gate requires 3000 independent samples")
    if negative.get("wrong_attributions") != 0:
        raise GateFailure("wrong attribution in negative gate")
    scale = _require(suites, "scale")
    if scale.get("same_source_versions", 0) < 1000:
        raise GateFailure("scale gate requires 1000 same-source versions")
    if scale.get("indexed_lookup") is not True:
        raise GateFailure("scale lookup must be indexed")
    attacks = _require(suites, "attacks")
    if not ATTACKS.issubset(attacks):
        raise GateFailure("required attack evidence missing")
    if any(attacks[name].get("samples", 0) <= 0 for name in ATTACKS):
        raise GateFailure("required attack samples missing")
    if any(attacks[name].get("wrong_traces") != 0 for name in ATTACKS):
        raise GateFailure("wrong trace in attack gate")
    quality = _require(suites, "quality")
    if not QUALITY_STRATA.issubset(quality):
        raise GateFailure("quality stratum missing")
    for name in QUALITY_STRATA:
        if quality[name].get("psnr", -1) < 38:
            raise GateFailure(f"{name} PSNR below 38")
        if quality[name].get("ssim", -1) < .95:
            raise GateFailure(f"{name} SSIM below 0.95")
    performance = _require(suites, "performance")
    if performance.get("standard_p95_seconds", 1e9) > 120:
        raise GateFailure("standard P95 exceeds 120 seconds")
    if performance.get("standard_max_seconds", 1e9) > 300:
        raise GateFailure("standard request exceeds 300 seconds")
    if performance.get("deep_max_seconds", 1e9) > 1000:
        raise GateFailure("deep job exceeds 1000 seconds")
    _require(performance, "stage_timings")


def sign_report(report: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    if not key:
        raise GateFailure("report signing key is missing")
    signed = copy.deepcopy(dict(report))
    signed.pop("signature", None)
    payload = json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
    signed["signature"] = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return signed


def verify_signed_report(report: Mapping[str, Any], key: bytes, *, git_commit: str) -> None:
    signature = report.get("signature")
    expected = sign_report(report, key)["signature"]
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
        raise GateFailure("release report signature is invalid")
    if report.get("git_commit") != git_commit:
        raise GateFailure("release report is stale")
    evaluate_release_report(report)
