import hashlib
import json
import re
import struct
import subprocess
import sys
import zlib
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tests.commercial_dataset_manifest import (
    load_manifest,
    manifest_counts,
    validate_negative_manifest,
    validate_route_manifest,
)


NEGATIVE_PREFIX = "tests/fixtures/commercial/samples/negative/"
ROUTE_PREFIX = "tests/fixtures/commercial/samples/real-platform/"
SOURCE_PREFIX = ROUTE_PREFIX + "source/"
RECEIVED_PREFIX = ROUTE_PREFIX + "received/"
ALLOWED_CATEGORIES = (
    "photo",
    "illustration",
    "ui",
    "low_texture",
    "high_texture",
    "similar_composition",
)


def negative_manifest(status: str = "pending_collection") -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "negative-test",
        "target_slots": 1,
        "samples": [
            {
                "id": "negative-0001",
                "category": "photo",
                "relative_path": f"{NEGATIVE_PREFIX}negative-0001.png",
                "sha256": None if status == "pending_collection" else "0" * 64,
                "status": status,
            }
        ],
    }


def route_manifest() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "routes-test",
        "routes": [
            {
                "source_id": f"source-{route}",
                "route": route,
                "attempt": 1,
                "sent_at": None,
                "received_at": None,
                "source_relative_path": f"{SOURCE_PREFIX}source-{route}.png",
                "output_relative_path": (
                    f"{RECEIVED_PREFIX}source-{route}--{route}--attempt-001.png"
                ),
                "source_sha256": None,
                "received_sha256": None,
                "status": "pending_collection",
                "operator": "",
                "device": "",
                "software": "",
                "software_version": "",
                "account_channel": "",
                "notes": "",
                "reviewer": "",
                "rejection_reason": "",
            }
            for route in ("wechat", "browser", "target_platform")
        ],
    }


def image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (24, 96, 160)).save(buffer, format=image_format)
    return buffer.getvalue()


def oversized_png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def collect_negative(data: dict, root: Path, image_format: str = "PNG") -> bytes:
    sample = data["samples"][0]
    content = image_bytes(image_format)
    evidence = root / sample["relative_path"]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(content)
    sample.update(status="collected", sha256=hashlib.sha256(content).hexdigest())
    return content


def collect_route(data: dict, root: Path, index: int = 0) -> dict:
    record = data["routes"][index]
    source_bytes = image_bytes()
    received_bytes = image_bytes()
    source = root / record["source_relative_path"]
    output = root / record["output_relative_path"]
    source.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_bytes)
    output.write_bytes(received_bytes)
    record.update(
        status="collected",
        sent_at="2026-07-13T10:20:30+08:00",
        received_at="2026-07-13T10:21:00+08:00",
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        received_sha256=hashlib.sha256(received_bytes).hexdigest().upper(),
        operator="qa-operator",
        device="Windows 11 workstation",
        software="WeChat",
        software_version="4.0.3",
        account_channel="approved-test-channel",
        notes="downloaded original output",
        reviewer="qa-reviewer",
        rejection_reason="",
    )
    return record


def test_load_manifest_reads_an_object_and_rejects_invalid_json(tmp_path: Path):
    valid = tmp_path / "valid.json"
    valid.write_text('{"schema_version": 1}', encoding="utf-8")
    assert load_manifest(valid) == {"schema_version": 1}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_manifest(malformed)


@pytest.mark.parametrize("value", [[], "manifest", 1, None])
def test_load_manifest_rejects_non_object_json(tmp_path: Path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_manifest(path)


def test_negative_manifest_accepts_pending_missing_file_and_valid_collected_image(tmp_path: Path):
    pending = negative_manifest()
    assert validate_negative_manifest(pending, expected_slots=1, root_path=tmp_path) == []

    collected = negative_manifest("collected")
    collect_negative(collected, tmp_path)
    assert validate_negative_manifest(collected, expected_slots=1, root_path=tmp_path) == []


def test_pending_negative_requires_explicit_null_sha256():
    data = negative_manifest()
    del data["samples"][0]["sha256"]
    assert validate_negative_manifest(data, 1) == ["samples[0].sha256 is required"]

    data["samples"][0]["sha256"] = "0" * 64
    assert validate_negative_manifest(data, 1) == [
        "samples[0].sha256 must be null for pending sample"
    ]


def test_negative_manifest_reports_duplicate_ids_bad_categories_and_count_mismatches_in_order():
    data = negative_manifest()
    data["target_slots"] = 3
    data["samples"] = [data["samples"][0], deepcopy(data["samples"][0])]
    data["samples"][1]["category"] = "photograph"

    errors = validate_negative_manifest(data, expected_slots=1)

    assert errors == [
        "target_slots must equal expected_slots (1)",
        "samples length must equal expected_slots (1)",
        "samples[1].id duplicates 'negative-0001'",
        "samples[1].category must be one of: photo, illustration, ui, low_texture, high_texture, similar_composition",
        "samples[1].relative_path duplicates normalized path "
        "'tests/fixtures/commercial/samples/negative/negative-0001.png'",
    ]


@pytest.mark.parametrize(
    "bad_path",
    [
        "/tmp/image.png",
        "C:/image.png",
        r"tests\fixtures\commercial\samples\negative\image.png",
        "tests/fixtures/commercial/samples/negative/../image.png",
        "elsewhere/image.png",
    ],
)
def test_negative_manifest_rejects_unsafe_or_out_of_dataset_paths(bad_path: str):
    data = negative_manifest()
    data["samples"][0]["relative_path"] = bad_path
    errors = validate_negative_manifest(data, expected_slots=1)
    assert errors == ["samples[0].relative_path must be a safe POSIX path under " + NEGATIVE_PREFIX.rstrip("/")]


def test_negative_manifest_rejects_collected_missing_file(tmp_path: Path):
    errors = validate_negative_manifest(negative_manifest("collected"), 1, tmp_path)
    assert errors == ["samples[0].relative_path does not exist for collected sample"]


def test_collected_negative_rejects_zero_byte_file(tmp_path: Path):
    data = negative_manifest("collected")
    evidence = tmp_path / data["samples"][0]["relative_path"]
    evidence.parent.mkdir(parents=True)
    evidence.touch()
    data["samples"][0]["sha256"] = hashlib.sha256(b"").hexdigest()
    assert validate_negative_manifest(data, 1, tmp_path) == [
        "samples[0].relative_path is empty for collected sample"
    ]


def test_collected_negative_requires_valid_matching_sha256(tmp_path: Path):
    data = negative_manifest("collected")
    collect_negative(data, tmp_path)
    data["samples"][0]["sha256"] = "not-a-sha256"
    assert "samples[0].sha256 must be 64 hexadecimal characters for collected sample" in (
        validate_negative_manifest(data, 1, tmp_path)
    )

    data["samples"][0]["sha256"] = "0" * 64
    assert "samples[0].sha256 does not match file bytes" in validate_negative_manifest(
        data, 1, tmp_path
    )


def test_collected_negative_rejects_corrupt_image_even_when_hash_matches(tmp_path: Path):
    data = negative_manifest("collected")
    corrupt = b"not a decodable image"
    evidence = tmp_path / data["samples"][0]["relative_path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(corrupt)
    data["samples"][0]["sha256"] = hashlib.sha256(corrupt).hexdigest()
    assert "samples[0].relative_path is not a decodable image" in validate_negative_manifest(
        data, 1, tmp_path
    )


def test_collected_negative_rejects_image_format_disguised_by_suffix(tmp_path: Path):
    data = negative_manifest("collected")
    content = collect_negative(data, tmp_path, "JPEG")
    data["samples"][0]["sha256"] = hashlib.sha256(content).hexdigest()
    assert (
        "samples[0].relative_path decoded format JPEG does not match suffix .png"
        in validate_negative_manifest(data, 1, tmp_path)
    )


@pytest.mark.parametrize("dimension", [10_000, 20_000])
def test_collected_negative_converts_decompression_bombs_to_validation_errors(
    tmp_path: Path, dimension: int
):
    data = negative_manifest("collected")
    content = oversized_png_header(dimension, dimension)
    evidence = tmp_path / data["samples"][0]["relative_path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(content)
    data["samples"][0]["sha256"] = hashlib.sha256(content).hexdigest()

    assert validate_negative_manifest(data, 1, tmp_path) == [
        "samples[0].relative_path exceeds Pillow image pixel limit"
    ]


@pytest.mark.parametrize(("image_format", "suffix"), [("JPEG", ".jpg"), ("TIFF", ".tiff")])
def test_collected_negative_reopens_and_loads_verified_truncated_images(
    tmp_path: Path, image_format: str, suffix: str
):
    data = negative_manifest("collected")
    sample = data["samples"][0]
    sample["relative_path"] = NEGATIVE_PREFIX + "negative-0001" + suffix
    content = image_bytes(image_format)[:-1]
    evidence = tmp_path / sample["relative_path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(content)
    sample["sha256"] = hashlib.sha256(content).hexdigest()

    assert validate_negative_manifest(data, 1, tmp_path) == [
        "samples[0].relative_path is not a decodable image"
    ]


def test_collected_negative_hashes_in_chunks_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = negative_manifest("collected")
    collect_negative(data, tmp_path)

    def reject_read_bytes(_path):
        raise AssertionError("validator must stream evidence files")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert validate_negative_manifest(data, 1, tmp_path) == []


def test_negative_manifest_requires_supported_image_extension():
    data = negative_manifest()
    data["samples"][0]["relative_path"] = NEGATIVE_PREFIX + "negative-0001.gif"
    assert validate_negative_manifest(data, 1) == [
        "samples[0].relative_path must use a supported image extension"
    ]


def test_negative_manifest_rejects_case_insensitive_normalized_duplicate_paths():
    data = negative_manifest()
    duplicate = deepcopy(data["samples"][0])
    duplicate["id"] = "negative-0002"
    duplicate["relative_path"] = (
        "tests/fixtures/commercial/samples/negative/./NEGATIVE-0001.PNG"
    )
    data["target_slots"] = 2
    data["samples"].append(duplicate)

    assert validate_negative_manifest(data, 2) == [
        "samples[1].relative_path duplicates normalized path "
        "'tests/fixtures/commercial/samples/negative/./NEGATIVE-0001.PNG'"
    ]


def test_negative_manifest_allows_exactly_the_six_categories():
    for category in ALLOWED_CATEGORIES:
        data = negative_manifest()
        data["samples"][0]["category"] = category
        assert validate_negative_manifest(data, 1) == []


def test_route_manifest_requires_coverage_and_unique_source_route_attempts():
    data = route_manifest()
    data["routes"][2]["route"] = "browser"
    data["routes"][2]["source_id"] = data["routes"][1]["source_id"]
    data["routes"][2]["source_relative_path"] = data["routes"][1]["source_relative_path"]
    data["routes"][2]["output_relative_path"] = data["routes"][1]["output_relative_path"]

    errors = validate_route_manifest(data)

    assert "routes must include route 'target_platform'" in errors
    assert (
        "routes[2] duplicates source_id/route/attempt ('source-browser', 'browser', 1)"
        in errors
    )


@pytest.mark.parametrize(
    "bad_path",
    ["/tmp/output.png", "D:/output.png", r"outputs\image.png", "outputs/../image.png"],
)
def test_route_manifest_rejects_unsafe_output_paths(bad_path: str):
    data = route_manifest()
    data["routes"][0]["output_relative_path"] = bad_path
    assert (
        "routes[0].output_relative_path must be a safe POSIX path under " + RECEIVED_PREFIX.rstrip("/")
        in validate_route_manifest(data)
    )


def test_route_manifest_requires_exact_evidence_prefix():
    data = route_manifest()
    data["routes"][0]["output_relative_path"] = "tests/fixtures/commercial/routes/wechat.png"
    assert (
        "routes[0].output_relative_path must be a safe POSIX path under " + RECEIVED_PREFIX.rstrip("/")
        in validate_route_manifest(data)
    )


def test_route_manifest_enforces_attempt_aware_received_basename():
    data = route_manifest()
    data["routes"][0]["output_relative_path"] = RECEIVED_PREFIX + "unmapped-name.png"
    assert (
        "routes[0].output_relative_path basename must equal "
        "'source-wechat--wechat--attempt-001.<image-ext>'"
        in validate_route_manifest(data)
    )


def test_route_manifest_enforces_source_path_mapping_and_positive_attempt():
    data = route_manifest()
    record = data["routes"][0]
    record["attempt"] = 0
    record["source_relative_path"] = SOURCE_PREFIX + "wrong-source.png"
    errors = validate_route_manifest(data)
    assert "routes[0].attempt must be a positive integer" in errors
    assert (
        "routes[0].source_relative_path basename must equal 'source-wechat.<image-ext>'"
        in errors
    )


@pytest.mark.parametrize("field", ["source_relative_path", "output_relative_path"])
def test_route_manifest_requires_image_artifact_extensions(field: str):
    data = route_manifest()
    data["routes"][0][field] = data["routes"][0][field].removesuffix(".png") + ".txt"
    assert f"routes[0].{field} must use a supported image extension" in validate_route_manifest(data)


def test_collected_route_accepts_matching_file_hashes_and_complete_metadata(tmp_path: Path):
    data = route_manifest()
    collect_route(data, tmp_path)
    assert validate_route_manifest(data, tmp_path) == []


def test_collected_route_accepts_chronology_across_timezone_offsets(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record["sent_at"] = "2026-07-13T10:20:30+08:00"
    record["received_at"] = "2026-07-12T22:30:00-04:00"
    assert validate_route_manifest(data, tmp_path) == []


def test_collected_route_rejects_received_before_sent_across_timezone_offsets(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record["sent_at"] = "2026-07-13T10:20:30+08:00"
    record["received_at"] = "2026-07-13T02:19:00Z"
    assert "routes[0].received_at must be at or after sent_at for collected route" in (
        validate_route_manifest(data, tmp_path)
    )


@pytest.mark.parametrize("field", ["account_channel", "notes"])
def test_collected_route_requires_nonempty_audit_context(field: str, tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record[field] = ""
    assert f"routes[0].{field} must be nonempty for collected route" in validate_route_manifest(
        data, tmp_path
    )


def test_collected_route_rejects_missing_files_and_incomplete_metadata(tmp_path: Path):
    data = route_manifest()
    record = data["routes"][0]
    record.update(
        status="collected",
        sent_at="yesterday",
        received_at="",
        source_sha256="0" * 64,
        received_sha256="1" * 64,
        operator="",
        device="",
        software="",
        software_version="",
        reviewer="",
    )
    errors = validate_route_manifest(data, tmp_path)
    for expected in (
        "routes[0].sent_at must be a nonempty RFC3339 timestamp for collected route",
        "routes[0].received_at must be a nonempty RFC3339 timestamp for collected route",
        "routes[0].operator must be nonempty for collected route",
        "routes[0].device must be nonempty for collected route",
        "routes[0].software must be nonempty for collected route",
        "routes[0].software_version must be nonempty for collected route",
        "routes[0].reviewer must be nonempty for collected route",
        "routes[0].source_relative_path does not exist for collected route",
        "routes[0].output_relative_path does not exist for collected route",
    ):
        assert expected in errors


def test_collected_route_rejects_malformed_and_mismatched_hashes(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record["source_sha256"] = "not-a-sha256"
    record["received_sha256"] = "0" * 64
    errors = validate_route_manifest(data, tmp_path)
    assert "routes[0].source_sha256 must be 64 hexadecimal characters for collected route" in errors
    assert "routes[0].received_sha256 does not match output file bytes" in errors


def test_collected_route_rejects_valid_sha256_values_that_do_not_match_files(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record["source_sha256"] = "0" * 64
    record["received_sha256"] = "1" * 64
    errors = validate_route_manifest(data, tmp_path)
    assert "routes[0].source_sha256 does not match source file bytes" in errors
    assert "routes[0].received_sha256 does not match output file bytes" in errors


def test_collected_route_rejects_corrupt_source_image_even_when_hash_matches(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    corrupt = b"not a decodable image"
    (tmp_path / record["source_relative_path"]).write_bytes(corrupt)
    record["source_sha256"] = hashlib.sha256(corrupt).hexdigest()
    assert "routes[0].source_relative_path is not a decodable image" in validate_route_manifest(
        data, tmp_path
    )


def test_collected_route_rejects_received_image_format_disguised_by_suffix(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    disguised = image_bytes("JPEG")
    (tmp_path / record["output_relative_path"]).write_bytes(disguised)
    record["received_sha256"] = hashlib.sha256(disguised).hexdigest()
    assert (
        "routes[0].output_relative_path decoded format JPEG does not match suffix .png"
        in validate_route_manifest(data, tmp_path)
    )


def test_collected_route_hashes_in_chunks_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = route_manifest()
    collect_route(data, tmp_path)

    def reject_read_bytes(_path):
        raise AssertionError("validator must stream evidence files")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert validate_route_manifest(data, tmp_path) == []


def test_collected_route_requires_an_independent_reviewer(tmp_path: Path):
    data = route_manifest()
    record = collect_route(data, tmp_path)
    record["reviewer"] = record["operator"]
    assert "routes[0].reviewer must differ from operator for collected route" in validate_route_manifest(
        data, tmp_path
    )


def test_pending_route_allows_null_evidence_and_empty_metadata():
    assert validate_route_manifest(route_manifest()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sent_at", "2026-07-13T10:20:30+08:00"),
        ("received_at", "2026-07-13T10:21:00+08:00"),
        ("operator", "qa-operator"),
        ("received_sha256", "1" * 64),
        ("device", "test-device"),
        ("software", "WeChat"),
        ("software_version", "4.0.3"),
        ("account_channel", "approved-test-channel"),
        ("reviewer", "qa-reviewer"),
    ],
)
def test_pending_route_with_partial_evidence_requires_rejection_record(field: str, value):
    data = route_manifest()
    data["routes"][0][field] = value
    errors = validate_route_manifest(data)
    assert "routes[0].rejection_reason must be nonempty when pending route contains evidence" in errors
    assert "routes[0].notes must be nonempty when pending route contains evidence" in errors


def test_pending_route_with_existing_received_file_requires_rejection_record(tmp_path: Path):
    data = route_manifest()
    evidence = tmp_path / data["routes"][0]["output_relative_path"]
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(image_bytes())
    errors = validate_route_manifest(data, tmp_path)
    assert "routes[0].rejection_reason must be nonempty when pending route contains evidence" in errors
    assert "routes[0].notes must be nonempty when pending route contains evidence" in errors


def test_pending_route_with_prepared_source_hash_and_file_is_untouched(tmp_path: Path):
    data = route_manifest()
    source = tmp_path / data["routes"][0]["source_relative_path"]
    source.parent.mkdir(parents=True)
    source_bytes = image_bytes()
    source.write_bytes(source_bytes)
    data["routes"][0]["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    assert validate_route_manifest(data, tmp_path) == []


def test_committed_pending_routes_share_prepared_source_without_starting_attempts(tmp_path: Path):
    manifest_path = (
        Path(__file__).parent
        / "fixtures"
        / "commercial"
        / "manifests"
        / "real-platform-routes.json"
    )
    data = load_manifest(manifest_path)
    source = tmp_path / data["routes"][0]["source_relative_path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(image_bytes())

    assert validate_route_manifest(data, tmp_path) == []

    data["routes"][1].update(
        sent_at="2026-07-13T10:20:30+08:00",
        operator="qa-operator",
    )
    assert validate_route_manifest(data, tmp_path) == [
        "routes[1].rejection_reason must be nonempty when pending route contains evidence",
        "routes[1].notes must be nonempty when pending route contains evidence",
    ]


def test_pending_route_with_partial_evidence_accepts_complete_rejection_record():
    data = route_manifest()
    data["routes"][0].update(
        sent_at="2026-07-13T10:20:30+08:00",
        rejection_reason="upload failed after send",
        notes="platform did not produce a received artifact",
    )
    assert validate_route_manifest(data) == []


def test_route_records_require_every_schema_field():
    data = route_manifest()
    required_fields = set(data["routes"][0])
    for field in required_fields:
        candidate = deepcopy(data)
        del candidate["routes"][0][field]
        assert f"routes[0].{field} is required" in validate_route_manifest(candidate)


@pytest.mark.parametrize("field", ["operator", "device", "software", "software_version", "account_channel", "notes", "reviewer", "rejection_reason"])
def test_pending_route_requires_string_metadata(field: str):
    data = route_manifest()
    data["routes"][0][field] = {}
    assert f"routes[0].{field} must be a string" in validate_route_manifest(data)


@pytest.mark.parametrize("field", ["source_sha256", "received_sha256"])
def test_pending_route_hashes_must_be_null_or_sha256(field: str):
    data = route_manifest()
    data["routes"][0][field] = "bad"
    assert f"routes[0].{field} must be null or 64 hexadecimal characters" in validate_route_manifest(data)


def test_uncertain_sample_stays_pending_with_reason_and_notes():
    data = route_manifest()
    record = data["routes"][0]
    record.update(notes="download was interrupted", rejection_reason="received file is corrupt")
    assert validate_route_manifest(data) == []

    record["status"] = "collected"
    errors = validate_route_manifest(data)
    assert "routes[0].rejection_reason must be empty for collected route" in errors

    record["status"] = "rejected"
    assert "routes[0].status must be one of: pending_collection, collected" in validate_route_manifest(data)


def test_pending_rejection_reason_requires_explanatory_notes():
    data = route_manifest()
    data["routes"][0]["rejection_reason"] = "route could not be confirmed"
    assert "routes[0].notes must be nonempty when rejection_reason is nonempty" in validate_route_manifest(data)


@pytest.mark.parametrize("bad_value", [[], {}, 7, True, None])
@pytest.mark.parametrize("field", ["id", "category", "relative_path", "sha256", "status"])
def test_negative_validator_is_total_for_nested_json_types(field: str, bad_value):
    data = negative_manifest()
    data["samples"][0][field] = bad_value
    errors = validate_negative_manifest(data, 1)
    assert isinstance(errors, list)
    if not (field == "sha256" and bad_value is None):
        assert errors


@pytest.mark.parametrize("bad_value", [[], {}, 7, True, None])
@pytest.mark.parametrize(
    "field",
    [
        "source_id",
        "route",
        "attempt",
        "source_relative_path",
        "output_relative_path",
        "source_sha256",
        "received_sha256",
        "status",
        "sent_at",
        "received_at",
        "operator",
        "device",
        "software",
        "software_version",
        "account_channel",
        "notes",
        "reviewer",
        "rejection_reason",
    ],
)
def test_route_validator_is_total_for_nested_json_types(field: str, bad_value):
    data = route_manifest()
    data["routes"][0][field] = bad_value
    errors = validate_route_manifest(data)
    assert isinstance(errors, list)


@pytest.mark.parametrize("value", [None, [], "manifest", 4, True, {"samples": {}}, {"routes": {}}])
def test_manifest_counts_is_total_for_json_values(value):
    assert manifest_counts(value) == {"slots": 0, "collected": 0, "pending": 0}


def test_collected_negative_rejects_symlink_escape(tmp_path_factory):
    root = tmp_path_factory.mktemp("manifest-root")
    outside = tmp_path_factory.mktemp("manifest-outside")
    evidence = outside / "evidence.png"
    evidence.touch()
    prefix = root / NEGATIVE_PREFIX
    prefix.mkdir(parents=True)
    link = prefix / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation denied by OS: {exc}")

    data = negative_manifest("collected")
    data["samples"][0]["relative_path"] = NEGATIVE_PREFIX + "escape/evidence.png"
    assert validate_negative_manifest(data, 1, root) == [
        "samples[0].relative_path resolves outside root_path or expected prefix"
    ]


def test_collected_route_rejects_symlink_escape(tmp_path_factory):
    root = tmp_path_factory.mktemp("route-root")
    outside = tmp_path_factory.mktemp("route-outside")
    evidence = outside / "evidence.png"
    evidence.touch()
    route_prefix = RECEIVED_PREFIX
    prefix = root / route_prefix
    prefix.mkdir(parents=True)
    link = prefix / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation denied by OS: {exc}")

    data = route_manifest()
    record = data["routes"][0]
    record.update(
        status="collected",
        sent_at="2026-07-13T10:20:30+08:00",
        received_at="2026-07-13T02:21:00Z",
        source_sha256="0" * 64,
        received_sha256=hashlib.sha256(b"evidence").hexdigest(),
        operator="qa-operator",
        device="test-device",
        software="test-software",
        software_version="1.0",
        reviewer="qa-reviewer",
        output_relative_path=route_prefix + "escape/evidence.png",
    )
    assert (
        "routes[0].output_relative_path resolves outside root_path or expected prefix"
        in validate_route_manifest(data, root)
    )


def test_manifest_counts_counts_statuses_without_treating_pending_as_collected():
    data = {"samples": [{"status": "pending_collection"}, {"status": "collected"}, {"status": "unknown"}]}
    assert manifest_counts(data) == {"slots": 3, "collected": 1, "pending": 1}


def test_committed_negative_manifests_have_exact_counts_and_development_is_release_prefix():
    fixture_dir = Path(__file__).parent / "fixtures" / "commercial" / "manifests"
    development = load_manifest(fixture_dir / "negative-development.json")
    release = load_manifest(fixture_dir / "negative-release.json")

    assert validate_negative_manifest(development, 100) == []
    assert validate_negative_manifest(release, 300) == []
    assert manifest_counts(development) == {"slots": 100, "collected": 0, "pending": 100}
    assert manifest_counts(release) == {"slots": 300, "collected": 0, "pending": 300}
    assert development["samples"] == release["samples"][:100]
    assert [sample["id"] for sample in release["samples"]] == [
        f"negative-{number:04d}" for number in range(1, 301)
    ]
    assert [sample["category"] for sample in release["samples"]] == [
        ALLOWED_CATEGORIES[index % len(ALLOWED_CATEGORIES)] for index in range(300)
    ]
    assert [sample["relative_path"] for sample in release["samples"]] == [
        f"{NEGATIVE_PREFIX}negative-{number:04d}.png" for number in range(1, 301)
    ]
    assert len({sample["relative_path"].casefold() for sample in release["samples"]}) == 300
    assert all(sample["sha256"] is None for sample in development["samples"])
    assert all(sample["sha256"] is None for sample in release["samples"])


def test_committed_route_manifest_is_valid_and_covers_required_routes():
    path = Path(__file__).parent / "fixtures" / "commercial" / "manifests" / "real-platform-routes.json"
    data = load_manifest(path)
    assert validate_route_manifest(data) == []
    assert manifest_counts(data) == {"slots": 3, "collected": 0, "pending": 3}
    for record in data["routes"]:
        assert record["source_relative_path"] == (
            SOURCE_PREFIX + record["source_id"] + Path(record["source_relative_path"]).suffix
        )
        assert re.fullmatch(
            re.escape(RECEIVED_PREFIX)
            + re.escape(record["source_id"])
            + "--"
            + re.escape(record["route"])
            + rf"--attempt-{record['attempt']:03d}\.(?:png|jpe?g|webp|bmp|tiff?)",
            record["output_relative_path"],
            flags=re.IGNORECASE,
        )


def test_route_manifest_cli_accepts_valid_and_rejects_invalid_manifest(tmp_path: Path):
    manifest = tmp_path / "routes.json"
    manifest.write_text(json.dumps(route_manifest()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "tests.commercial_dataset_manifest",
        str(manifest),
        "--kind",
        "routes",
        "--root",
        str(tmp_path),
    ]
    valid = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert valid.returncode == 0, valid.stderr
    assert '"slots": 3' in valid.stdout
    assert '"pending": 3' in valid.stdout

    data = route_manifest()
    data["routes"][0]["attempt"] = 0
    manifest.write_text(json.dumps(data), encoding="utf-8")
    invalid = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert invalid.returncode == 1
    assert "routes[0].attempt must be a positive integer" in invalid.stderr
    assert "routes[0].attempt must be a positive integer" not in invalid.stdout


def test_real_sample_intake_document_has_operator_contract():
    path = Path(__file__).parents[1] / "docs" / "commercial" / "real-sample-intake.md"
    assert path.is_file(), "real propagation evidence intake procedure must exist"
    content = path.read_text(encoding="utf-8")

    required_headings = (
        "## 采集前准备",
        "## 路由操作步骤",
        "### wechat",
        "### browser",
        "### target_platform",
        "## 文件保全与命名",
        "## SHA-256 计算与记录",
        "## 证据记录字段",
        "## 状态更新与拒收处理",
        "## 证据保管链与复核清单",
        "## 隐私与安全",
    )
    for heading in required_headings:
        assert heading in content

    critical_terms = (
        "发送/上传",
        "接收/下载/截图",
        "不可变源文件",
        "不可变接收文件",
        "禁止覆盖原件",
        "禁止图像编辑",
        "禁止重新压缩",
        "禁止清理元数据",
        "禁止重命名",
        "独立工作副本",
        "operator",
        "sent_at",
        "received_at",
        "route",
        "source_id",
        "output_relative_path",
        "设备名称和版本",
        "应用名称和版本",
        "浏览器名称和版本",
        "平台名称和版本",
        "账号/频道",
        "notes",
        "tests/fixtures/commercial/samples/real-platform/",
        "相对 POSIX 路径",
        "禁止绝对路径",
        "禁止密钥",
        "source_id--route--attempt-NNN",
        "Get-FileHash -Algorithm SHA256 tests/fixtures/commercial/samples/real-platform/source/route-source-0001.png",
        "Get-FileHash -Algorithm SHA256 tests/fixtures/commercial/samples/real-platform/received/route-source-0001--wechat--attempt-001.png",
        'python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path(\'tests/fixtures/commercial/samples/real-platform/source/route-source-0001.png\').read_bytes()).hexdigest())"',
        'python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path(\'tests/fixtures/commercial/samples/real-platform/received/route-source-0001--wechat--attempt-001.png\').read_bytes()).hexdigest())"',
        "PowerShell 与 Python 的结果必须一致",
        "source_sha256",
        "received_sha256",
        "real-platform-routes.json",
        "status: collected",
        "validator passes",
        "pending_collection",
        "拒收原因",
        "不得计入 collected/pass/fail evidence",
        "模拟样本不得标记为真实路由证据",
        "单独报告",
        "独立第二人复核",
        "批准的测试账号",
        "批准的测试内容",
        "禁止凭据、令牌、聊天导出和私人个人内容",
        "Copy-Item tests/fixtures/commercial/manifests/real-platform-routes.json tests/fixtures/commercial/manifests/real-platform-routes.working.json",
        "python -m tests.commercial_dataset_manifest tests/fixtures/commercial/manifests/real-platform-routes.working.json --kind routes --root .",
        "os.replace('tests/fixtures/commercial/manifests/real-platform-routes.working.json', 'tests/fixtures/commercial/manifests/real-platform-routes.json')",
        "官方清单保持 `pending_collection`",
        "新的拒收工作副本",
        "status` 仍为 `pending_collection`",
        "Copy-Item tests/fixtures/commercial/manifests/real-platform-routes.json tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json",
        "python -m tests.commercial_dataset_manifest tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json --kind routes --root .",
        "os.replace('tests/fixtures/commercial/manifests/real-platform-routes.rejection.working.json', 'tests/fixtures/commercial/manifests/real-platform-routes.json')",
    )
    for term in critical_terms:
        assert term in content


def test_real_sample_intake_document_contains_complete_json_record_template():
    path = Path(__file__).parents[1] / "docs" / "commercial" / "real-sample-intake.md"
    content = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    assert match, "procedure must contain an exact JSON record template"
    record = json.loads(match.group(1))
    assert set(record) == {
        "source_id",
        "route",
        "attempt",
        "sent_at",
        "received_at",
        "source_relative_path",
        "output_relative_path",
        "source_sha256",
        "received_sha256",
        "status",
        "operator",
        "device",
        "software",
        "software_version",
        "account_channel",
        "notes",
        "reviewer",
        "rejection_reason",
    }
    assert record["source_relative_path"] == SOURCE_PREFIX + "route-source-0001.png"
    assert record["output_relative_path"] == (
        RECEIVED_PREFIX + "route-source-0001--wechat--attempt-001.png"
    )
