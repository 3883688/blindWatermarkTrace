import argparse
import hashlib
import json
import posixpath
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TypeAlias

from PIL import Image


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]


NEGATIVE_CATEGORIES = (
    "photo",
    "illustration",
    "ui",
    "low_texture",
    "high_texture",
    "similar_composition",
)
STATUSES = ("pending_collection", "collected")
ROUTES = ("wechat", "browser", "target_platform")
NEGATIVE_PATH_PREFIX = "tests/fixtures/commercial/samples/negative"
ROUTE_PATH_PREFIX = "tests/fixtures/commercial/samples/real-platform"
ROUTE_SOURCE_PATH_PREFIX = ROUTE_PATH_PREFIX + "/source"
ROUTE_RECEIVED_PATH_PREFIX = ROUTE_PATH_PREFIX + "/received"
ROUTE_REQUIRED_FIELDS = (
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
)
ROUTE_STRING_FIELDS = (
    "operator",
    "device",
    "software",
    "software_version",
    "account_channel",
    "notes",
    "reviewer",
    "rejection_reason",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
IMAGE_FORMAT_BY_EXTENSION = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
HASH_CHUNK_SIZE = 1024 * 1024
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def load_manifest(path: Path) -> JSONObject:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in manifest {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")
    return data


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_relative_posix_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if re.match(r"^[A-Za-z]:", value) or value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return value != "." and ".." not in path.parts and not path.is_absolute()


def _safe_prefixed_posix_path(value: object, prefix: str) -> bool:
    return (
        _safe_relative_posix_path(value)
        and isinstance(value, str)
        and value.startswith(prefix + "/")
    )


def _normalized_path(value: str) -> str:
    return posixpath.normpath(value).casefold()


def _collected_path_error(root: Path, relative_path: str, prefix: str) -> str | None:
    resolved_root = root.resolve(strict=False)
    resolved_prefix = (resolved_root / PurePosixPath(prefix)).resolve(strict=False)
    resolved_candidate = (resolved_root / PurePosixPath(relative_path)).resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root) or not resolved_candidate.is_relative_to(
        resolved_prefix
    ):
        return "resolves outside root_path or expected prefix"
    if not resolved_candidate.is_file():
        return "does not exist"
    return None


def _is_rfc3339(value: object) -> bool:
    return _parse_rfc3339(value) is not None


def _parse_rfc3339(value: object) -> datetime | None:
    if not isinstance(value, str) or not _RFC3339.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _image_error(path: Path, suffix: str) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                decoded_format = image.format
                image.verify()
            with Image.open(path) as image:
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return "exceeds Pillow image pixel limit"
    except (OSError, SyntaxError, ValueError):
        return "is not a decodable image"
    expected_format = IMAGE_FORMAT_BY_EXTENSION[suffix.casefold()]
    if decoded_format != expected_format:
        return f"decoded format {decoded_format} does not match suffix {suffix.casefold()}"
    return None


def _collected_image_evidence(
    root: Path, relative_path: str, prefix: str, expected_hash: object
) -> tuple[str | None, bool, str | None]:
    path_error = _collected_path_error(root, relative_path, prefix)
    if path_error:
        return path_error, False, None

    resolved_path = (
        root.resolve(strict=False) / PurePosixPath(relative_path)
    ).resolve(strict=False)
    if resolved_path.stat().st_size == 0:
        return "is empty", False, None

    hash_mismatch = False
    if isinstance(expected_hash, str) and _SHA256.fullmatch(expected_hash):
        hash_mismatch = _sha256_file(resolved_path).casefold() != expected_hash.casefold()

    extension = PurePosixPath(relative_path).suffix.casefold()
    image_error = _image_error(resolved_path, extension)
    return None, hash_mismatch, image_error


def _validate_header(
    data: JSONValue, collection_field: str
) -> tuple[list[str], list[JSONValue]]:
    if not isinstance(data, dict):
        return ["manifest must be an object"], []

    errors = []
    if data.get("schema_version") != 1 or not _is_int(data.get("schema_version")):
        errors.append("schema_version must equal 1")
    if not isinstance(data.get("dataset_id"), str) or not data.get("dataset_id"):
        errors.append("dataset_id must be a nonempty string")

    collection = data.get(collection_field)
    if not isinstance(collection, list):
        errors.append(f"{collection_field} must be a list")
        collection = []
    return errors, collection


def validate_negative_manifest(
    data: JSONValue, expected_slots: int, root_path: Path | None = None
) -> list[str]:
    errors, samples = _validate_header(data, "samples")
    if not isinstance(data, dict):
        return errors

    target_slots = data.get("target_slots")
    if not _is_int(target_slots):
        errors.append("target_slots must be an integer")
    elif target_slots != expected_slots:
        errors.append(f"target_slots must equal expected_slots ({expected_slots})")
    if len(samples) != expected_slots:
        errors.append(f"samples length must equal expected_slots ({expected_slots})")

    root = Path.cwd() if root_path is None else Path(root_path)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    allowed_categories = ", ".join(NEGATIVE_CATEGORIES)
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue

        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}.id must be a nonempty string")
        elif sample_id in seen_ids:
            errors.append(f"{prefix}.id duplicates '{sample_id}'")
        else:
            seen_ids.add(sample_id)

        if sample.get("category") not in NEGATIVE_CATEGORIES:
            errors.append(f"{prefix}.category must be one of: {allowed_categories}")

        if "sha256" not in sample:
            errors.append(f"{prefix}.sha256 is required")

        relative_path = sample.get("relative_path")
        safe_path = _safe_prefixed_posix_path(relative_path, NEGATIVE_PATH_PREFIX)
        supported_extension = False
        if not safe_path:
            errors.append(
                f"{prefix}.relative_path must be a safe POSIX path under {NEGATIVE_PATH_PREFIX}"
            )
        else:
            normalized_path = _normalized_path(relative_path)
            if normalized_path in seen_paths:
                errors.append(
                    f"{prefix}.relative_path duplicates normalized path '{relative_path}'"
                )
            else:
                seen_paths.add(normalized_path)
            supported_extension = (
                PurePosixPath(relative_path).suffix.casefold() in IMAGE_EXTENSIONS
            )
            if not supported_extension:
                errors.append(
                    f"{prefix}.relative_path must use a supported image extension"
                )

        status = sample.get("status")
        if status not in STATUSES:
            errors.append(f"{prefix}.status must be one of: {', '.join(STATUSES)}")
        elif status == "pending_collection":
            if "sha256" in sample and sample.get("sha256") is not None:
                errors.append(f"{prefix}.sha256 must be null for pending sample")
        else:
            expected_hash = sample.get("sha256")
            valid_hash = isinstance(expected_hash, str) and bool(
                _SHA256.fullmatch(expected_hash)
            )
            if "sha256" in sample and not valid_hash:
                errors.append(
                    f"{prefix}.sha256 must be 64 hexadecimal characters for collected sample"
                )
            if not (safe_path and supported_extension):
                continue
            path_error, hash_mismatch, image_error = _collected_image_evidence(
                root, relative_path, NEGATIVE_PATH_PREFIX, expected_hash
            )
            if path_error == "resolves outside root_path or expected prefix":
                errors.append(f"{prefix}.relative_path {path_error}")
            elif path_error:
                errors.append(f"{prefix}.relative_path {path_error} for collected sample")
            else:
                if hash_mismatch:
                    errors.append(f"{prefix}.sha256 does not match file bytes")
                if image_error:
                    errors.append(f"{prefix}.relative_path {image_error}")
    return errors


def validate_route_manifest(
    data: JSONValue, root_path: Path | None = None
) -> list[str]:
    errors, records = _validate_header(data, "routes")
    if not isinstance(data, dict):
        return errors

    present_routes = {
        route
        for record in records
        if isinstance(record, dict)
        for route in (record.get("route"),)
        if isinstance(route, str) and route in ROUTES
    }
    for required_route in ROUTES:
        if required_route not in present_routes:
            errors.append(f"routes must include route '{required_route}'")

    root = Path.cwd() if root_path is None else Path(root_path)
    seen_attempts: set[tuple[str, str, int]] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"routes[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue

        for field in ROUTE_REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"{prefix}.{field} is required")

        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}.source_id must be a nonempty string")

        route = record.get("route")
        if route not in ROUTES:
            errors.append(f"{prefix}.route must be one of: {', '.join(ROUTES)}")

        attempt = record.get("attempt")
        valid_attempt = _is_int(attempt) and attempt > 0
        if not valid_attempt:
            errors.append(f"{prefix}.attempt must be a positive integer")
        if isinstance(source_id, str) and source_id and route in ROUTES and valid_attempt:
            attempt_key = (source_id, route, attempt)
            if attempt_key in seen_attempts:
                errors.append(
                    f"{prefix} duplicates source_id/route/attempt {attempt_key!r}"
                )
            else:
                seen_attempts.add(attempt_key)

        for field in ROUTE_STRING_FIELDS:
            if field in record and not isinstance(record.get(field), str):
                errors.append(f"{prefix}.{field} must be a string")

        status = record.get("status")
        if status == "pending_collection":
            for timestamp_field in ("sent_at", "received_at"):
                value = record.get(timestamp_field)
                if timestamp_field in record and value is not None and not _is_rfc3339(value):
                    errors.append(
                        f"{prefix}.{timestamp_field} must be null or an RFC3339 timestamp"
                    )
            for hash_field in ("source_sha256", "received_sha256"):
                value = record.get(hash_field)
                if hash_field in record and value is not None and (
                    not isinstance(value, str) or not _SHA256.fullmatch(value)
                ):
                    errors.append(
                        f"{prefix}.{hash_field} must be null or 64 hexadecimal characters"
                    )
        elif status == "collected":
            sent_at = _parse_rfc3339(record.get("sent_at"))
            received_at = _parse_rfc3339(record.get("received_at"))
            if "sent_at" in record and sent_at is None:
                errors.append(
                    f"{prefix}.sent_at must be a nonempty RFC3339 timestamp for collected route"
                )
            if "received_at" in record and received_at is None:
                errors.append(
                    f"{prefix}.received_at must be a nonempty RFC3339 timestamp for collected route"
                )
            if sent_at is not None and received_at is not None and received_at < sent_at:
                errors.append(
                    f"{prefix}.received_at must be at or after sent_at for collected route"
                )
            for field in (
                "operator",
                "device",
                "software",
                "software_version",
                "account_channel",
                "notes",
                "reviewer",
            ):
                if field in record and (
                    not isinstance(record.get(field), str) or not record.get(field)
                ):
                    errors.append(f"{prefix}.{field} must be nonempty for collected route")
            if (
                isinstance(record.get("reviewer"), str)
                and record.get("reviewer")
                and record.get("reviewer") == record.get("operator")
            ):
                errors.append(f"{prefix}.reviewer must differ from operator for collected route")
            for hash_field in ("source_sha256", "received_sha256"):
                value = record.get(hash_field)
                if hash_field in record and (
                    not isinstance(value, str) or not _SHA256.fullmatch(value)
                ):
                    errors.append(
                        f"{prefix}.{hash_field} must be 64 hexadecimal characters for collected route"
                    )
            if record.get("rejection_reason") != "":
                errors.append(f"{prefix}.rejection_reason must be empty for collected route")

        source_path = record.get("source_relative_path")
        safe_source_path = _safe_prefixed_posix_path(source_path, ROUTE_SOURCE_PATH_PREFIX)
        if not safe_source_path:
            errors.append(
                f"{prefix}.source_relative_path must be a safe POSIX path under "
                f"{ROUTE_SOURCE_PATH_PREFIX}"
            )
        else:
            source_extension = PurePosixPath(source_path).suffix.casefold()
            if source_extension not in IMAGE_EXTENSIONS:
                errors.append(f"{prefix}.source_relative_path must use a supported image extension")
            if isinstance(source_id, str) and source_id:
                expected_source_name = source_id + PurePosixPath(source_path).suffix
                if PurePosixPath(source_path).name != expected_source_name:
                    errors.append(
                        f"{prefix}.source_relative_path basename must equal "
                        f"'{source_id}.<image-ext>'"
                    )

        output_path = record.get("output_relative_path")
        safe_output_path = _safe_prefixed_posix_path(output_path, ROUTE_RECEIVED_PATH_PREFIX)
        if not safe_output_path:
            errors.append(
                f"{prefix}.output_relative_path must be a safe POSIX path under "
                f"{ROUTE_RECEIVED_PATH_PREFIX}"
            )
        else:
            output_extension = PurePosixPath(output_path).suffix.casefold()
            if output_extension not in IMAGE_EXTENSIONS:
                errors.append(f"{prefix}.output_relative_path must use a supported image extension")
            if (
                isinstance(source_id, str)
                and source_id
                and route in ROUTES
                and valid_attempt
            ):
                expected_stem = f"{source_id}--{route}--attempt-{attempt:03d}"
                expected_output_name = expected_stem + PurePosixPath(output_path).suffix
                if PurePosixPath(output_path).name != expected_output_name:
                    errors.append(
                        f"{prefix}.output_relative_path basename must equal "
                        f"'{expected_stem}.<image-ext>'"
                    )
            normalized_path = _normalized_path(output_path)
            if normalized_path in seen_paths:
                errors.append(
                    f"{prefix}.output_relative_path duplicates normalized path '{output_path}'"
                )
            else:
                seen_paths.add(normalized_path)

        if status == "pending_collection":
            pending_has_evidence = any(
                (
                    record.get("sent_at") is not None,
                    record.get("received_at") is not None,
                    isinstance(record.get("operator"), str) and bool(record.get("operator")),
                    record.get("received_sha256") is not None,
                    *(
                        isinstance(record.get(field), str) and bool(record.get(field))
                        for field in (
                            "device",
                            "software",
                            "software_version",
                            "account_channel",
                            "reviewer",
                        )
                    ),
                )
            )
            if safe_output_path and (
                _collected_path_error(root, output_path, ROUTE_RECEIVED_PATH_PREFIX) is None
            ):
                pending_has_evidence = True

            rejection_reason = record.get("rejection_reason")
            notes = record.get("notes")
            if pending_has_evidence:
                if not (isinstance(rejection_reason, str) and rejection_reason):
                    errors.append(
                        f"{prefix}.rejection_reason must be nonempty when pending route contains evidence"
                    )
                if not (isinstance(notes, str) and notes):
                    errors.append(
                        f"{prefix}.notes must be nonempty when pending route contains evidence"
                    )
            elif (
                isinstance(rejection_reason, str)
                and rejection_reason
                and not (isinstance(notes, str) and notes)
            ):
                errors.append(
                    f"{prefix}.notes must be nonempty when rejection_reason is nonempty"
                )

        if status not in STATUSES:
            errors.append(f"{prefix}.status must be one of: {', '.join(STATUSES)}")
        elif status == "collected":
            evidence_paths = (
                ("source_relative_path", source_path, ROUTE_SOURCE_PATH_PREFIX, "source_sha256", "source"),
                ("output_relative_path", output_path, ROUTE_RECEIVED_PATH_PREFIX, "received_sha256", "output"),
            )
            for path_field, relative_path, path_prefix, hash_field, artifact_name in evidence_paths:
                safe_path = _safe_prefixed_posix_path(relative_path, path_prefix)
                if not safe_path:
                    continue
                extension = PurePosixPath(relative_path).suffix.casefold()
                if extension not in IMAGE_FORMAT_BY_EXTENSION:
                    continue
                expected_hash = record.get(hash_field)
                path_error, hash_mismatch, image_error = _collected_image_evidence(
                    root, relative_path, path_prefix, expected_hash
                )
                if path_error == "resolves outside root_path or expected prefix":
                    errors.append(f"{prefix}.{path_field} {path_error}")
                elif path_error:
                    errors.append(f"{prefix}.{path_field} {path_error} for collected route")
                else:
                    if hash_mismatch:
                        errors.append(
                            f"{prefix}.{hash_field} does not match {artifact_name} file bytes"
                        )
                    if image_error:
                        errors.append(f"{prefix}.{path_field} {image_error}")
    return errors


def manifest_counts(data: JSONValue) -> dict[str, int]:
    records = []
    if isinstance(data, dict):
        if isinstance(data.get("samples"), list):
            records = data["samples"]
        elif isinstance(data.get("routes"), list):
            records = data["routes"]
    statuses = [record.get("status") for record in records if isinstance(record, dict)]
    return {
        "slots": len(records),
        "collected": statuses.count("collected"),
        "pending": statuses.count("pending_collection"),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a commercial dataset manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--kind", choices=("routes",), required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    try:
        data = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    errors = validate_route_manifest(data, args.root)
    for error in errors:
        print(error, file=sys.stderr)
    print(json.dumps(manifest_counts(data), sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(_main())
