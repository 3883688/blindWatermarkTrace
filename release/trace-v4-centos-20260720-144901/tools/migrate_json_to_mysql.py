from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, select, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database_store import DatabaseStore  # noqa: E402
from password_security import hash_password, verify_password  # noqa: E402


SOURCE_NAMES = (
    "images.json",
    "detection_stats.json",
    "watermark_stats.json",
    "roles.json",
    "users.json",
)


@dataclass(frozen=True)
class SourceData:
    images: list[dict[str, Any]]
    detection_stats: dict[str, Any]
    watermark_stats: dict[str, Any]
    roles: dict[str, dict[str, Any]]
    users: dict[str, dict[str, str]]
    raw_files: dict[str, bytes]


@dataclass(frozen=True)
class MigrationResult:
    image_count: int
    role_count: int
    user_count: int
    backup_dir: Path


def _decode_json(name: str, raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc


def load_source_data(data_dir: Path) -> SourceData:
    raw_files: dict[str, bytes] = {}
    decoded: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"missing required source file: {name}")
        raw = path.read_bytes()
        raw_files[name] = raw
        decoded[name] = _decode_json(name, raw)

    images = decoded["images.json"]
    if not isinstance(images, list) or not all(
        isinstance(record, dict) and str(record.get("id") or "")
        for record in images
    ):
        raise ValueError("images.json must contain records with non-empty ids")
    image_ids = [str(record["id"]) for record in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("images.json contains duplicate ids")

    detection_stats = decoded["detection_stats.json"]
    if not isinstance(detection_stats, dict) or not all(
        isinstance(detection_stats.get(key), int)
        for key in ("attempts", "successes")
    ):
        raise ValueError("detection_stats.json has an invalid shape")

    watermark_stats = decoded["watermark_stats.json"]
    if not isinstance(watermark_stats, dict) or not isinstance(
        watermark_stats.get("daily"), dict
    ):
        raise ValueError("watermark_stats.json has an invalid shape")

    roles_document = decoded["roles.json"]
    roles = roles_document.get("roles") if isinstance(roles_document, dict) else None
    if not isinstance(roles, dict) or not roles:
        raise ValueError("roles.json must contain a non-empty roles object")
    for role_key, info in roles.items():
        if (
            not str(role_key)
            or not isinstance(info, dict)
            or not isinstance(info.get("menus"), list)
        ):
            raise ValueError("roles.json contains an invalid role")

    users_document = decoded["users.json"]
    users = users_document.get("users") if isinstance(users_document, dict) else None
    if not isinstance(users, dict) or not users:
        raise ValueError("users.json must contain a non-empty users object")
    normalized_users: dict[str, dict[str, str]] = {}
    for username, info in users.items():
        if not str(username) or not isinstance(info, dict):
            raise ValueError("users.json contains an invalid user")
        password = info.get("password")
        role_key = str(info.get("role") or "")
        if not isinstance(password, str) or not password or role_key not in roles:
            raise ValueError("users.json contains an invalid password or role")
        normalized_users[str(username)] = {
            "password": password,
            "role": role_key,
        }

    return SourceData(
        images=[dict(record) for record in images],
        detection_stats=dict(detection_stats),
        watermark_stats=dict(watermark_stats),
        roles={str(key): dict(value) for key, value in roles.items()},
        users=normalized_users,
        raw_files=raw_files,
    )


def _verify_source_files(source: SourceData, data_dir: Path) -> None:
    for name, expected in source.raw_files.items():
        path = data_dir / name
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"source file changed after validation: {name}")


def _verify_database(
    store: DatabaseStore, source: SourceData, connection
) -> None:
    if store.read_records(connection) != source.images:
        raise RuntimeError("image record verification failed")
    if store.read_roles(connection) != source.roles:
        raise RuntimeError("role verification failed")
    if store.get_stats("detection_stats", {}, connection) != source.detection_stats:
        raise RuntimeError("detection statistics verification failed")
    if store.get_stats("watermark_stats", {}, connection) != source.watermark_stats:
        raise RuntimeError("watermark statistics verification failed")

    rows = connection.execute(
        select(
            store.users.c.username,
            store.users.c.password_hash,
            store.users.c.role_key,
        )
    ).mappings()
    actual = {row["username"]: row for row in rows}
    if set(actual) != set(source.users):
        raise RuntimeError("user key verification failed")
    for username, expected in source.users.items():
        row = actual[username]
        if row["role_key"] != expected["role"] or not verify_password(
            expected["password"], row["password_hash"]
        ):
            raise RuntimeError("user verification failed")


def _next_backup_dir(backup_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = backup_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = backup_root / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def _backup_and_remove(
    source: SourceData, data_dir: Path, backup_root: Path
) -> Path:
    destination = _next_backup_dir(backup_root)
    destination.mkdir(parents=True)
    for name, expected in source.raw_files.items():
        target = destination / name
        shutil.copy2(data_dir / name, target)
        if target.read_bytes() != expected:
            raise RuntimeError(f"backup verification failed: {name}")
    for name in SOURCE_NAMES:
        (data_dir / name).unlink()
    return destination


def migrate(
    engine: Engine,
    source: SourceData,
    data_dir: Path,
    backup_dir: Path,
) -> MigrationResult:
    _verify_source_files(source, data_dir)
    store = DatabaseStore(engine)
    store.create_schema()
    with engine.begin() as connection:
        store.clear_all(connection)
        store.replace_roles(source.roles, connection)
        store.replace_records(source.images, connection)
        store.set_stats("detection_stats", source.detection_stats, connection)
        store.set_stats("watermark_stats", source.watermark_stats, connection)
        for username, info in source.users.items():
            store.upsert_user_hash(
                username,
                hash_password(info["password"]),
                info["role"],
                connection,
            )
        _verify_database(store, source, connection)

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS app_json_store"))

    completed_backup = _backup_and_remove(source, data_dir, backup_dir)
    return MigrationResult(
        image_count=len(source.images),
        role_count=len(source.roles),
        user_count=len(source.users),
        backup_dir=completed_backup,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate Trace JSON data to MySQL")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=ROOT.parent / "trace-private-migration-backups",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=True)
    db_url = os.getenv("DB_URL", "").strip()
    if not db_url:
        print("Migration failed: DB_URL is missing", file=sys.stderr)
        return 1
    try:
        source = load_source_data(args.data_dir)
        result = migrate(
            create_engine(db_url, pool_pre_ping=True, future=True),
            source,
            args.data_dir,
            args.backup_dir,
        )
    except Exception as exc:
        print(f"Migration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "Migration completed: "
        f"images={result.image_count} roles={result.role_count} "
        f"users={result.user_count} backup={result.backup_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
