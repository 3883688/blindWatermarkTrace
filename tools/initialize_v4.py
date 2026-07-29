"""Guarded offline V4 backup, reset, and restore workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from trace_app.v4.startup import (
    initialize_v4_schema,
    read_ready_marker,
    verify_postgres_v4_schema,
    write_ready_marker,
)


def validate_destructive_target(path: Path, *, workspace: Path) -> Path:
    target = Path(path).expanduser().resolve()
    workspace = Path(workspace).expanduser().resolve()
    forbidden = {Path(target.anchor).resolve(), Path.home().resolve(), workspace}
    if target in forbidden or workspace not in target.parents:
        raise ValueError("destructive target must be a non-root workspace child")
    return target


def _safe_relative(name: str) -> Path:
    value = Path(name)
    if value.is_absolute() or not name or ".." in value.parts:
        raise ValueError("unsafe archive member")
    return value


def create_upload_backup(source: Path, archive: Path) -> None:
    source = Path(source).resolve()
    archive = Path(archive).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    with tarfile.open(archive, "w") as output:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("upload backup refuses symbolic links")
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            output.add(path, arcname=relative, recursive=False)
        payload = json.dumps(checksums, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(".v4-checksums.json")
        info.size = len(payload)
        from io import BytesIO
        output.addfile(info, BytesIO(payload))
    verify_upload_backup(archive)


def verify_upload_backup(archive: Path) -> dict[str, str]:
    with tarfile.open(archive, "r") as source:
        members = source.getmembers()
        if any(not item.isfile() or item.issym() or item.islnk() for item in members):
            raise ValueError("upload archive contains an unsafe member")
        by_name = {item.name: item for item in members}
        if ".v4-checksums.json" not in by_name:
            raise ValueError("upload archive checksum manifest is missing")
        for name in by_name:
            _safe_relative(name)
        manifest_file = source.extractfile(by_name[".v4-checksums.json"])
        checksums = json.loads(manifest_file.read())
        for name, expected in checksums.items():
            member = by_name.get(name)
            if member is None:
                raise ValueError("upload archive member is missing")
            content = source.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("upload archive checksum mismatch")
        return checksums


def restore_upload_backup(archive: Path, target: Path, *, workspace: Path) -> None:
    target = validate_destructive_target(target, workspace=workspace)
    verify_upload_backup(archive)
    if target.exists():
        for child in target.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as source:
        for member in source.getmembers():
            if member.name == ".v4-checksums.json":
                continue
            relative = _safe_relative(member.name)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.extractfile(member).read())


@dataclass
class GuardedInitializer:
    environment: str
    database_name: str
    workspace: Path
    upload_dir: Path
    backup_dir: Path
    ready_marker: Path
    dump_database: Callable[[Path], None]
    restore_database: Callable[[Path], None]
    snapshot_identities: Callable[[], tuple]
    clear_algorithm_data: Callable[[], None]
    create_schema: Callable[[], None]
    smoke_test: Callable[[], None]
    rotate_key: Callable[[], str]

    @property
    def confirmation(self) -> str:
        return f"RESET-V4:{self.environment}:{self.database_name}"

    @property
    def database_backup(self) -> Path:
        return Path(self.backup_dir) / "database.dump"

    @property
    def upload_backup(self) -> Path:
        return Path(self.backup_dir) / "uploads.tar"

    def preflight(self) -> None:
        validate_destructive_target(self.upload_dir, workspace=self.workspace)
        Path(self.backup_dir).resolve().mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(Path(self.backup_dir).resolve()).free <= 0:
            raise RuntimeError("backup storage has no free space")

    def backup(self) -> None:
        self.preflight()
        self.dump_database(self.database_backup)
        if not self.database_backup.is_file() or self.database_backup.stat().st_size == 0:
            raise RuntimeError("database backup verification failed")
        create_upload_backup(self.upload_dir, self.upload_backup)

    def apply(self, confirmation: str) -> None:
        if confirmation != self.confirmation:
            raise ValueError("reset confirmation does not match")
        if not self.database_backup.is_file():
            raise RuntimeError("verified backup is required")
        verify_upload_backup(self.upload_backup)
        self.ready_marker.unlink(missing_ok=True)
        identities = self.snapshot_identities()
        self.clear_algorithm_data()
        restore_upload_backup(self._empty_archive(), self.upload_dir, workspace=self.workspace)
        self.create_schema()
        if self.snapshot_identities() != identities:
            raise RuntimeError("user or role identities changed during reset")
        key_id = self.rotate_key()
        self.smoke_test()
        write_ready_marker(
            self.ready_marker,
            {"schema_id": "v4", "model_id": "v4-models", "key_id": key_id},
        )

    def restore(self) -> None:
        self.ready_marker.unlink(missing_ok=True)
        verify_upload_backup(self.upload_backup)
        self.restore_database(self.database_backup)
        restore_upload_backup(self.upload_backup, self.upload_dir, workspace=self.workspace)

    def _empty_archive(self) -> Path:
        empty = Path(self.backup_dir) / "empty-uploads"
        empty.mkdir(exist_ok=True)
        archive = Path(self.backup_dir) / "empty-uploads.tar"
        create_upload_backup(empty, archive)
        return archive


RESET_TABLES = (
    "deep_forensics_jobs", "audit_events", "v4_records", "source_group_features",
    "source_group_embeddings", "source_groups", "media_objects", "v4_counters",
    "auth_sessions", "rate_limit_buckets", "image_records", "stats",
)


def _build_from_env(env_file: str | None) -> GuardedInitializer:
    values = {**os.environ, **({k: v for k, v in dotenv_values(env_file).items() if v is not None} if env_file else {})}
    db_url = values.get("DB_URL", "")
    if not db_url:
        raise RuntimeError("DB_URL is required")
    url = make_url(db_url)
    if url.get_backend_name() != "postgresql" or not url.database:
        raise RuntimeError("V4 initialization requires PostgreSQL")
    workspace = Path(values.get("WORKSPACE_DIR", Path(__file__).resolve().parents[1])).resolve()
    upload_dir = Path(values.get("UPLOAD_DIR", workspace / "uploads")).resolve()
    backup_dir = Path(values.get("V4_BACKUP_DIR", workspace.parent / "trace-v4-backup")).resolve()
    marker = Path(values.get("V4_READY_MARKER", workspace / "data" / "v4-ready.json")).resolve()
    manifest = Path(values.get("V4_MODEL_MANIFEST_PATH", workspace / "models" / "v4-models.example.json")).resolve()
    key_file = Path(values.get("V4_KEY_FILE", workspace / "data" / "v4-auth.key")).resolve()
    engine = create_engine(db_url, pool_pre_ping=True)

    def database_preflight() -> None:
        with engine.connect() as connection:
            if not connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')")).scalar_one():
                raise RuntimeError("pgvector is unavailable")
        if not manifest.is_file():
            raise RuntimeError("V4 model manifest is unavailable")
        if int(values.get("V4_SYNC_WORKER_QUOTA", "1")) <= 0 or int(values.get("V4_DEEP_WORKER_QUOTA", "1")) <= 0:
            raise RuntimeError("V4 worker configuration is invalid")

    def dump_database(path: Path) -> None:
        subprocess.run(["pg_dump", "--format=custom", f"--file={path}", db_url], check=True)
        subprocess.run(["pg_restore", "--list", str(path)], check=True, stdout=subprocess.DEVNULL)

    def restore_database(path: Path) -> None:
        subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", f"--dbname={db_url}", str(path)], check=True)

    def identities() -> tuple:
        with engine.connect() as connection:
            users = connection.execute(text("SELECT id, username, role_key FROM users ORDER BY id")).all()
            roles = connection.execute(text("SELECT role_key FROM roles ORDER BY role_key")).all()
        return tuple(users), tuple(roles)

    def clear_data() -> None:
        quoted = ", ".join(f'"{name}"' for name in RESET_TABLES)
        with engine.begin() as connection:
            connection.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))

    def rotate_key() -> str:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(secrets.token_bytes(32))
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            pass
        return f"v4-{secrets.token_hex(8)}"

    workflow = GuardedInitializer(
        environment=values.get("ENVIRONMENT", "production"), database_name=url.database,
        workspace=workspace, upload_dir=upload_dir, backup_dir=backup_dir, ready_marker=marker,
        dump_database=dump_database, restore_database=restore_database,
        snapshot_identities=identities, clear_algorithm_data=clear_data,
        create_schema=lambda: initialize_v4_schema(engine, require_postgres=True),
        smoke_test=lambda: verify_postgres_v4_schema(engine), rotate_key=rotate_key,
    )
    original_preflight = workflow.preflight
    workflow.preflight = lambda: (original_preflight(), database_preflight())
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "backup", "apply", "verify", "restore"))
    parser.add_argument("--env-file")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    workflow = _build_from_env(args.env_file)
    if args.command == "preflight":
        workflow.preflight()
        print(f"environment={workflow.environment} database={workflow.database_name} upload={workflow.upload_dir.name}")
        print("preflight: ok")
    elif args.command == "backup":
        workflow.backup(); print("backup: ok")
    elif args.command == "apply":
        workflow.apply(args.confirm or ""); print("apply: ok")
    elif args.command == "verify":
        if read_ready_marker(workflow.ready_marker) is None:
            raise RuntimeError("V4 ready marker is unavailable")
        print("verify: ok")
    else:
        workflow.restore(); print("restore: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
