import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select, text

from trace_app.database.store import DatabaseStore
from tools.migrate_json_to_mysql import load_source_data, migrate


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _source_files(data_dir: Path) -> dict[str, bytes]:
    data_dir.mkdir()
    _write_json(
        data_dir / "images.json",
        [{"id": "image-1", "trace_id": "TR-1", "created_at": "2026-07-15"}],
    )
    _write_json(data_dir / "detection_stats.json", {"attempts": 7, "successes": 3})
    _write_json(data_dir / "watermark_stats.json", {"daily": {"2026-07-15": 2}})
    _write_json(
        data_dir / "roles.json",
        {
            "roles": {
                "admin": {"label": "管理员", "menus": ["watermark", "role"]},
                "viewer": {"label": "查看员", "menus": ["trace"]},
            }
        },
    )
    _write_json(
        data_dir / "users.json",
        {
            "users": {
                "admin-user": {"password": "admin-secret", "role": "admin"},
                "reader": {"password": "reader-secret", "role": "viewer"},
            }
        },
    )
    return {path.name: path.read_bytes() for path in data_dir.glob("*.json")}


def test_load_source_data_rejects_missing_and_invalid_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    originals = _source_files(data_dir)
    (data_dir / "users.json").unlink()

    with pytest.raises(ValueError, match="users.json"):
        load_source_data(data_dir)

    (data_dir / "users.json").write_bytes(originals["users.json"])
    _write_json(data_dir / "roles.json", {"roles": []})
    with pytest.raises(ValueError, match="roles.json"):
        load_source_data(data_dir)


def test_migrate_imports_verifies_backs_up_and_removes_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    originals = _source_files(data_dir)
    source = load_source_data(data_dir)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")

    result = migrate(engine, source, data_dir, tmp_path / "private-backups")

    store = DatabaseStore(engine)
    assert result.image_count == 1
    assert result.user_count == 2
    assert result.role_count == 2
    assert store.read_records() == source.images
    assert store.read_roles() == source.roles
    assert store.get_stats("detection_stats", {}) == source.detection_stats
    assert store.get_stats("watermark_stats", {}) == source.watermark_stats
    assert store.authenticate("admin-user", "admin-secret") == "admin"
    assert store.authenticate("reader", "reader-secret") == "viewer"
    with engine.connect() as connection:
        hashes = connection.execute(select(store.users.c.password_hash)).scalars().all()
    assert all(value.startswith("scrypt$v1$") for value in hashes)
    assert all("secret" not in value for value in hashes)
    assert not list(data_dir.glob("*.json"))
    assert {
        path.name: path.read_bytes() for path in result.backup_dir.glob("*.json")
    } == originals


def test_migration_is_idempotent_when_inputs_are_restored(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _source_files(data_dir)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    first = migrate(
        engine,
        load_source_data(data_dir),
        data_dir,
        tmp_path / "private-backups",
    )
    data_dir.mkdir(exist_ok=True)
    for backup in first.backup_dir.glob("*.json"):
        shutil.copy2(backup, data_dir / backup.name)

    migrate(
        engine,
        load_source_data(data_dir),
        data_dir,
        tmp_path / "private-backups",
    )

    store = DatabaseStore(engine)
    assert len(store.read_records()) == 1
    assert set(store.list_users()) == {"admin-user", "reader"}


def test_migration_removes_legacy_json_store_after_verification(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _source_files(data_dir)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE app_json_store ("
                "store_key VARCHAR(64) PRIMARY KEY, data TEXT NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO app_json_store (store_key, data) "
                "VALUES ('users', '{\"users\":{}}')"
            )
        )

    migrate(
        engine,
        load_source_data(data_dir),
        data_dir,
        tmp_path / "private-backups",
    )

    assert "app_json_store" not in inspect(engine).get_table_names()


def test_migration_failure_rolls_back_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    _source_files(data_dir)
    source = load_source_data(data_dir)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    store = DatabaseStore(engine)
    store.create_schema()
    store.replace_roles({"sentinel": {"label": "Sentinel", "menus": []}})

    original = DatabaseStore.set_stats

    def fail_on_watermark(self, stat_key, data, connection=None):
        if stat_key == "watermark_stats":
            raise RuntimeError("injected verification failure")
        return original(self, stat_key, data, connection)

    monkeypatch.setattr(DatabaseStore, "set_stats", fail_on_watermark)

    with pytest.raises(RuntimeError, match="injected"):
        migrate(engine, source, data_dir, tmp_path / "private-backups")

    assert store.read_roles() == {
        "sentinel": {"label": "Sentinel", "menus": []}
    }
    assert {path.name for path in data_dir.glob("*.json")} == {
        "images.json",
        "detection_stats.json",
        "watermark_stats.json",
        "roles.json",
        "users.json",
    }
