import json

import pytest
from sqlalchemy import create_engine, inspect, select

from database_store import DatabaseStore


@pytest.fixture
def store() -> DatabaseStore:
    database = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    database.create_schema()
    return database


def test_schema_uses_dedicated_unprefixed_tables(store: DatabaseStore) -> None:
    assert set(inspect(store.engine).get_table_names()) == {
        "image_records",
        "roles",
        "stats",
        "users",
    }


def test_user_crud_stores_only_hashes(store: DatabaseStore) -> None:
    store.replace_roles(
        {
            "admin": {"label": "管理员", "menus": ["watermark", "role"]},
            "operator": {"label": "操作员", "menus": ["watermark"]},
        }
    )

    store.create_user("alice", "secret", "operator")

    with store.engine.connect() as connection:
        password_hash = connection.execute(
            select(store.users.c.password_hash).where(store.users.c.username == "alice")
        ).scalar_one()
    assert password_hash.startswith("scrypt$v1$")
    assert "secret" not in password_hash
    assert store.authenticate("alice", "secret") == "operator"
    assert store.authenticate("alice", "wrong") is None
    assert store.list_users() == {"alice": {"role": "operator"}}

    store.update_user_role("alice", "admin")
    assert store.list_users()["alice"]["role"] == "admin"
    assert store.delete_user("alice") is True
    assert store.delete_user("alice") is False


def test_duplicate_user_is_rejected(store: DatabaseStore) -> None:
    store.replace_roles({"viewer": {"label": "查看员", "menus": ["trace"]}})
    store.create_user("alice", "first", "viewer")

    with pytest.raises(ValueError, match="already exists"):
        store.create_user("alice", "second", "viewer")


def test_roles_records_and_stats_round_trip(store: DatabaseStore) -> None:
    role_data = {
        "admin": {"label": "管理员", "menus": ["watermark", "trace", "role"]},
        "viewer": {"label": "查看员", "menus": ["trace"]},
    }
    records = [{"id": "b", "created_at": "2026-07-02"}, {"id": "a"}]

    store.replace_roles(role_data)
    store.replace_records(records)
    store.set_stats("detection_stats", {"attempts": 2, "successes": 1})
    store.set_stats("watermark_stats", {"daily": {"2026-07-15": 3}})

    assert store.read_roles() == role_data
    assert store.read_records() == records
    assert store.get_stats("detection_stats", {}) == {
        "attempts": 2,
        "successes": 1,
    }
    assert store.get_stats("watermark_stats", {}) == {
        "daily": {"2026-07-15": 3}
    }
    with store.engine.connect() as connection:
        menus = connection.execute(
            select(store.roles.c.menus).where(store.roles.c.role_key == "admin")
        ).scalar_one()
    assert json.loads(menus) == ["watermark", "trace", "role"]
