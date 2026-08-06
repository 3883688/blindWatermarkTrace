import json
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, inspect, select

from trace_app.database.store import DatabaseStore
from trace_app.database.repositories import Repository


def test_database_store_imports_in_a_fresh_python_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from trace_app.database.store import DatabaseStore; "
                "from trace_app.auth.password_security import hash_password; "
                "from trace_app.auth import AuthService; "
                "from trace_app.database import Repository, create_runtime; "
                "assert DatabaseStore.__module__ == 'trace_app.database.store'; "
                "assert callable(hash_password); "
                "assert AuthService.__module__ == 'trace_app.auth.service'; "
                "assert Repository.__module__ == 'trace_app.database.repositories'; "
                "assert callable(create_runtime)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.fixture
def store() -> DatabaseStore:
    database = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    database.create_schema()
    return database


def test_schema_uses_dedicated_unprefixed_tables(store: DatabaseStore) -> None:
    assert set(inspect(store.engine).get_table_names()) == {
        "image_records",
        "role_menus",
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


def test_user_identity_uses_numeric_primary_key(store: DatabaseStore) -> None:
    store.replace_roles(
        {"operator": {"label": "操作员", "menus": ["watermark"]}}
    )
    store.create_user("alice", "secret", "operator")

    identity = store.authenticate_user("alice", "secret")

    assert identity is not None
    assert isinstance(identity["id"], int)
    assert identity["username"] == "alice"
    assert identity["role"] == "operator"
    assert store.get_user_by_id(identity["id"]) == identity
    assert store.get_user_by_username("alice") == identity
    assert store.authenticate_user("alice", "wrong") is None
    assert store.authenticate("alice", "secret") == "operator"
    assert store.list_users() == {"alice": {"role": "operator"}}

    repository = Repository(store)
    assert repository.authenticate_user("alice", "secret") == identity
    assert repository.get_user_by_id(identity["id"]) == identity
    assert repository.get_user_by_username("alice") == identity

    inspector = inspect(store.engine)
    assert {column["name"] for column in inspector.get_columns("users")} == {
        "id",
        "username",
        "password_hash",
        "role_key",
        "created_at",
        "updated_at",
    }
    assert inspector.get_pk_constraint("users")["constrained_columns"] == ["id"]


def test_duplicate_user_is_rejected(store: DatabaseStore) -> None:
    store.replace_roles({"viewer": {"label": "查看员", "menus": ["trace"]}})
    store.create_user("alice", "first", "viewer")

    with pytest.raises(ValueError, match="already exists"):
        store.create_user("alice", "second", "viewer")


def test_image_records_are_scoped_by_numeric_owner(store: DatabaseStore) -> None:
    store.replace_roles(
        {
            "admin": {"label": "管理员", "menus": ["manage"]},
            "operator": {"label": "操作员", "menus": ["manage"]},
        }
    )
    store.create_user("admin", "admin-secret", "admin")
    store.create_user("alice", "alice-secret", "operator")
    admin_id = store.get_user_by_username("admin")["id"]
    alice_id = store.get_user_by_username("alice")["id"]
    legacy = {"id": "legacy", "user_id": "watermark-admin"}
    owned = {"id": "alice-image", "user_id": "watermark-alice"}

    store.replace_records([legacy])
    assert store.backfill_image_owners(admin_id) == 1
    store.insert_record(owned, owner_user_id=alice_id)

    assert store.read_records(owner_user_id=admin_id) == [legacy]
    assert store.read_records(owner_user_id=alice_id) == [owned]
    assert store.delete_record("legacy", owner_user_id=alice_id) is None
    assert store.delete_record("alice-image", owner_user_id=alice_id) == owned
    assert store.read_records(owner_user_id=alice_id) == []
    assert "user_id" in {
        column["name"]
        for column in inspect(store.engine).get_columns("image_records")
    }


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
            select(store.role_menus.c.menu_key)
            .where(store.role_menus.c.role_key == "admin")
            .order_by(store.role_menus.c.position_index)
        ).scalars().all()
    assert menus == ["watermark", "trace", "role"]
