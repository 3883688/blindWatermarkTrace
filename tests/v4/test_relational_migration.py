from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text

from tools.migrate_v4_relational_only import (
    LEGACY_TABLES,
    decode_legacy_menus,
    migrate,
    role_menu_order_clause,
)


def test_migration_has_fixed_v4_only_cleanup_scope_and_decodes_legacy_menus() -> None:
    assert {
        "image_records",
        "stats",
        "source_group_sift_feature_blobs",
        "source_groups",
        "source_group_embeddings",
        "source_group_features",
        "media_objects",
        "v4_records",
        "deep_forensics_jobs",
    } <= set(LEGACY_TABLES)
    assert decode_legacy_menus('["watermark", "trace", "watermark"]') == (
        "watermark",
        "trace",
    )
    assert decode_legacy_menus("not-json") == ()
    assert role_menu_order_clause({"role_key", "menu_key"}) == "role_key, menu_key"
    assert role_menu_order_clause(
        {"role_key", "menu_key", "position_index"}
    ) == "role_key, position_index"


def test_postgres_migration_preserves_users_roles_and_normalizes_menus() -> None:
    url = os.getenv("TEST_POSTGRES_URL", "").strip()
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    schema = "test_v4_migration_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE roles (role_key varchar(64) PRIMARY KEY, label varchar(128) NOT NULL, menus text NOT NULL)"))
            connection.execute(text("CREATE TABLE users (id serial PRIMARY KEY, username varchar(128) UNIQUE NOT NULL, password_hash text NOT NULL, role varchar(64) REFERENCES roles(role_key))"))
            connection.execute(text("INSERT INTO roles VALUES ('admin', '管理员', '[\"watermark\",\"trace\",\"role\"]')"))
            connection.execute(text("INSERT INTO users(username,password_hash,role) VALUES ('admin','unchanged-hash','admin')"))
            connection.execute(text("CREATE TABLE image_records (id integer PRIMARY KEY, metadata_json jsonb)"))
        migrate(engine)
        migrate(engine)
        names = set(inspect(engine).get_table_names())
        assert "image_records" not in names
        assert {"users", "roles", "role_menus", "v4_records", "source_groups"} <= names
        with engine.connect() as connection:
            assert connection.execute(text("SELECT username,password_hash,role FROM users")).one() == (
                "admin",
                "unchanged-hash",
                "admin",
            )
            assert connection.execute(text("SELECT role_key,label FROM roles")).one() == (
                "admin",
                "管理员",
            )
            assert connection.execute(text("SELECT menu_key FROM role_menus ORDER BY position_index")).scalars().all() == [
                "watermark",
                "trace",
                "role",
            ]
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()
