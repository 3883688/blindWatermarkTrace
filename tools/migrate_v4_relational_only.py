"""Destructively replace legacy image data with the normalized V4 schema."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine

from trace_app.config import settings
from trace_app.database.store import DatabaseStore
from trace_app.v4.startup import initialize_v4_schema


LEGACY_TABLES = (
    "deep_forensics_jobs",
    "audit_events",
    "v4_counters",
    "rate_limit_buckets",
    "auth_sessions",
    "v4_records",
    "source_group_sift_feature_blobs",
    "source_group_features",
    "source_group_embeddings",
    "media_objects",
    "source_groups",
    "image_records",
    "stats",
)


def decode_legacy_menus(value: Any) -> tuple[str, ...]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return ()
    if not isinstance(decoded, list):
        return ()
    result: list[str] = []
    for item in decoded:
        if isinstance(item, str) and item and item not in result:
            result.append(item)
    return tuple(result)


def role_menu_order_clause(columns: set[str]) -> str:
    if "position_index" in columns:
        return "role_key, position_index"
    if "position" in columns:
        return "role_key, position"
    return "role_key, menu_key"


def _qualified_table(engine: Engine, schema: str, table: str) -> str:
    preparer = engine.dialect.identifier_preparer
    return f"{preparer.quote_schema(schema)}.{preparer.quote(table)}"


def _snapshot_users(engine: Engine, schema: str) -> tuple[tuple[Any, ...], ...]:
    table = Table("users", MetaData(), schema=schema, autoload_with=engine)
    primary = tuple(table.primary_key.columns)
    statement = select(table)
    if primary:
        statement = statement.order_by(*primary)
    with engine.connect() as connection:
        return tuple(tuple(row) for row in connection.execute(statement))


def _snapshot_roles(engine: Engine, schema: str) -> tuple[tuple[Any, ...], ...]:
    roles = _qualified_table(engine, schema, "roles")
    with engine.connect() as connection:
        return tuple(
            connection.execute(
                text(f"SELECT role_key, label FROM {roles} ORDER BY role_key")
            ).all()
        )


def _snapshot_menus(engine: Engine, schema: str) -> dict[str, tuple[str, ...]]:
    inspector = inspect(engine)
    role_columns = {
        item["name"] for item in inspector.get_columns("roles", schema=schema)
    }
    menus: dict[str, tuple[str, ...]] = {}
    if "role_menus" in inspector.get_table_names(schema=schema):
        menu_columns = {
            item["name"]
            for item in inspector.get_columns("role_menus", schema=schema)
        }
        order_clause = role_menu_order_clause(menu_columns)
        role_menus = _qualified_table(engine, schema, "role_menus")
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    f"SELECT role_key, menu_key FROM {role_menus} "
                    f"ORDER BY {order_clause}"
                )
            ).all()
        for role_key, menu_key in rows:
            menus.setdefault(str(role_key), tuple())
            menus[str(role_key)] += (str(menu_key),)
    if "menus" in role_columns:
        roles = _qualified_table(engine, schema, "roles")
        with engine.connect() as connection:
            rows = connection.execute(
                text(f"SELECT role_key, menus FROM {roles} ORDER BY role_key")
            ).all()
        for role_key, value in rows:
            decoded = decode_legacy_menus(value)
            if decoded:
                menus[str(role_key)] = decoded
    return menus


def migrate(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        raise RuntimeError("V4 relational migration requires PostgreSQL")
    with engine.connect() as connection:
        schema = str(connection.execute(text("SELECT current_schema()")).scalar_one())
    names = set(inspect(engine).get_table_names(schema=schema))
    if not {"users", "roles"} <= names:
        raise RuntimeError("identity tables users and roles must exist before migration")
    users_before = _snapshot_users(engine, schema)
    roles_before = _snapshot_roles(engine, schema)
    menus_before = _snapshot_menus(engine, schema)

    schema_engine = engine.execution_options(schema_translate_map={None: schema})
    DatabaseStore(schema_engine).create_schema(identity_only=True)
    role_menus = _qualified_table(engine, schema, "role_menus")
    roles = _qualified_table(engine, schema, "roles")
    with engine.begin() as connection:
        menu_columns = {
            item["name"]
            for item in inspect(connection).get_columns("role_menus", schema=schema)
        }
        if "position_index" not in menu_columns:
            connection.execute(
                text(f"ALTER TABLE {role_menus} ADD COLUMN position_index INTEGER")
            )
        connection.execute(text(f"DELETE FROM {role_menus}"))
        for role_key, menus in menus_before.items():
            for position, menu_key in enumerate(menus):
                connection.execute(
                    text(
                        f"INSERT INTO {role_menus}"
                        "(role_key, menu_key, position_index) "
                        "VALUES (:role_key, :menu_key, :position_index)"
                    ),
                    {
                        "role_key": role_key,
                        "menu_key": menu_key,
                        "position_index": position,
                    },
                )
        connection.execute(
            text(
                f"ALTER TABLE {role_menus} "
                "ALTER COLUMN position_index SET NOT NULL"
            )
        )
        quoted = ", ".join(
            _qualified_table(engine, schema, name) for name in LEGACY_TABLES
        )
        connection.execute(text(f"DROP TABLE IF EXISTS {quoted} CASCADE"))
        columns = {
            item["name"]
            for item in inspect(connection).get_columns("roles", schema=schema)
        }
        if "menus" in columns:
            connection.execute(text(f"ALTER TABLE {roles} DROP COLUMN menus"))

    initialize_v4_schema(schema_engine, require_postgres=True)
    if _snapshot_users(engine, schema) != users_before:
        raise RuntimeError("user identity verification failed after V4 migration")
    if _snapshot_roles(engine, schema) != roles_before:
        raise RuntimeError("role identity verification failed after V4 migration")
    if _snapshot_menus(engine, schema) != menus_before:
        raise RuntimeError("role permission verification failed after V4 migration")


def main() -> int:
    backup = Path(os.environ.get("V4_MIGRATION_BACKUP_PATH", ""))
    if not backup.is_file() or backup.stat().st_size <= 0:
        raise RuntimeError("a completed pg_dump is required before V4 migration")
    engine = create_engine(settings.db_url, pool_pre_ping=True, future=True)
    try:
        migrate(engine)
    finally:
        engine.dispose()
    print("V4 relational-only migration complete; identities preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "LEGACY_TABLES",
    "decode_legacy_menus",
    "migrate",
    "role_menu_order_clause",
)
