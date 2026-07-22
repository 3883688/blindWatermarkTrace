from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Connection, Engine, URL, make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database_store import DatabaseStore  # noqa: E402
from trace_app.config import DEFAULT_ROLES, MENU_LABELS  # noqa: E402


@dataclass(frozen=True)
class NormalizedRoles:
    roles: dict[str, dict[str, Any]]
    menus: dict[str, dict[str, Any]]
    role_menus: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SourceSnapshot:
    images: tuple[dict[str, Any], ...]
    roles: dict[str, dict[str, Any]]
    users: tuple[dict[str, Any], ...]
    stats: dict[str, Any]


def normalize_roles(roles: Mapping[str, Mapping[str, Any]]) -> NormalizedRoles:
    known_order = {key: index for index, key in enumerate(MENU_LABELS)}
    menu_keys: set[str] = set()
    normalized_roles: dict[str, dict[str, Any]] = {}
    role_links: list[tuple[str, str]] = []
    for raw_key, raw_info in roles.items():
        role_key = str(raw_key).strip()
        if not role_key or not isinstance(raw_info, Mapping):
            raise ValueError("roles contains an invalid role")
        label = str(raw_info.get("label") or role_key)
        raw_menus = raw_info.get("menus")
        if not isinstance(raw_menus, list):
            raise ValueError(f"role {role_key} has invalid menus")
        normalized_menu_keys: list[str] = []
        for raw_menu in raw_menus:
            menu_key = str(raw_menu).strip()
            if not menu_key or menu_key in normalized_menu_keys:
                continue
            normalized_menu_keys.append(menu_key)
            menu_keys.add(menu_key)
            role_links.append((role_key, menu_key))
        normalized_roles[role_key] = {
            "label": label,
            "is_system": role_key in DEFAULT_ROLES,
            "menus": normalized_menu_keys,
        }

    def menu_sort_key(menu_key: str) -> tuple[int, str]:
        return (known_order.get(menu_key, len(known_order)), menu_key)

    normalized_menus = {}
    unknown_index = len(known_order)
    for menu_key in sorted(menu_keys, key=menu_sort_key):
        sort_order = known_order.get(menu_key)
        if sort_order is None:
            sort_order = unknown_index
            unknown_index += 1
        normalized_menus[menu_key] = {
            "label": MENU_LABELS.get(menu_key, menu_key),
            "sort_order": sort_order,
            "enabled": True,
        }
    return NormalizedRoles(
        roles=normalized_roles,
        menus=normalized_menus,
        role_menus=tuple(sorted(set(role_links))),
    )


def build_role_menu_rows(normalized: NormalizedRoles) -> tuple[tuple[str, str], ...]:
    return normalized.role_menus


def validate_source_snapshot(
    *,
    images: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    roles: Mapping[str, Mapping[str, Any]],
    users: Mapping[str, Mapping[str, Any]] | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    stats: Mapping[str, Any],
) -> None:
    image_ids = [str(record.get("id") or "") for record in images]
    if any(not value for value in image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("image records contain missing or duplicate IDs")
    normalized = normalize_roles(roles)
    role_keys = set(normalized.roles)
    user_rows = (
        [
            {"username": username, **dict(info)}
            for username, info in users.items()
        ]
        if isinstance(users, Mapping)
        else users
    )
    seen_users: set[str] = set()
    for user in user_rows:
        username = str(user.get("username") or "")
        if not username or username in seen_users:
            raise ValueError("users contain missing or duplicate usernames")
        seen_users.add(username)
        if str(user.get("role_key") or user.get("role") or "") not in role_keys:
            raise ValueError(f"user {username} references an unknown role")
        if not str(user.get("password_hash") or ""):
            raise ValueError(f"user {username} has an empty password hash")
    if not isinstance(stats, Mapping):
        raise ValueError("stats must be a mapping")


def _json_value(value: Any, *, field: str) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} contains invalid JSON") from exc
    raise ValueError(f"{field} contains unsupported data")


def read_source(engine: Engine) -> SourceSnapshot:
    store = DatabaseStore(engine)
    with engine.connect() as connection:
        image_rows = connection.execute(
            select(
                store.image_records.c.id,
                store.image_records.c.position_index,
                store.image_records.c.user_id,
                store.image_records.c.data,
                store.image_records.c.created_at,
            ).order_by(store.image_records.c.position_index)
        ).mappings()
        images = []
        for row in image_rows:
            data = _json_value(row["data"], field="image_records.data")
            if not isinstance(data, dict):
                raise ValueError("image_records.data must contain JSON objects")
            images.append(
                {
                    "id": str(row["id"]),
                    "position_index": int(row["position_index"]),
                    "user_id": row["user_id"],
                    "data": data,
                    "created_at": str(row["created_at"] or ""),
                }
            )
        role_rows = connection.execute(
            select(store.roles.c.role_key, store.roles.c.label, store.roles.c.menus)
        ).mappings()
        roles = {}
        for row in role_rows:
            menus = _json_value(row["menus"], field="roles.menus")
            roles[str(row["role_key"])] = {
                "label": str(row["label"] or row["role_key"]),
                "menus": menus,
            }
        user_rows = connection.execute(
            select(
                store.users.c.id,
                store.users.c.username,
                store.users.c.password_hash,
                store.users.c.role_key,
            ).order_by(store.users.c.id)
        ).mappings()
        users = [dict(row) for row in user_rows]
        stat_rows = connection.execute(select(store.stats.c.stat_key, store.stats.c.data)).mappings()
        stats = {
            str(row["stat_key"]): _json_value(row["data"], field="stats.data")
            for row in stat_rows
        }
    validate_source_snapshot(images=images, roles=roles, users=users, stats=stats)
    return SourceSnapshot(tuple(images), roles, tuple(users), stats)


def define_permission_tables(store: DatabaseStore) -> tuple[Table, Table]:
    menus = Table(
        "menus",
        store.metadata,
        Column("menu_key", String(64), primary_key=True),
        Column("label", String(128), nullable=False),
        Column("sort_order", Integer, nullable=False),
        Column("enabled", Boolean, nullable=False, server_default=text("true")),
    )
    role_menus = Table(
        "role_menus",
        store.metadata,
        Column("role_key", String(64), ForeignKey("roles.role_key", ondelete="CASCADE"), primary_key=True),
        Column("menu_key", String(64), ForeignKey("menus.menu_key", ondelete="CASCADE"), primary_key=True),
    )
    return menus, role_menus


def _clear_target(connection: Connection, store: DatabaseStore, menus: Table, role_menus: Table) -> None:
    connection.execute(delete(role_menus))
    connection.execute(delete(menus))
    connection.execute(delete(store.image_records))
    connection.execute(delete(store.users))
    connection.execute(delete(store.roles))
    connection.execute(delete(store.stats))


def write_target(engine: Engine, source: SourceSnapshot) -> None:
    store = DatabaseStore(engine)
    menus, role_menus = define_permission_tables(store)
    ensure_target_schema(engine, store)
    normalized = normalize_roles(source.roles)
    with engine.begin() as connection:
        _clear_target(connection, store, menus, role_menus)
        connection.execute(
            text(
                "insert into roles (role_key, label, menus, is_system) "
                "values (:role_key, :label, :menus, :is_system)"
            ),
            [
                {
                    "role_key": key,
                    "label": info["label"],
                    "menus": json.dumps(info["menus"], ensure_ascii=False),
                    "is_system": info["is_system"],
                }
                for key, info in normalized.roles.items()
            ],
        )
        connection.execute(
            insert(menus),
            [{"menu_key": key, **info} for key, info in normalized.menus.items()],
        )
        if normalized.role_menus:
            connection.execute(
                insert(role_menus),
                [{"role_key": role, "menu_key": menu} for role, menu in normalized.role_menus],
            )
        if source.users:
            connection.execute(
                insert(store.users),
                [
                    {
                        "id": int(user["id"]),
                        "username": str(user["username"]),
                        "password_hash": str(user["password_hash"]),
                        "role_key": str(user["role_key"]),
                    }
                    for user in source.users
                ],
            )
        if source.images:
            connection.execute(
                insert(store.image_records),
                [
                    {
                        "id": image["id"],
                        "user_id": image["user_id"],
                        "position_index": image["position_index"],
                        "data": json.dumps(image["data"], ensure_ascii=False),
                        "created_at": image["created_at"],
                    }
                    for image in source.images
                ],
            )
        if source.stats:
            connection.execute(
                insert(store.stats),
                [
                    {"stat_key": key, "data": json.dumps(value, ensure_ascii=False)}
                    for key, value in source.stats.items()
                ],
            )
        _verify_target(connection, store, menus, role_menus, source)
        if source.users:
            connection.execute(
                text("select setval(pg_get_serial_sequence('users', 'id'), (select max(id) from users))")
            )


def _verify_target(
    connection: Connection,
    store: DatabaseStore,
    menus: Table,
    role_menus: Table,
    source: SourceSnapshot,
) -> None:
    normalized = normalize_roles(source.roles)
    if connection.execute(select(store.image_records.c.id)).all().__len__() != len(source.images):
        raise RuntimeError("image count verification failed")
    actual_roles = {
        row["role_key"]: (row["menus"], bool(row["is_system"]))
        for row in connection.execute(
            text("select role_key, menus, is_system from roles")
        ).mappings()
    }
    expected_roles = {
        key: (json.dumps(info["menus"], ensure_ascii=False), bool(info["is_system"]))
        for key, info in normalized.roles.items()
    }
    if actual_roles != expected_roles:
        raise RuntimeError("role verification failed")
    actual_links = tuple(
        sorted((row["role_key"], row["menu_key"]) for row in connection.execute(select(role_menus)).mappings())
    )
    if actual_links != normalized.role_menus:
        raise RuntimeError("role menu verification failed")
    actual_users = {
        row["username"]: dict(row)
        for row in connection.execute(select(store.users)).mappings()
    }
    expected_users = {str(user["username"]): user for user in source.users}
    if set(actual_users) != set(expected_users):
        raise RuntimeError("user verification failed")
    for username, expected in expected_users.items():
        actual = actual_users[username]
        if actual["id"] != expected["id"] or actual["role_key"] != expected["role_key"] or actual["password_hash"] != expected["password_hash"]:
            raise RuntimeError(f"user verification failed: {username}")
    actual_stats = {
        row["stat_key"]: _json_value(row["data"], field="target stats")
        for row in connection.execute(select(store.stats.c.stat_key, store.stats.c.data)).mappings()
    }
    if actual_stats != source.stats:
        raise RuntimeError("stats verification failed")


def ensure_target_schema(engine: Engine, store: DatabaseStore) -> None:
    store.create_schema()
    with engine.begin() as connection:
        connection.execute(
            text("alter table roles add column if not exists is_system boolean not null default false")
        )
        connection.execute(
            text("alter table roles add column if not exists created_at timestamp with time zone not null default current_timestamp")
        )
        connection.execute(
            text("alter table roles add column if not exists updated_at timestamp with time zone not null default current_timestamp")
        )
        connection.execute(text("create extension if not exists vector"))


def _create_database_if_missing(postgres_url: str) -> None:
    url = make_url(postgres_url)
    if not url.database or url.database == "postgres":
        return
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with admin_engine.connect() as connection:
            exists = connection.execute(
                text("select 1 from pg_database where datname = :name"), {"name": url.database}
            ).scalar()
            if not exists:
                safe_name = url.database.replace('"', '""')
                connection.execute(text(f'create database "{safe_name}"'))
    finally:
        admin_engine.dispose()


def run(*, source_url: str, target_url: str, dry_run: bool = False) -> SourceSnapshot:
    if not source_url or not target_url:
        raise ValueError("DB_URL and POSTGRES_URL are required")
    source_engine = create_engine(source_url, pool_pre_ping=True, future=True)
    try:
        source = read_source(source_engine)
    finally:
        source_engine.dispose()
    if not dry_run:
        _create_database_if_missing(target_url)
        target_engine = create_engine(target_url, pool_pre_ping=True, future=True)
        try:
            write_target(target_engine, source)
        finally:
            target_engine.dispose()
    return source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate Trace data from MySQL to PostgreSQL")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=True)
    try:
        source = run(
            source_url=os.getenv("DB_URL", "").strip(),
            target_url=os.getenv("POSTGRES_URL", "").strip(),
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Migration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"Migration {'dry-run' if args.dry_run else 'completed'}: "
        f"images={len(source.images)} roles={len(source.roles)} users={len(source.users)} "
        f"stats={len(source.stats)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
