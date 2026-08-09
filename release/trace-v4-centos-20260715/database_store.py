import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection

from password_security import hash_password, verify_password


class DatabaseStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.metadata = MetaData()
        self.image_records = Table(
            "image_records",
            self.metadata,
            Column("id", String(64), primary_key=True),
            Column("position_index", Integer, nullable=False),
            Column("data", Text, nullable=False),
            Column("created_at", String(32)),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
            ),
        )
        self.roles = Table(
            "roles",
            self.metadata,
            Column("role_key", String(64), primary_key=True),
            Column("label", String(128), nullable=False),
            Column("menus", Text, nullable=False),
            Column(
                "created_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
            ),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
                onupdate=func.current_timestamp(),
            ),
        )
        self.users = Table(
            "users",
            self.metadata,
            Column("username", String(128), primary_key=True),
            Column("password_hash", String(512), nullable=False),
            Column(
                "role_key",
                String(64),
                ForeignKey("roles.role_key"),
                nullable=False,
            ),
            Column(
                "created_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
            ),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
                onupdate=func.current_timestamp(),
            ),
        )
        self.stats = Table(
            "stats",
            self.metadata,
            Column("stat_key", String(64), primary_key=True),
            Column("data", Text, nullable=False),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                server_default=func.current_timestamp(),
                onupdate=func.current_timestamp(),
            ),
        )

    def create_schema(self, connection: Connection | None = None) -> None:
        self.metadata.create_all(connection or self.engine)

    @contextmanager
    def _transaction(
        self, connection: Connection | None = None
    ) -> Iterator[Connection]:
        if connection is not None:
            yield connection
            return
        with self.engine.begin() as managed:
            yield managed

    def replace_records(
        self,
        records: list[dict[str, Any]],
        connection: Connection | None = None,
    ) -> None:
        with self._transaction(connection) as conn:
            conn.execute(delete(self.image_records))
            rows = []
            for index, source in enumerate(records):
                record = dict(source)
                record_id = str(record.get("id") or uuid.uuid4().hex)
                record["id"] = record_id
                rows.append(
                    {
                        "id": record_id,
                        "position_index": index,
                        "data": json.dumps(record, ensure_ascii=False),
                        "created_at": str(record.get("created_at") or ""),
                    }
                )
            if rows:
                conn.execute(insert(self.image_records), rows)

    def read_records(
        self, connection: Connection | None = None
    ) -> list[dict[str, Any]]:
        with self._transaction(connection) as conn:
            rows = conn.execute(
                select(self.image_records.c.data).order_by(
                    self.image_records.c.position_index
                )
            ).scalars()
            return [json.loads(data) for data in rows]

    def replace_roles(
        self,
        roles: dict[str, dict[str, Any]],
        connection: Connection | None = None,
    ) -> None:
        with self._transaction(connection) as conn:
            conn.execute(delete(self.roles))
            rows = [
                {
                    "role_key": role_key,
                    "label": str(info.get("label") or role_key),
                    "menus": json.dumps(info.get("menus") or [], ensure_ascii=False),
                }
                for role_key, info in roles.items()
            ]
            if rows:
                conn.execute(insert(self.roles), rows)

    def read_roles(
        self, connection: Connection | None = None
    ) -> dict[str, dict[str, Any]]:
        with self._transaction(connection) as conn:
            rows = conn.execute(
                select(
                    self.roles.c.role_key,
                    self.roles.c.label,
                    self.roles.c.menus,
                ).order_by(self.roles.c.role_key)
            ).mappings()
            return {
                row["role_key"]: {
                    "label": row["label"],
                    "menus": json.loads(row["menus"]),
                }
                for row in rows
            }

    def update_role_menus(self, role_key: str, menus: list[str]) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.roles)
                .where(self.roles.c.role_key == role_key)
                .values(menus=json.dumps(menus, ensure_ascii=False))
            )
            return bool(result.rowcount)

    def set_stats(
        self,
        stat_key: str,
        data: Any,
        connection: Connection | None = None,
    ) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        with self._transaction(connection) as conn:
            exists = conn.execute(
                select(self.stats.c.stat_key).where(self.stats.c.stat_key == stat_key)
            ).first()
            if exists:
                conn.execute(
                    update(self.stats)
                    .where(self.stats.c.stat_key == stat_key)
                    .values(data=payload)
                )
            else:
                conn.execute(
                    insert(self.stats).values(stat_key=stat_key, data=payload)
                )

    def get_stats(
        self,
        stat_key: str,
        default: Any,
        connection: Connection | None = None,
    ) -> Any:
        with self._transaction(connection) as conn:
            data = conn.execute(
                select(self.stats.c.data).where(self.stats.c.stat_key == stat_key)
            ).scalar_one_or_none()
            return default if data is None else json.loads(data)

    def create_user(self, username: str, password: str, role_key: str) -> None:
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(self.users.c.username).where(self.users.c.username == username)
            ).first()
            if exists:
                raise ValueError(f"user {username} already exists")
            connection.execute(
                insert(self.users).values(
                    username=username,
                    password_hash=hash_password(password),
                    role_key=role_key,
                )
            )

    def upsert_user_hash(
        self,
        username: str,
        password_hash: str,
        role_key: str,
        connection: Connection | None = None,
    ) -> None:
        with self._transaction(connection) as conn:
            exists = conn.execute(
                select(self.users.c.username).where(self.users.c.username == username)
            ).first()
            values = {"password_hash": password_hash, "role_key": role_key}
            if exists:
                conn.execute(
                    update(self.users)
                    .where(self.users.c.username == username)
                    .values(**values)
                )
            else:
                conn.execute(insert(self.users).values(username=username, **values))

    def list_users(
        self, connection: Connection | None = None
    ) -> dict[str, dict[str, str]]:
        with self._transaction(connection) as conn:
            rows = conn.execute(
                select(self.users.c.username, self.users.c.role_key).order_by(
                    self.users.c.username
                )
            ).mappings()
            return {
                row["username"]: {"role": row["role_key"]}
                for row in rows
            }

    def update_user_role(self, username: str, role_key: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.users)
                .where(self.users.c.username == username)
                .values(role_key=role_key)
            )
            return bool(result.rowcount)

    def delete_user(self, username: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(self.users).where(self.users.c.username == username)
            )
            return bool(result.rowcount)

    def authenticate(self, username: str, password: str) -> str | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.users.c.password_hash, self.users.c.role_key).where(
                    self.users.c.username == username
                )
            ).mappings().first()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return str(row["role_key"])

    def clear_all(self, connection: Connection | None = None) -> None:
        with self._transaction(connection) as conn:
            conn.execute(delete(self.users))
            conn.execute(delete(self.roles))
            conn.execute(delete(self.image_records))
            conn.execute(delete(self.stats))
