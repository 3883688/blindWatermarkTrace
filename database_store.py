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
            Column("user_id", Integer, ForeignKey("users.id"), nullable=True),
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
        self.role_menus = Table(
            "role_menus",
            self.metadata,
            Column(
                "role_key",
                String(64),
                ForeignKey("roles.role_key", ondelete="CASCADE"),
                primary_key=True,
            ),
            Column("menu_key", String(64), primary_key=True),
            Column("position_index", Integer, nullable=False),
        )
        self.users = Table(
            "users",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("username", String(128), nullable=False, unique=True),
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

    def create_schema(
        self,
        connection: Connection | None = None,
        *,
        identity_only: bool = False,
    ) -> None:
        bind = connection or self.engine
        if not identity_only:
            self.metadata.create_all(bind)
            return
        self.roles.create(bind, checkfirst=True)
        self.users.create(bind, checkfirst=True)
        self.role_menus.create(bind, checkfirst=True)

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
        *,
        owner_user_id: int | None = None,
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
                        "user_id": owner_user_id,
                        "position_index": index,
                        "data": json.dumps(record, ensure_ascii=False),
                        "created_at": str(record.get("created_at") or ""),
                    }
                )
            if rows:
                conn.execute(insert(self.image_records), rows)

    def read_records(
        self,
        connection: Connection | None = None,
        *,
        owner_user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = select(self.image_records.c.data).order_by(
            self.image_records.c.position_index
        )
        if owner_user_id is not None:
            query = query.where(self.image_records.c.user_id == owner_user_id)
        with self._transaction(connection) as conn:
            rows = conn.execute(query).scalars()
            return [json.loads(data) for data in rows]

    def backfill_image_owners(self, owner_user_id: int) -> int:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.image_records)
                .where(self.image_records.c.user_id.is_(None))
                .values(user_id=owner_user_id)
            )
            return int(result.rowcount or 0)

    def insert_record(
        self,
        record: dict[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> None:
        source = dict(record)
        record_id = str(source.get("id") or uuid.uuid4().hex)
        source["id"] = record_id
        with self.engine.begin() as connection:
            connection.execute(
                update(self.image_records).values(
                    position_index=self.image_records.c.position_index + 1
                )
            )
            connection.execute(
                insert(self.image_records).values(
                    id=record_id,
                    user_id=owner_user_id,
                    position_index=0,
                    data=json.dumps(source, ensure_ascii=False),
                    created_at=str(source.get("created_at") or ""),
                )
            )

    def delete_record(
        self, image_id: str, *, owner_user_id: int | None = None
    ) -> dict[str, Any] | None:
        condition = self.image_records.c.id == image_id
        if owner_user_id is not None:
            condition = condition & (self.image_records.c.user_id == owner_user_id)
        with self.engine.begin() as connection:
            row = connection.execute(
                select(
                    self.image_records.c.data,
                    self.image_records.c.position_index,
                ).where(condition)
            ).mappings().first()
            if row is None:
                return None
            connection.execute(delete(self.image_records).where(condition))
            connection.execute(
                update(self.image_records)
                .where(self.image_records.c.position_index > row["position_index"])
                .values(position_index=self.image_records.c.position_index - 1)
            )
        return json.loads(row["data"])

    def replace_roles(
        self,
        roles: dict[str, dict[str, Any]],
        connection: Connection | None = None,
    ) -> None:
        with self._transaction(connection) as conn:
            conn.execute(delete(self.role_menus))
            conn.execute(delete(self.roles))
            rows = [
                {
                    "role_key": role_key,
                    "label": str(info.get("label") or role_key),
                }
                for role_key, info in roles.items()
            ]
            if rows:
                conn.execute(insert(self.roles), rows)
                menu_rows = [
                    {
                        "role_key": role_key,
                        "menu_key": str(menu_key),
                        "position_index": position,
                    }
                    for role_key, info in roles.items()
                    for position, menu_key in enumerate(info.get("menus") or [])
                ]
                if menu_rows:
                    conn.execute(insert(self.role_menus), menu_rows)

    def read_roles(
        self, connection: Connection | None = None
    ) -> dict[str, dict[str, Any]]:
        with self._transaction(connection) as conn:
            role_rows = tuple(
                conn.execute(
                    select(self.roles.c.role_key, self.roles.c.label).order_by(
                        self.roles.c.role_key
                    )
                ).mappings()
            )
            menu_rows = tuple(
                conn.execute(
                    select(
                        self.role_menus.c.role_key,
                        self.role_menus.c.menu_key,
                    ).order_by(
                        self.role_menus.c.role_key,
                        self.role_menus.c.position_index,
                    )
                ).mappings()
            )
        menus: dict[str, list[str]] = {}
        for row in menu_rows:
            menus.setdefault(str(row["role_key"]), []).append(str(row["menu_key"]))
        return {
            str(row["role_key"]): {
                "label": str(row["label"]),
                "menus": menus.get(str(row["role_key"]), []),
            }
            for row in role_rows
        }

    def update_role_menus(self, role_key: str, menus: list[str]) -> bool:
        with self.engine.begin() as connection:
            exists = connection.execute(
                select(self.roles.c.role_key).where(self.roles.c.role_key == role_key)
            ).first()
            if exists is None:
                return False
            connection.execute(
                delete(self.role_menus).where(self.role_menus.c.role_key == role_key)
            )
            if menus:
                connection.execute(
                    insert(self.role_menus),
                    [
                        {
                            "role_key": role_key,
                            "menu_key": menu_key,
                            "position_index": position,
                        }
                        for position, menu_key in enumerate(menus)
                    ],
                )
            return True

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

    @staticmethod
    def _user_identity(row: Any) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "role": str(row["role_key"]),
        }

    def get_user_by_id(
        self, user_id: int, connection: Connection | None = None
    ) -> dict[str, Any] | None:
        with self._transaction(connection) as conn:
            row = conn.execute(
                select(
                    self.users.c.id,
                    self.users.c.username,
                    self.users.c.role_key,
                ).where(self.users.c.id == user_id)
            ).mappings().first()
        return None if row is None else self._user_identity(row)

    def get_user_by_username(
        self, username: str, connection: Connection | None = None
    ) -> dict[str, Any] | None:
        with self._transaction(connection) as conn:
            row = conn.execute(
                select(
                    self.users.c.id,
                    self.users.c.username,
                    self.users.c.role_key,
                ).where(self.users.c.username == username)
            ).mappings().first()
        return None if row is None else self._user_identity(row)

    def authenticate_user(
        self, username: str, password: str
    ) -> dict[str, Any] | None:
        row = self.get_login_identity(username)
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return self._user_identity(row)

    def get_login_identity(self, username: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    self.users.c.id,
                    self.users.c.username,
                    self.users.c.password_hash,
                    self.users.c.role_key,
                ).where(self.users.c.username == username)
            ).mappings().first()
        return None if row is None else dict(row)

    def authenticate(self, username: str, password: str) -> str | None:
        identity = self.authenticate_user(username, password)
        return None if identity is None else str(identity["role"])

    def clear_all(self, connection: Connection | None = None) -> None:
        with self._transaction(connection) as conn:
            conn.execute(delete(self.image_records))
            conn.execute(delete(self.users))
            conn.execute(delete(self.role_menus))
            conn.execute(delete(self.roles))
            conn.execute(delete(self.stats))
