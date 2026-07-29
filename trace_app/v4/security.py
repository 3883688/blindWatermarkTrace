"""Database-backed sessions and atomic login throttling."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Callable

from sqlalchemy import and_, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from trace_app.v4.schema import V4Tables


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class DatabaseSessionStore:
    def __init__(
        self,
        engine: Engine,
        *,
        tables: V4Tables | None = None,
        idle_seconds: int = 1800,
        absolute_seconds: int = 86400,
        clock: Clock = _utc_now,
    ) -> None:
        if idle_seconds <= 0 or absolute_seconds < idle_seconds:
            raise ValueError("invalid session expiry configuration")
        self.engine = engine
        self.tables = tables or V4Tables.build()
        self.idle_seconds = idle_seconds
        self.absolute_seconds = absolute_seconds
        self.clock = clock

    @staticmethod
    def token_hash(token: str) -> bytes:
        return hashlib.sha256(token.encode("ascii")).digest()

    def issue(self, user_id: int) -> str:
        now = _aware(self.clock())
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            connection.execute(
                insert(self.tables.auth_sessions).values(
                    token_hash=self.token_hash(token),
                    user_id=user_id,
                    created_at=now,
                    idle_expires_at=now + timedelta(seconds=self.idle_seconds),
                    absolute_expires_at=now
                    + timedelta(seconds=self.absolute_seconds),
                    last_used_at=now,
                    revoked_at=None,
                )
            )
        return token

    def resolve(self, token: str) -> int | None:
        digest = self.token_hash(token)
        now = _aware(self.clock())
        table = self.tables.auth_sessions
        with self.engine.begin() as connection:
            statement = select(table).where(table.c.token_hash == digest)
            if self.engine.dialect.name == "postgresql":
                statement = statement.with_for_update()
            row = connection.execute(statement).mappings().first()
            if row is None or row["revoked_at"] is not None:
                return None
            idle_expiry = _aware(row["idle_expires_at"])
            absolute_expiry = _aware(row["absolute_expires_at"])
            if now >= idle_expiry or now >= absolute_expiry:
                connection.execute(
                    update(table)
                    .where(table.c.token_hash == digest)
                    .values(revoked_at=now)
                )
                return None
            next_idle = min(
                now + timedelta(seconds=self.idle_seconds), absolute_expiry
            )
            connection.execute(
                update(table)
                .where(
                    table.c.token_hash == digest,
                    table.c.revoked_at.is_(None),
                )
                .values(last_used_at=now, idle_expires_at=next_idle)
            )
            return int(row["user_id"])

    def revoke(self, token: str) -> bool:
        now = _aware(self.clock())
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.tables.auth_sessions)
                .where(
                    self.tables.auth_sessions.c.token_hash == self.token_hash(token),
                    self.tables.auth_sessions.c.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
        return bool(result.rowcount)

    def revoke_user(self, user_id: int) -> int:
        now = _aware(self.clock())
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.tables.auth_sessions)
                .where(
                    self.tables.auth_sessions.c.user_id == user_id,
                    self.tables.auth_sessions.c.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
        return int(result.rowcount or 0)


class RateLimitExceeded(RuntimeError):
    pass


class LoginRateLimiter:
    def __init__(
        self,
        engine: Engine,
        *,
        tables: V4Tables | None = None,
        limit: int = 5,
        window_seconds: int = 300,
        clock: Clock = _utc_now,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("invalid login rate limit configuration")
        self.engine = engine
        self.tables = tables or V4Tables.build()
        self.limit = limit
        self.window_seconds = window_seconds
        self.clock = clock

    def consume(self, *, username: str, client_ip: str) -> None:
        now = _aware(self.clock())
        epoch = int(now.timestamp())
        start_epoch = epoch - (epoch % self.window_seconds)
        window_start = datetime.fromtimestamp(start_epoch, UTC)
        subjects = (("account", username.casefold()), ("ip", client_ip))
        counts: list[int] = []
        table = self.tables.rate_limit_buckets
        with self.engine.begin() as connection:
            for kind, value in subjects:
                key = hashlib.sha256(
                    f"login\n{kind}\n{value}\n{start_epoch}".encode("utf-8")
                ).digest()
                values = {
                    "bucket_key": key,
                    "endpoint_class": f"login:{kind}",
                    "window_start": window_start,
                    "request_count": 1,
                    "updated_at": now,
                }
                if self.engine.dialect.name == "postgresql":
                    statement = postgres_insert(table).values(**values)
                elif self.engine.dialect.name == "sqlite":
                    statement = sqlite_insert(table).values(**values)
                else:
                    current = connection.execute(
                        select(table.c.request_count).where(table.c.bucket_key == key)
                    ).scalar_one_or_none()
                    if current is None:
                        connection.execute(insert(table).values(**values))
                        counts.append(1)
                    else:
                        count = int(current) + 1
                        connection.execute(
                            update(table)
                            .where(table.c.bucket_key == key)
                            .values(request_count=count, updated_at=now)
                        )
                        counts.append(count)
                    continue
                statement = statement.on_conflict_do_update(
                    index_elements=["bucket_key"],
                    set_={
                        "request_count": table.c.request_count + 1,
                        "updated_at": now,
                    },
                ).returning(table.c.request_count)
                counts.append(int(connection.execute(statement).scalar_one()))
        if any(count > self.limit for count in counts):
            raise RateLimitExceeded("login rate limit exceeded")


__all__ = (
    "DatabaseSessionStore",
    "LoginRateLimiter",
    "RateLimitExceeded",
)
