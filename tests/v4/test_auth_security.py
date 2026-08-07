from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert, select

from trace_app.v4.schema import V4Tables
from trace_app.v4.security import (
    DatabaseSessionStore,
    LoginRateLimiter,
    RateLimitExceeded,
)


@pytest.fixture
def security_context():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    tables = V4Tables.build()
    tables.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(tables.users).values(id=7))
    return engine, tables


def test_sessions_store_only_token_hash_and_enforce_both_expiries(
    security_context,
) -> None:
    engine, tables = security_context
    now = datetime(2026, 7, 29, tzinfo=UTC)
    sessions = DatabaseSessionStore(
        engine,
        tables=tables,
        idle_seconds=60,
        absolute_seconds=180,
        clock=lambda: now,
    )

    sessions.clock = lambda: now
    token = sessions.issue(7)
    with engine.connect() as connection:
        row = connection.execute(select(tables.auth_sessions)).mappings().one()
    assert row["token_hash"] == hashlib.sha256(token.encode("ascii")).digest()
    assert token.encode("ascii") not in bytes(row["token_hash"])
    assert sessions.resolve(token) == 7

    sessions.clock = lambda: now + timedelta(seconds=61)
    assert sessions.resolve(token) is None

    sessions.clock = lambda: now
    token = sessions.issue(7)
    sessions.clock = lambda: now + timedelta(seconds=181)
    assert sessions.resolve(token) is None


def test_session_revocation_by_token_and_user(security_context) -> None:
    engine, tables = security_context
    sessions = DatabaseSessionStore(engine, tables=tables)
    first = sessions.issue(7)
    second = sessions.issue(7)

    sessions.revoke(first)
    assert sessions.resolve(first) is None
    assert sessions.resolve(second) == 7
    assert sessions.revoke_user(7) == 1
    assert sessions.resolve(second) is None


def test_login_rate_limit_is_atomic_by_account_and_ip(security_context) -> None:
    engine, tables = security_context
    limiter = LoginRateLimiter(
        engine,
        tables=tables,
        limit=2,
        window_seconds=300,
        clock=lambda: datetime(2026, 7, 29, tzinfo=UTC),
    )

    limiter.consume(username="alice", client_ip="203.0.113.10")
    limiter.consume(username="alice", client_ip="203.0.113.10")
    with pytest.raises(RateLimitExceeded):
        limiter.consume(username="alice", client_ip="203.0.113.10")

    with engine.connect() as connection:
        rows = connection.execute(select(tables.rate_limit_buckets)).mappings().all()
    serialized = repr(rows)
    assert "alice" not in serialized
    assert "203.0.113.10" not in serialized
