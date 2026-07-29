from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from trace_app.v4.startup import initialize_v4_schema


pytestmark = pytest.mark.postgres


def _postgres_url() -> str:
    value = os.getenv("TEST_POSTGRES_URL", "").strip()
    if not value:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    return value


def _plan(connection, sql: str, parameters: dict[str, object]) -> str:
    rows = connection.execute(
        text(f"EXPLAIN (FORMAT TEXT) {sql}"), parameters
    ).scalars()
    return "\n".join(str(row) for row in rows)


def test_exact_and_group_auth_queries_use_named_indexes_at_release_scale() -> None:
    engine = create_engine(_postgres_url())
    initialize_v4_schema(engine, require_postgres=True)
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        exact = _plan(
            connection,
            "SELECT id FROM v4_records WHERE owner_user_id=:owner "
            "AND original_file_md5=:md5 AND original_file_sha256=:sha",
            {"owner": 7, "md5": b"m" * 16, "sha": b"s" * 32},
        )
        authenticated = _plan(
            connection,
            "SELECT id FROM v4_records WHERE source_group_id=:group_id "
            "AND auth_tag=:tag AND owner_user_id=:owner",
            {
                "group_id": "00000000-0000-0000-0000-000000000001",
                "tag": b"12345678",
                "owner": 7,
            },
        )
    assert "ix_v4_original_file_hashes" in exact
    assert "uq_v4_group_auth_tag" in authenticated
