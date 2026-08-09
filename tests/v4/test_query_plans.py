from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect, text

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


def _seed_planner_statistics(connection) -> None:
    user_columns = {
        item["name"] for item in inspect(connection).get_columns("users")
    }
    if {"username", "password_hash", "role_key"} <= user_columns:
        connection.execute(
            text(
                "INSERT INTO roles (role_key, label) VALUES "
                "('v4-plan-test', 'V4 plan test') ON CONFLICT DO NOTHING"
            )
        )
        connection.execute(
            text(
                "INSERT INTO users (id, username, password_hash, role_key) VALUES "
                "(7, 'v4-plan-test-user', 'not-used', 'v4-plan-test') "
                "ON CONFLICT DO NOTHING"
            )
        )
    else:
        connection.execute(
            text("INSERT INTO users (id) VALUES (7) ON CONFLICT DO NOTHING")
        )
    connection.execute(
        text(
            "INSERT INTO source_groups ("
            "id, owner_user_id, original_image_sha256, image_width, image_height, "
            "model_version, feature_schema_version, status"
            ") VALUES ("
            "'00000000-0000-4000-8000-000000000001', 7, :sha, 100, 100, "
            "'test', 'test', 'active'"
            ") ON CONFLICT DO NOTHING"
        ),
        {"sha": b"g" * 32},
    )
    connection.execute(
        text(
            "INSERT INTO v4_records ("
            "id, source_group_id, owner_user_id, trace_id, codec, auth_tag, key_id, "
            "original_file_md5, original_file_sha256, watermarked_file_md5, "
            "watermarked_file_sha256, original_pixel_sha256, watermarked_pixel_sha256, "
            "evidence_uuid, original_filename, status"
            ") SELECT "
            "('00000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid, "
            "'00000000-0000-4000-8000-000000000001', 7, 'plan-' || g, 'test', "
            "int8send(g), 'test', decode(md5('om' || g), 'hex'), "
            "decode(md5('os1' || g) || md5('os2' || g), 'hex'), "
            "decode(md5('wm' || g), 'hex'), "
            "decode(md5('ws1' || g) || md5('ws2' || g), 'hex'), "
            "decode(md5('op1' || g) || md5('op2' || g), 'hex'), "
            "decode(md5('wp1' || g) || md5('wp2' || g), 'hex'), "
            "('10000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid, "
            "'plan-test.bin', 'active' FROM generate_series(1, 1000) AS g"
        )
    )
    connection.execute(text("ANALYZE v4_records"))


def test_exact_and_group_auth_queries_use_named_indexes_at_release_scale() -> None:
    engine = create_engine(_postgres_url())
    initialize_v4_schema(engine, require_postgres=True)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            _seed_planner_statistics(connection)
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
                    "group_id": "00000000-0000-4000-8000-000000000001",
                    "tag": b"\x00" * 7 + b"\x01",
                    "owner": 7,
                },
            )
        finally:
            transaction.rollback()
    assert "ix_v4_original_file_hashes" in exact
    assert "uq_v4_group_auth_tag" in authenticated
