from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from trace_app.v4.schema import V4Tables
from trace_app.v4.startup import (
    REQUIRED_INDEXES,
    initialize_v4_schema,
    verify_postgres_v4_schema,
)


pytestmark = pytest.mark.postgres


def _postgres_url() -> str:
    value = os.getenv("TEST_POSTGRES_URL", "").strip()
    if not value:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    return value


def test_postgres_schema_has_pgvector_hnsw_and_required_indexes() -> None:
    engine = create_engine(_postgres_url())
    tables = V4Tables.build()
    initialize_v4_schema(engine, tables=tables, require_postgres=True)
    verify_postgres_v4_schema(engine)

    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one() == 1
        definitions = dict(
            connection.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = current_schema()"
                )
            ).all()
        )
    hnsw = definitions["ix_source_group_embeddings_hnsw"].lower()
    assert "using hnsw" in hnsw
    assert "vector_cosine_ops" in hnsw
    assert "ix_v4_original_file_hashes" in definitions
    assert "ix_v4_watermarked_file_hashes" in definitions
    assert REQUIRED_INDEXES <= definitions.keys()
