from __future__ import annotations

from sqlalchemy import create_engine

from trace_app.v4.schema import V4Tables
from trace_app.v4.startup import verify_portable_v4_schema


EXPECTED_TABLES = {
    "users",
    "source_groups",
    "source_group_embeddings",
    "source_group_features",
    "v4_records",
    "media_objects",
    "auth_sessions",
    "rate_limit_buckets",
    "v4_counters",
    "audit_events",
    "deep_forensics_jobs",
}


def test_v4_schema_defines_all_relational_tables_and_constraints() -> None:
    tables = V4Tables.build()

    assert set(tables.metadata.tables) == EXPECTED_TABLES
    assert "uq_source_group_owner_sha256" in {
        item.name for item in tables.source_groups.constraints
    }
    assert {"uq_v4_owner_trace", "uq_v4_group_auth_tag"} <= {
        item.name for item in tables.v4_records.constraints
    }
    assert tables.v4_records.c.auth_tag.type.length == 8
    assert tables.v4_records.c.metadata_json.type.__class__.__name__ == "JSON"


def test_v4_schema_has_required_lookup_and_owner_indexes() -> None:
    tables = V4Tables.build()
    indexes = {
        index.name
        for table in tables.metadata.tables.values()
        for index in table.indexes
    }

    assert {
        "ix_v4_original_file_hashes",
        "ix_v4_watermarked_file_hashes",
        "ix_v4_records_owner_created",
        "ix_source_group_embeddings_owner",
        "ix_source_group_embeddings_hnsw",
    } <= indexes


def test_v4_schema_creates_on_sqlite_with_binary_embedding_adapter() -> None:
    tables = V4Tables.build()
    engine = create_engine("sqlite+pysqlite:///:memory:")

    tables.create_all(engine)
    verify_portable_v4_schema(engine)

    assert tables.source_group_embeddings.c.embedding.type.load_dialect_impl(
        engine.dialect
    ).__class__.__name__ == "LargeBinary"
    assert set(tables.metadata.tables) == EXPECTED_TABLES
