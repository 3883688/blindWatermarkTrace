"""Creation and fail-closed verification for the V4 relational schema."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from trace_app.v4.schema import V4Tables


REQUIRED_INDEXES = frozenset(
    {
        "ix_source_group_embeddings_hnsw",
        "ix_source_group_embeddings_owner",
        "ix_v4_original_file_hashes",
        "ix_v4_watermarked_file_hashes",
        "ix_v4_records_owner_created",
        "ix_v4_records_source_group",
        "ix_v4_records_codec",
        "ix_v4_records_key_id",
        "uq_source_group_owner_sha256",
        "uq_v4_owner_trace",
        "uq_v4_group_auth_tag",
    }
)


def initialize_v4_schema(
    engine: Engine,
    *,
    tables: V4Tables | None = None,
    require_postgres: bool = False,
) -> V4Tables:
    schema = tables or V4Tables.build()
    is_postgres = engine.dialect.name == "postgresql"
    if require_postgres and not is_postgres:
        raise RuntimeError("V4 production requires PostgreSQL with pgvector")
    if is_postgres:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    schema.create_all(engine)
    if is_postgres:
        verify_postgres_v4_schema(engine)
    return schema


def verify_postgres_v4_schema(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        raise RuntimeError("V4 production requires PostgreSQL with pgvector")
    with engine.connect() as connection:
        has_vector = connection.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        ).scalar_one()
        rows = connection.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = current_schema()"
            )
        ).all()
    if not has_vector:
        raise RuntimeError("Required PostgreSQL extension pgvector is unavailable")
    definitions = {str(name): str(definition).lower() for name, definition in rows}
    missing = REQUIRED_INDEXES - definitions.keys()
    if missing:
        raise RuntimeError(f"Required V4 indexes are unavailable: {sorted(missing)}")
    hnsw = definitions["ix_source_group_embeddings_hnsw"]
    if "using hnsw" not in hnsw or "vector_cosine_ops" not in hnsw:
        raise RuntimeError("V4 pgvector HNSW cosine index is invalid")


def verify_portable_v4_schema(engine: Engine) -> None:
    names = set(inspect(engine).get_table_names())
    required = set(V4Tables.build().metadata.tables)
    if missing := required - names:
        raise RuntimeError(f"Required V4 tables are unavailable: {sorted(missing)}")


__all__ = (
    "REQUIRED_INDEXES",
    "initialize_v4_schema",
    "verify_portable_v4_schema",
    "verify_postgres_v4_schema",
)
