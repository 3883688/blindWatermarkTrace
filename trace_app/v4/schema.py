"""Relational schema for the V4-only production pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeDecorator


class Embedding384(TypeDecorator):
    """Use pgvector in production and opaque bytes in SQLite contract tests."""

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(384))
        return dialect.type_descriptor(LargeBinary())


def _timestamps() -> tuple[Column, Column]:
    return (
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.current_timestamp(),
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
    )


@dataclass(frozen=True, slots=True)
class V4Tables:
    metadata: MetaData
    users: Table
    source_groups: Table
    source_group_embeddings: Table
    source_group_features: Table
    v4_records: Table
    media_objects: Table
    auth_sessions: Table
    rate_limit_buckets: Table
    v4_counters: Table
    audit_events: Table
    deep_forensics_jobs: Table

    @classmethod
    def build(cls) -> "V4Tables":
        metadata = MetaData()
        users = Table("users", metadata, Column("id", Integer, primary_key=True))
        source_groups = Table(
            "source_groups",
            metadata,
            Column("id", Uuid, primary_key=True),
            Column("owner_user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("original_image_sha256", LargeBinary(32), nullable=False),
            Column("image_width", Integer, nullable=False),
            Column("image_height", Integer, nullable=False),
            Column("original_media_id", String(64), nullable=True),
            Column("model_version", String(64), nullable=False),
            Column("feature_schema_version", String(32), nullable=False),
            Column("status", String(24), nullable=False),
            *_timestamps(),
            UniqueConstraint(
                "owner_user_id",
                "original_image_sha256",
                name="uq_source_group_owner_sha256",
            ),
            UniqueConstraint(
                "id", "owner_user_id", name="uq_source_group_id_owner"
            ),
            CheckConstraint(
                "image_width > 0 AND image_height > 0",
                name="ck_source_group_dimensions",
            ),
            CheckConstraint(
                "length(original_image_sha256) = 32",
                name="ck_source_group_sha256_length",
            ),
            CheckConstraint(
                "status IN ('staging', 'active', 'disabled')",
                name="ck_source_group_status",
            ),
        )
        source_group_embeddings = Table(
            "source_group_embeddings",
            metadata,
            Column(
                "source_group_id",
                Uuid,
                primary_key=True,
            ),
            Column("view_index", Integer, primary_key=True),
            Column("owner_user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("view_kind", String(32), nullable=False),
            Column("embedding", Embedding384(), nullable=False),
            Column("model_version", String(64), nullable=False),
            ForeignKeyConstraint(
                ["source_group_id", "owner_user_id"],
                ["source_groups.id", "source_groups.owner_user_id"],
                ondelete="CASCADE",
                name="fk_embedding_source_group_owner",
            ),
            CheckConstraint("view_index >= 0", name="ck_embedding_view_index"),
        )
        Index(
            "ix_source_group_embeddings_owner",
            source_group_embeddings.c.owner_user_id,
        )
        Index(
            "ix_source_group_embeddings_hnsw",
            source_group_embeddings.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ).ddl_if(dialect="postgresql")

        source_group_features = Table(
            "source_group_features",
            metadata,
            Column(
                "source_group_id",
                Uuid,
                ForeignKey("source_groups.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            Column("feature_kind", String(16), primary_key=True),
            Column("schema_version", String(32), nullable=False),
            Column("model_version", String(64), nullable=False),
            Column("feature_bytes", LargeBinary, nullable=False),
            Column("feature_sha256", LargeBinary(32), nullable=False),
            CheckConstraint(
                "feature_kind IN ('orb', 'superpoint')",
                name="ck_source_group_feature_kind",
            ),
            CheckConstraint(
                "length(feature_sha256) = 32",
                name="ck_source_group_feature_sha256_length",
            ),
        )

        media_objects = Table(
            "media_objects",
            metadata,
            Column("id", String(64), primary_key=True),
            Column("owner_user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("variant", String(16), nullable=False),
            Column("storage_key", String(512), nullable=False, unique=True),
            Column("content_type", String(128), nullable=False),
            Column("byte_size", BigInteger, nullable=False),
            Column("sha256", LargeBinary(32), nullable=False),
            Column("status", String(24), nullable=False),
            *_timestamps(),
            CheckConstraint(
                "variant IN ('original', 'watermarked', 'thumbnail')",
                name="ck_media_variant",
            ),
            CheckConstraint("byte_size >= 0", name="ck_media_byte_size"),
            CheckConstraint("length(sha256) = 32", name="ck_media_sha256_length"),
            CheckConstraint(
                "status IN ('staged', 'active', 'deleted')",
                name="ck_media_status",
            ),
        )

        json_type = JSON().with_variant(JSONB(), "postgresql")
        v4_records = Table(
            "v4_records",
            metadata,
            Column("id", Uuid, primary_key=True),
            Column(
                "source_group_id",
                Uuid,
                nullable=False,
            ),
            Column("owner_user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("trace_id", String(128), nullable=False),
            Column("codec", String(96), nullable=False),
            Column("auth_tag", LargeBinary(8), nullable=False),
            Column("key_id", String(64), nullable=False),
            Column("original_file_md5", LargeBinary(16), nullable=False),
            Column("original_file_sha256", LargeBinary(32), nullable=False),
            Column("watermarked_file_md5", LargeBinary(16), nullable=False),
            Column("watermarked_file_sha256", LargeBinary(32), nullable=False),
            Column("original_pixel_sha256", LargeBinary(32), nullable=False),
            Column("watermarked_pixel_sha256", LargeBinary(32), nullable=False),
            Column("output_media_id", String(64), ForeignKey("media_objects.id")),
            Column("thumbnail_media_id", String(64), ForeignKey("media_objects.id")),
            Column("evidence_uuid", Uuid, nullable=False),
            Column("status", String(24), nullable=False),
            Column("metadata_json", json_type, nullable=False, default=dict),
            *_timestamps(),
            ForeignKeyConstraint(
                ["source_group_id", "owner_user_id"],
                ["source_groups.id", "source_groups.owner_user_id"],
                ondelete="CASCADE",
                name="fk_v4_record_source_group_owner",
            ),
            UniqueConstraint("owner_user_id", "trace_id", name="uq_v4_owner_trace"),
            UniqueConstraint(
                "source_group_id", "auth_tag", name="uq_v4_group_auth_tag"
            ),
            CheckConstraint(
                "status IN ('staging', 'active', 'disabled', 'deleted')",
                name="ck_v4_record_status",
            ),
            CheckConstraint("length(auth_tag) = 8", name="ck_v4_auth_tag_length"),
            CheckConstraint(
                "length(original_file_md5) = 16 AND "
                "length(watermarked_file_md5) = 16",
                name="ck_v4_md5_lengths",
            ),
            CheckConstraint(
                "length(original_file_sha256) = 32 AND "
                "length(watermarked_file_sha256) = 32 AND "
                "length(original_pixel_sha256) = 32 AND "
                "length(watermarked_pixel_sha256) = 32",
                name="ck_v4_sha256_lengths",
            ),
        )
        Index(
            "ix_v4_original_file_hashes",
            v4_records.c.owner_user_id,
            v4_records.c.original_file_md5,
            v4_records.c.original_file_sha256,
        )
        Index(
            "ix_v4_watermarked_file_hashes",
            v4_records.c.owner_user_id,
            v4_records.c.watermarked_file_md5,
            v4_records.c.watermarked_file_sha256,
        )
        Index(
            "ix_v4_records_owner_created",
            v4_records.c.owner_user_id,
            v4_records.c.created_at,
        )
        Index("ix_v4_records_source_group", v4_records.c.source_group_id)
        Index("ix_v4_records_codec", v4_records.c.codec)
        Index("ix_v4_records_key_id", v4_records.c.key_id)

        auth_sessions = Table(
            "auth_sessions",
            metadata,
            Column("token_hash", LargeBinary(32), primary_key=True),
            Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("idle_expires_at", DateTime(timezone=True), nullable=False),
            Column("absolute_expires_at", DateTime(timezone=True), nullable=False),
            Column("last_used_at", DateTime(timezone=True), nullable=False),
            Column("revoked_at", DateTime(timezone=True)),
            CheckConstraint(
                "length(token_hash) = 32", name="ck_auth_session_token_hash_length"
            ),
        )
        Index("ix_auth_sessions_user", auth_sessions.c.user_id)

        rate_limit_buckets = Table(
            "rate_limit_buckets",
            metadata,
            Column("bucket_key", LargeBinary(32), primary_key=True),
            Column("endpoint_class", String(64), nullable=False),
            Column("window_start", DateTime(timezone=True), nullable=False),
            Column("request_count", Integer, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            CheckConstraint("request_count >= 0", name="ck_rate_limit_count"),
        )
        v4_counters = Table(
            "v4_counters",
            metadata,
            Column("owner_user_id", Integer, ForeignKey("users.id"), primary_key=True),
            Column("counter_key", String(64), primary_key=True),
            Column("counter_value", BigInteger, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            CheckConstraint("counter_value >= 0", name="ck_v4_counter_value"),
        )
        audit_events = Table(
            "audit_events",
            metadata,
            Column("id", Uuid, primary_key=True),
            Column("actor_user_id", Integer, ForeignKey("users.id")),
            Column("action", String(96), nullable=False),
            Column("target_id", String(128)),
            Column("outcome", String(32), nullable=False),
            Column("correlation_id", Uuid, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        Index("ix_audit_actor_created", audit_events.c.actor_user_id, audit_events.c.created_at)
        deep_forensics_jobs = Table(
            "deep_forensics_jobs",
            metadata,
            Column("id", Uuid, primary_key=True),
            Column("owner_user_id", Integer, ForeignKey("users.id"), nullable=False),
            Column("status", String(24), nullable=False),
            Column("progress", Integer, nullable=False),
            Column("lease_owner", String(128)),
            Column("lease_expires_at", DateTime(timezone=True)),
            Column("deadline_at", DateTime(timezone=True), nullable=False),
            Column("result_json", json_type),
            *_timestamps(),
            CheckConstraint("progress BETWEEN 0 AND 100", name="ck_deep_job_progress"),
            CheckConstraint(
                "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
                name="ck_deep_job_status",
            ),
        )
        Index("ix_deep_jobs_owner_created", deep_forensics_jobs.c.owner_user_id, deep_forensics_jobs.c.created_at)

        return cls(
            metadata,
            users,
            source_groups,
            source_group_embeddings,
            source_group_features,
            v4_records,
            media_objects,
            auth_sessions,
            rate_limit_buckets,
            v4_counters,
            audit_events,
            deep_forensics_jobs,
        )

    def create_all(self, engine: Engine) -> None:
        self.metadata.create_all(engine)


__all__ = ("Embedding384", "V4Tables")
