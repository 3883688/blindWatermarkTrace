"""Indexed, owner-scoped persistence operations for the V4 pipeline."""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from sqlalchemy import Select, and_, delete, insert, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from trace_app.v4.domain import OwnerScope
from trace_app.v4.schema import V4Tables


class AuthTagCollision(RuntimeError):
    """A transaction hit a retryable trace ID or group-local tag constraint."""


@dataclass(frozen=True, slots=True)
class SourceGroupInput:
    owner_user_id: int
    original_image_sha256: bytes
    image_width: int
    image_height: int
    original_media_id: str | None
    model_version: str
    feature_schema_version: str
    view_policy_version: str = "v4-multiview-1"


@dataclass(frozen=True, slots=True)
class StoredSourceGroup:
    id: UUID
    owner_user_id: int
    original_image_sha256: bytes
    image_width: int
    image_height: int
    original_media_id: str | None
    model_version: str
    feature_schema_version: str
    status: str
    view_policy_version: str = "v4-multiview-1"


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    view_index: int
    view_kind: str
    embedding: bytes | Sequence[float]
    model_version: str


@dataclass(frozen=True, slots=True)
class FeatureInput:
    feature_kind: str
    schema_version: str
    model_version: str
    feature_bytes: bytes
    feature_sha256: bytes


@dataclass(frozen=True, slots=True)
class V4RecordInput:
    id: UUID
    source_group_id: UUID
    owner_user_id: int
    trace_id: str
    codec: str
    auth_tag: bytes
    key_id: str
    original_file_md5: bytes
    original_file_sha256: bytes
    watermarked_file_md5: bytes
    watermarked_file_sha256: bytes
    original_pixel_sha256: bytes
    watermarked_pixel_sha256: bytes
    output_media_id: str | None
    thumbnail_media_id: str | None
    evidence_uuid: UUID
    metadata_json: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredV4Record:
    id: UUID
    source_group_id: UUID
    owner_user_id: int
    trace_id: str
    codec: str
    auth_tag: bytes
    key_id: str
    original_file_md5: bytes
    original_file_sha256: bytes
    watermarked_file_md5: bytes
    watermarked_file_sha256: bytes
    original_pixel_sha256: bytes
    watermarked_pixel_sha256: bytes
    output_media_id: str | None
    thumbnail_media_id: str | None
    evidence_uuid: UUID
    status: str
    metadata_json: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecalledGroup:
    source_group_id: UUID
    best_distance: float
    matching_view_count: int
    distance_consistency: float


@dataclass(frozen=True, slots=True)
class StoredFeature:
    source_group_id: UUID
    owner_user_id: int
    image_width: int
    image_height: int
    feature_kind: str
    schema_version: str
    model_version: str
    feature_bytes: bytes
    feature_sha256: bytes


@dataclass(frozen=True, slots=True)
class MediaObjectInput:
    id: str
    owner_user_id: int
    variant: str
    storage_key: str
    content_type: str
    byte_size: int
    sha256: bytes
    status: str = "active"


@dataclass(frozen=True, slots=True)
class StoredMediaObject:
    id: str
    owner_user_id: int
    variant: str
    storage_key: str
    content_type: str
    byte_size: int
    sha256: bytes
    status: str


class V4Repository:
    def __init__(self, engine: Engine, *, tables: V4Tables | None = None) -> None:
        self.engine = engine
        self.tables = tables or V4Tables.build()

    def find_or_create_source_group(self, value: SourceGroupInput) -> StoredSourceGroup:
        table = self.tables.source_groups
        group_id = uuid4()
        values = {
            "id": group_id,
            **asdict(value),
            "status": "active",
        }
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                statement = postgres_insert(table).values(**values).on_conflict_do_nothing(
                    constraint="uq_source_group_owner_sha256"
                )
            elif self.engine.dialect.name == "sqlite":
                statement = sqlite_insert(table).values(**values).on_conflict_do_nothing(
                    index_elements=["owner_user_id", "original_image_sha256"]
                )
            else:
                statement = insert(table).values(**values)
            connection.execute(statement)
            row = connection.execute(
                select(table).where(
                    table.c.owner_user_id == value.owner_user_id,
                    table.c.original_image_sha256 == value.original_image_sha256,
                )
            ).mappings().one()
        return self._source_group(row)

    def find_source_group(
        self, owner_user_id: int, source_hash: bytes
    ) -> StoredSourceGroup | None:
        table = self.tables.source_groups
        with self.engine.connect() as connection:
            row = connection.execute(
                select(table).where(
                    table.c.owner_user_id == owner_user_id,
                    table.c.original_image_sha256 == source_hash,
                    table.c.status == "active",
                )
            ).mappings().first()
        return None if row is None else self._source_group(row)

    def auth_tag_exists(self, source_group_id: UUID, tag: bytes) -> bool:
        table = self.tables.v4_records
        with self.engine.connect() as connection:
            return connection.execute(
                select(table.c.id).where(
                    table.c.source_group_id == source_group_id,
                    table.c.auth_tag == tag,
                    table.c.status != "deleted",
                )
            ).first() is not None

    def commit_generation(self, unit: Any) -> tuple[StoredV4Record, bool]:
        groups = self.tables.source_groups
        embeddings = self.tables.source_group_embeddings
        features = self.tables.source_group_features
        media = self.tables.media_objects
        records = self.tables.v4_records
        audits = self.tables.audit_events
        group_values = {
            "id": unit.provisional_group_id,
            **asdict(unit.group),
            "status": "active",
        }
        try:
            with self.engine.begin() as connection:
                if self.engine.dialect.name == "postgresql":
                    group_insert = postgres_insert(groups).values(**group_values).on_conflict_do_nothing(
                        constraint="uq_source_group_owner_sha256"
                    )
                elif self.engine.dialect.name == "sqlite":
                    group_insert = sqlite_insert(groups).values(**group_values).on_conflict_do_nothing(
                        index_elements=["owner_user_id", "original_image_sha256"]
                    )
                else:
                    group_insert = insert(groups).values(**group_values)
                result = connection.execute(group_insert)
                row = connection.execute(
                    select(groups).where(
                        groups.c.owner_user_id == unit.group.owner_user_id,
                        groups.c.original_image_sha256 == unit.group.original_image_sha256,
                    )
                ).mappings().one()
                group = self._source_group(row)
                created = bool(result.rowcount) and group.id == unit.provisional_group_id
                if created:
                    if unit.group_artifacts is None:
                        raise ValueError("new source group requires model artifacts")
                    connection.execute(
                        insert(embeddings),
                        [
                            {
                                "source_group_id": group.id,
                                "owner_user_id": group.owner_user_id,
                                **asdict(item),
                                "embedding": (
                                    list(item.embedding)
                                    if self.engine.dialect.name == "postgresql"
                                    else array("f", item.embedding).tobytes()
                                ),
                            }
                            for item in unit.group_artifacts.embeddings
                        ],
                    )
                    connection.execute(
                        insert(features),
                        [
                            {"source_group_id": group.id, **asdict(item)}
                            for item in unit.group_artifacts.features
                        ],
                    )
                connection.execute(
                    insert(media), [asdict(item.media_input) for item in unit.media]
                )
                record_input = replace(unit.record, source_group_id=group.id)
                record_values = asdict(record_input)
                record_values["metadata_json"] = dict(record_input.metadata_json)
                record_values["status"] = "active"
                connection.execute(insert(records).values(**record_values))
                connection.execute(
                    insert(audits).values(
                        id=uuid4(),
                        actor_user_id=group.owner_user_id,
                        action="v4.generate",
                        target_id=str(record_input.id),
                        outcome="success",
                        correlation_id=unit.correlation_id,
                        created_at=datetime.now(UTC),
                    )
                )
            return self._record(record_values), created
        except IntegrityError as exc:
            message = str(exc).lower()
            constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
            sqlite_identity_collision = "unique constraint failed" in message and (
                (
                    "v4_records.source_group_id" in message
                    and "v4_records.auth_tag" in message
                )
                or (
                    "v4_records.owner_user_id" in message
                    and "v4_records.trace_id" in message
                )
            )
            if constraint in {"uq_v4_group_auth_tag", "uq_v4_owner_trace"} or sqlite_identity_collision:
                raise AuthTagCollision("V4 generation identity collision") from exc
            raise

    def append_generation_failure(
        self, *, owner_user_id: int, correlation_id: UUID, reason: str
    ) -> None:
        self.append_audit(
            actor_user_id=owner_user_id,
            action="v4.generate",
            target_id=None,
            outcome=reason,
            correlation_id=correlation_id,
            created_at=datetime.now(UTC),
        )

    def insert_media(self, value: MediaObjectInput) -> StoredMediaObject:
        with self.engine.begin() as connection:
            connection.execute(insert(self.tables.media_objects).values(**asdict(value)))
        return StoredMediaObject(**asdict(value))

    def get_media(self, media_id: str) -> StoredMediaObject | None:
        table = self.tables.media_objects
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    table.c.id,
                    table.c.owner_user_id,
                    table.c.variant,
                    table.c.storage_key,
                    table.c.content_type,
                    table.c.byte_size,
                    table.c.sha256,
                    table.c.status,
                ).where(table.c.id == media_id, table.c.status == "active")
            ).mappings().first()
        return None if row is None else StoredMediaObject(**dict(row))

    def insert_embeddings(
        self,
        *,
        source_group_id: UUID,
        owner_user_id: int,
        embeddings: Sequence[EmbeddingInput],
    ) -> None:
        if not embeddings:
            return
        rows = [
            {
                "source_group_id": source_group_id,
                "owner_user_id": owner_user_id,
                **asdict(item),
            }
            for item in embeddings
        ]
        with self.engine.begin() as connection:
            connection.execute(insert(self.tables.source_group_embeddings), rows)

    def put_feature(self, source_group_id: UUID, value: FeatureInput) -> None:
        table = self.tables.source_group_features
        values = {"source_group_id": source_group_id, **asdict(value)}
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                statement = postgres_insert(table).values(**values).on_conflict_do_update(
                    index_elements=["source_group_id", "feature_kind"],
                    set_={key: values[key] for key in values if key != "source_group_id"},
                )
            elif self.engine.dialect.name == "sqlite":
                statement = sqlite_insert(table).values(**values).on_conflict_do_update(
                    index_elements=["source_group_id", "feature_kind"],
                    set_={key: values[key] for key in values if key != "source_group_id"},
                )
            else:
                existing = connection.execute(
                    select(table.c.source_group_id).where(
                        table.c.source_group_id == source_group_id,
                        table.c.feature_kind == value.feature_kind,
                    )
                ).first()
                if existing:
                    connection.execute(
                        update(table)
                        .where(
                            table.c.source_group_id == source_group_id,
                            table.c.feature_kind == value.feature_kind,
                        )
                        .values(**values)
                    )
                    return
                statement = insert(table).values(**values)
            connection.execute(statement)

    def get_features_for_groups(
        self,
        scope: OwnerScope,
        source_group_ids: Sequence[UUID],
    ) -> tuple[StoredFeature, ...]:
        group_ids = tuple(dict.fromkeys(source_group_ids))
        if len(group_ids) > 40:
            raise ValueError("at most 40 recalled source groups may load features")
        if not group_ids:
            return ()
        features = self.tables.source_group_features
        groups = self.tables.source_groups
        conditions = [groups.c.id.in_(group_ids), groups.c.status == "active"]
        if scope.query_owner_id is not None:
            conditions.append(groups.c.owner_user_id == scope.query_owner_id)
        statement = (
            select(
                features.c.source_group_id,
                groups.c.owner_user_id,
                groups.c.image_width,
                groups.c.image_height,
                features.c.feature_kind,
                features.c.schema_version,
                features.c.model_version,
                features.c.feature_bytes,
                features.c.feature_sha256,
            )
            .select_from(
                features.join(groups, features.c.source_group_id == groups.c.id)
            )
            .where(*conditions)
            .order_by(features.c.source_group_id, features.c.feature_kind)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings()
            return tuple(StoredFeature(**dict(row)) for row in rows)

    def insert_record(self, value: V4RecordInput) -> StoredV4Record:
        values = asdict(value)
        values["metadata_json"] = dict(value.metadata_json)
        values["status"] = "active"
        with self.engine.begin() as connection:
            connection.execute(insert(self.tables.v4_records).values(**values))
        return self._record({**values})

    def exact_file_statement(self) -> Select:
        table = self.tables.v4_records
        return select(*self._record_columns()).where(
            table.c.owner_user_id == text(":owner_user_id"),
            or_(
                and_(
                    table.c.original_file_md5 == text(":md5"),
                    table.c.original_file_sha256 == text(":sha256"),
                ),
                and_(
                    table.c.watermarked_file_md5 == text(":md5"),
                    table.c.watermarked_file_sha256 == text(":sha256"),
                ),
            ),
            table.c.status == "active",
        )

    def find_exact_file(
        self, scope: OwnerScope, *, md5: bytes, sha256: bytes
    ) -> tuple[StoredV4Record, ...]:
        table = self.tables.v4_records
        conditions = [
            or_(
                and_(table.c.original_file_md5 == md5, table.c.original_file_sha256 == sha256),
                and_(
                    table.c.watermarked_file_md5 == md5,
                    table.c.watermarked_file_sha256 == sha256,
                ),
            ),
            table.c.status == "active",
        ]
        if scope.query_owner_id is not None:
            conditions.append(table.c.owner_user_id == scope.query_owner_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(*self._record_columns()).where(*conditions)
            ).mappings()
            return tuple(self._record(row) for row in rows)

    def find_record_by_auth_tag(
        self,
        scope: OwnerScope,
        *,
        source_group_id: UUID,
        auth_tag: bytes,
    ) -> StoredV4Record | None:
        table = self.tables.v4_records
        conditions = [
            table.c.source_group_id == source_group_id,
            table.c.auth_tag == auth_tag,
            table.c.status == "active",
        ]
        if scope.query_owner_id is not None:
            conditions.append(table.c.owner_user_id == scope.query_owner_id)
        with self.engine.connect() as connection:
            row = connection.execute(
                select(*self._record_columns()).where(*conditions)
            ).mappings().first()
        return None if row is None else self._record(row)

    def list_records(self, scope: OwnerScope) -> tuple[StoredV4Record, ...]:
        table = self.tables.v4_records
        statement = select(table).where(table.c.status != "deleted")
        if scope.query_owner_id is not None:
            statement = statement.where(table.c.owner_user_id == scope.query_owner_id)
        statement = statement.order_by(table.c.created_at.desc(), table.c.id)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings()
            return tuple(self._record(row) for row in rows)

    def delete_record(self, scope: OwnerScope, record_id: UUID) -> bool:
        table = self.tables.v4_records
        conditions = [table.c.id == record_id]
        if scope.query_owner_id is not None:
            conditions.append(table.c.owner_user_id == scope.query_owner_id)
        with self.engine.begin() as connection:
            result = connection.execute(delete(table).where(*conditions))
        return bool(result.rowcount)

    def increment_counter(self, owner_user_id: int, key: str, delta: int = 1) -> int:
        if delta <= 0:
            raise ValueError("counter delta must be positive")
        table = self.tables.v4_counters
        now = datetime.now(UTC)
        values = {
            "owner_user_id": owner_user_id,
            "counter_key": key,
            "counter_value": delta,
            "updated_at": now,
        }
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                statement = postgres_insert(table).values(**values).on_conflict_do_update(
                    index_elements=["owner_user_id", "counter_key"],
                    set_={
                        "counter_value": table.c.counter_value + delta,
                        "updated_at": now,
                    },
                ).returning(table.c.counter_value)
            elif self.engine.dialect.name == "sqlite":
                statement = sqlite_insert(table).values(**values).on_conflict_do_update(
                    index_elements=["owner_user_id", "counter_key"],
                    set_={
                        "counter_value": table.c.counter_value + delta,
                        "updated_at": now,
                    },
                ).returning(table.c.counter_value)
            else:
                current = connection.execute(
                    select(table.c.counter_value).where(
                        table.c.owner_user_id == owner_user_id,
                        table.c.counter_key == key,
                    )
                ).scalar_one_or_none()
                if current is None:
                    connection.execute(insert(table).values(**values))
                    return delta
                result = current + delta
                connection.execute(
                    update(table)
                    .where(
                        table.c.owner_user_id == owner_user_id,
                        table.c.counter_key == key,
                    )
                    .values(counter_value=result, updated_at=now)
                )
                return result
            return int(connection.execute(statement).scalar_one())

    def append_audit(
        self,
        *,
        actor_user_id: int | None,
        action: str,
        target_id: str | None,
        outcome: str,
        correlation_id: UUID,
        created_at: datetime,
    ) -> UUID:
        event_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                insert(self.tables.audit_events).values(
                    id=event_id,
                    actor_user_id=actor_user_id,
                    action=action,
                    target_id=target_id,
                    outcome=outcome,
                    correlation_id=correlation_id,
                    created_at=created_at,
                )
            )
        return event_id

    def recall_groups(
        self,
        scope: OwnerScope,
        embedding: Sequence[float],
        *,
        group_limit: int = 40,
        neighbor_limit: int = 400,
    ) -> tuple[RecalledGroup, ...]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("pgvector recall requires PostgreSQL")
        if len(embedding) != 384:
            raise ValueError("DINO embedding must contain 384 values")
        if not 1 <= group_limit <= 40 or neighbor_limit < group_limit:
            raise ValueError("invalid V4 recall limits")
        owner_clause = "" if scope.query_owner_id is None else "WHERE owner_user_id = :owner_user_id"
        statement = text(
            f"""
            WITH neighbors AS (
                SELECT source_group_id, embedding <=> CAST(:embedding AS vector) AS distance
                FROM source_group_embeddings
                {owner_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :neighbor_limit
            ), grouped AS (
                SELECT source_group_id, MIN(distance) AS best_distance,
                       COUNT(*) AS matching_view_count,
                       COALESCE(STDDEV_POP(distance), 0) AS distance_consistency
                FROM neighbors GROUP BY source_group_id
            )
            SELECT source_group_id, best_distance, matching_view_count, distance_consistency
            FROM grouped
            ORDER BY best_distance ASC, matching_view_count DESC,
                     distance_consistency ASC, source_group_id ASC
            LIMIT :group_limit
            """
        )
        parameters: dict[str, Any] = {
            "embedding": list(embedding),
            "neighbor_limit": neighbor_limit,
            "group_limit": group_limit,
        }
        if scope.query_owner_id is not None:
            parameters["owner_user_id"] = scope.query_owner_id
        with self.engine.connect() as connection:
            rows = connection.execute(statement, parameters).mappings()
            return tuple(
                RecalledGroup(
                    source_group_id=row["source_group_id"],
                    best_distance=float(row["best_distance"]),
                    matching_view_count=int(row["matching_view_count"]),
                    distance_consistency=float(row["distance_consistency"]),
                )
                for row in rows
            )

    def _record_columns(self):
        table = self.tables.v4_records
        return tuple(column for column in table.c if column.name != "metadata_json")

    @staticmethod
    def _source_group(row: Mapping[str, Any]) -> StoredSourceGroup:
        return StoredSourceGroup(
            **{
                key: row[key]
                for key in StoredSourceGroup.__dataclass_fields__
            }
        )

    @staticmethod
    def _record(row: Mapping[str, Any]) -> StoredV4Record:
        values = {
            key: row[key]
            for key in StoredV4Record.__dataclass_fields__
            if key in row
        }
        values["metadata_json"] = MappingProxyType(dict(values.get("metadata_json", {})))
        return StoredV4Record(**values)


__all__ = (
    "AuthTagCollision",
    "EmbeddingInput",
    "FeatureInput",
    "MediaObjectInput",
    "RecalledGroup",
    "SourceGroupInput",
    "StoredSourceGroup",
    "StoredMediaObject",
    "StoredFeature",
    "StoredV4Record",
    "V4RecordInput",
    "V4Repository",
)
