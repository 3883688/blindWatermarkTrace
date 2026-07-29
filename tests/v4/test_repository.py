from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, insert
from sqlalchemy.exc import IntegrityError

from trace_app.v4.domain import OwnerScope
from trace_app.v4.repository import (
    EmbeddingInput,
    FeatureInput,
    SourceGroupInput,
    V4RecordInput,
    V4Repository,
)
from trace_app.v4.schema import V4Tables


@pytest.fixture
def repository() -> V4Repository:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    tables = V4Tables.build()
    tables.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(tables.users), [{"id": 7}, {"id": 8}])
    return V4Repository(engine, tables=tables)


def _group(repo: V4Repository, owner: int, digest: bytes):
    return repo.find_or_create_source_group(
        SourceGroupInput(
            owner_user_id=owner,
            original_image_sha256=digest,
            image_width=640,
            image_height=480,
            original_media_id=None,
            model_version="dinov2-vits14",
            feature_schema_version="features-v1",
        )
    )


def _record(group_id, owner: int, trace_id: str, tag: bytes) -> V4RecordInput:
    return V4RecordInput(
        id=uuid4(),
        source_group_id=group_id,
        owner_user_id=owner,
        trace_id=trace_id,
        codec="hmac64_rs_16_8_split_repeat_sync_v4",
        auth_tag=tag,
        key_id="key-1",
        original_file_md5=b"o" * 16,
        original_file_sha256=b"O" * 32,
        watermarked_file_md5=b"w" * 16,
        watermarked_file_sha256=b"W" * 32,
        original_pixel_sha256=b"p" * 32,
        watermarked_pixel_sha256=b"P" * 32,
        output_media_id=None,
        thumbnail_media_id=None,
        evidence_uuid=uuid4(),
        metadata_json={"display": "allowed"},
    )


def test_source_group_is_reused_only_inside_owner_boundary(repository: V4Repository) -> None:
    alice = _group(repository, 7, b"s" * 32)
    same = _group(repository, 7, b"s" * 32)
    bob = _group(repository, 8, b"s" * 32)

    assert same.id == alice.id
    assert bob.id != alice.id
    assert alice.owner_user_id == 7


def test_exact_and_auth_queries_are_owner_scoped(repository: V4Repository) -> None:
    alice_group = _group(repository, 7, b"a" * 32)
    bob_group = _group(repository, 8, b"b" * 32)
    alice = _record(alice_group.id, 7, "TR-ALICE", b"12345678")
    bob = _record(bob_group.id, 8, "TR-BOB", b"87654321")
    repository.insert_record(alice)
    repository.insert_record(bob)

    exact = repository.find_exact_file(
        OwnerScope(7), md5=alice.original_file_md5, sha256=alice.original_file_sha256
    )
    found = repository.find_record_by_auth_tag(
        OwnerScope(7), source_group_id=alice_group.id, auth_tag=alice.auth_tag
    )
    hidden = repository.find_record_by_auth_tag(
        OwnerScope(8), source_group_id=alice_group.id, auth_tag=alice.auth_tag
    )

    assert {item.trace_id for item in exact} == {"TR-ALICE"}
    assert found is not None and found.trace_id == "TR-ALICE"
    assert hidden is None
    assert {item.trace_id for item in repository.list_records(OwnerScope(7))} == {
        "TR-ALICE"
    }
    assert {item.trace_id for item in repository.list_records(OwnerScope(7, True))} == {
        "TR-ALICE",
        "TR-BOB",
    }


def test_group_auth_collision_is_enforced_by_database(repository: V4Repository) -> None:
    group = _group(repository, 7, b"a" * 32)
    repository.insert_record(_record(group.id, 7, "TR-1", b"12345678"))

    with pytest.raises(IntegrityError):
        repository.insert_record(_record(group.id, 7, "TR-2", b"12345678"))


def test_auth_tag_must_be_exactly_eight_bytes(repository: V4Repository) -> None:
    group = _group(repository, 7, b"a" * 32)

    with pytest.raises(IntegrityError):
        repository.insert_record(_record(group.id, 7, "TR-BAD", b"short"))


def test_record_and_embedding_owner_must_match_source_group(
    repository: V4Repository,
) -> None:
    group = _group(repository, 7, b"a" * 32)

    with pytest.raises(IntegrityError):
        repository.insert_record(_record(group.id, 8, "TR-CROSS", b"12345678"))
    with pytest.raises(IntegrityError):
        repository.insert_embeddings(
            source_group_id=group.id,
            owner_user_id=8,
            embeddings=(
                EmbeddingInput(0, "full", b"vector", "dinov2-vits14"),
            ),
        )


def test_delete_and_atomic_counter_respect_owner_scope(repository: V4Repository) -> None:
    group = _group(repository, 7, b"a" * 32)
    record = _record(group.id, 7, "TR-1", b"12345678")
    repository.insert_record(record)

    assert repository.delete_record(OwnerScope(8), record.id) is False
    assert repository.delete_record(OwnerScope(7), record.id) is True
    assert repository.increment_counter(7, "generation", 2) == 2
    assert repository.increment_counter(7, "generation", 3) == 5


def test_repository_has_no_legacy_full_record_api(repository: V4Repository) -> None:
    assert not hasattr(repository, "read_records")
    assert "metadata_json" not in str(repository.exact_file_statement()).lower()


def test_geometry_features_load_only_recalled_owner_scoped_groups(
    repository: V4Repository,
) -> None:
    alice = _group(repository, 7, b"a" * 32)
    bob = _group(repository, 8, b"b" * 32)
    for group in (alice, bob):
        repository.put_feature(
            group.id,
            FeatureInput("orb", "1", "orb-v1", b"feature", b"f" * 32),
        )

    rows = repository.get_features_for_groups(OwnerScope(7), (alice.id, bob.id))
    admin_rows = repository.get_features_for_groups(
        OwnerScope(7, cross_owner=True), (alice.id, bob.id)
    )

    assert {(row.source_group_id, row.owner_user_id) for row in rows} == {(alice.id, 7)}
    assert {row.source_group_id for row in admin_rows} == {alice.id, bob.id}
    with pytest.raises(ValueError, match="40"):
        repository.get_features_for_groups(
            OwnerScope(7), tuple(uuid4() for _ in range(41))
        )


def test_audit_rows_accept_only_explicit_safe_columns(repository: V4Repository) -> None:
    correlation_id = uuid4()
    event_id = repository.append_audit(
        actor_user_id=7,
        action="v4.generate",
        target_id="record-id",
        outcome="success",
        correlation_id=correlation_id,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )

    assert event_id is not None
