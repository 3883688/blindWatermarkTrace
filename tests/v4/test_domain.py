from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

import pytest

from trace_app.v4.domain import (
    DetectionEvidence,
    DetectionOutcome,
    DetectionResult,
    OwnerScope,
    SourceGroup,
    V4Record,
)


def _timestamp() -> datetime:
    return datetime(2026, 7, 29, tzinfo=UTC)


def _record() -> V4Record:
    return V4Record(
        record_id=uuid4(),
        owner_user_id=7,
        source_group_id=uuid4(),
        content_hash=b"content-hash",
        auth_tag=b"auth-tag",
        created_at=_timestamp(),
        metadata={"source": "upload"},
    )


def test_v4_outcomes_are_closed_and_owner_scope_is_explicit() -> None:
    assert {item.value for item in DetectionOutcome} == {
        "success",
        "not_found",
        "ambiguous",
        "timeout",
        "resource_exhausted",
        "service_unavailable",
    }
    assert OwnerScope(user_id=7, cross_owner=False).query_owner_id == 7
    assert OwnerScope(user_id=7, cross_owner=True).query_owner_id is None


def test_v4_contracts_are_frozen_slots_and_defensively_normalize_mappings() -> None:
    source_metadata = {"source": "upload"}
    evidence_payload = {"score": 0.9}
    record = _record()
    source_group = SourceGroup(
        source_group_id=uuid4(),
        owner_user_id=7,
        source_hash=b"source-hash",
        created_at=_timestamp(),
        metadata=source_metadata,
    )
    evidence = DetectionEvidence(
        evidence_id=uuid4(),
        record_id=record.record_id,
        detector="v4",
        observed_at=_timestamp(),
        payload=evidence_payload,
    )
    result = DetectionResult(
        outcome=DetectionOutcome.SUCCESS,
        record=record,
        evidence={"primary": evidence},
    )

    for value in (OwnerScope(user_id=7), record, source_group, evidence, result):
        assert value.__dataclass_params__.frozen is True
        assert hasattr(value, "__slots__")

    source_metadata["source"] = "changed"
    evidence_payload["score"] = 0.1
    assert source_group.metadata == {"source": "upload"}
    assert evidence.payload == {"score": 0.9}
    assert isinstance(result.evidence, MappingProxyType)
    with pytest.raises(TypeError):
        result.evidence["another"] = evidence  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.owner_user_id = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (OwnerScope, {"user_id": 0}),
        (V4Record, {"owner_user_id": 0}),
        (SourceGroup, {"owner_user_id": -1}),
    ],
)
def test_owner_contracts_reject_non_positive_user_ids(factory, kwargs) -> None:
    common = {
        "record_id": uuid4(),
        "source_group_id": uuid4(),
        "content_hash": b"content-hash",
        "auth_tag": b"auth-tag",
        "created_at": _timestamp(),
        "source_hash": b"source-hash",
        "metadata": {},
    }

    with pytest.raises(ValueError, match="user_id"):
        factory(**{key: value for key, value in common.items() if key in factory.__dataclass_fields__}, **kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (
            V4Record,
            {
                "record_id": uuid4(),
                "owner_user_id": 7,
                "source_group_id": uuid4(),
                "content_hash": b"content-hash",
                "auth_tag": b"auth-tag",
                "created_at": datetime(2026, 7, 29),
            },
        ),
        (
            SourceGroup,
            {
                "source_group_id": uuid4(),
                "owner_user_id": 7,
                "source_hash": b"source-hash",
                "created_at": datetime(2026, 7, 29),
            },
        ),
        (
            DetectionEvidence,
            {
                "evidence_id": uuid4(),
                "record_id": uuid4(),
                "detector": "v4",
                "observed_at": datetime(2026, 7, 29),
            },
        ),
    ],
)
def test_timestamped_contracts_reject_naive_datetimes(factory, kwargs) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        factory(**kwargs)


def test_v4_records_use_domain_types_for_identifiers_and_binary_security_values() -> None:
    record = _record()

    assert isinstance(record.record_id, type(uuid4()))
    assert isinstance(record.source_group_id, type(uuid4()))
    assert isinstance(record.content_hash, bytes)
    assert isinstance(record.auth_tag, bytes)
    assert record.created_at.tzinfo is not None


def test_detection_result_exposes_a_record_only_for_success() -> None:
    record = _record()

    assert DetectionResult(DetectionOutcome.SUCCESS, record=record).record is record
    for outcome in DetectionOutcome:
        if outcome is DetectionOutcome.SUCCESS:
            continue
        assert DetectionResult(outcome).record is None
        with pytest.raises(ValueError, match="must not expose a record"):
            DetectionResult(outcome, record=record)
    with pytest.raises(ValueError, match="requires exactly one record"):
        DetectionResult(DetectionOutcome.SUCCESS)
