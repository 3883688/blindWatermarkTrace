"""Immutable, storage-agnostic contracts for the V4 detection domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _require_positive_user_id(user_id: int) -> None:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user_id must be a positive integer")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class DetectionOutcome(str, Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass(frozen=True, slots=True)
class OwnerScope:
    user_id: int
    cross_owner: bool = False

    def __post_init__(self) -> None:
        _require_positive_user_id(self.user_id)

    @property
    def query_owner_id(self) -> int | None:
        return None if self.cross_owner else self.user_id


@dataclass(frozen=True, slots=True)
class V4Record:
    record_id: UUID
    owner_user_id: int
    source_group_id: UUID
    content_hash: bytes
    auth_tag: bytes
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive_user_id(self.owner_user_id)
        _require_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SourceGroup:
    source_group_id: UUID
    owner_user_id: int
    source_hash: bytes
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive_user_id(self.owner_user_id)
        _require_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class DetectionEvidence:
    evidence_id: UUID
    record_id: UUID | None
    detector: str
    observed_at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_aware_datetime(self.observed_at, "observed_at")
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class DetectionResult:
    outcome: DetectionOutcome
    record: V4Record | None = None
    evidence: Mapping[str, DetectionEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome is DetectionOutcome.SUCCESS:
            if self.record is None:
                raise ValueError("success requires exactly one record")
        elif self.record is not None:
            raise ValueError(f"{self.outcome.value} must not expose a record")
        object.__setattr__(self, "evidence", _immutable_mapping(self.evidence))
