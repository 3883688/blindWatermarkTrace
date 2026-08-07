"""Allowlisted V4 telemetry and audit emission."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from uuid import UUID


STAGES = frozenset(
    {"decode", "exact_lookup", "dino_recall", "geometry", "observation", "authentication", "total"}
)


class StructuredTelemetry:
    def __init__(self, sink: Callable[[dict[str, Any]], None], *, correlation_id: UUID, owner_user_id: int) -> None:
        self.sink = sink
        self.correlation_id = correlation_id
        self.owner_user_id = owner_user_id

    def stage(
        self,
        stage: str,
        *,
        duration_ms: float,
        candidate_count: int = 0,
        fallback_count: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if stage not in STAGES or duration_ms < 0 or min(candidate_count, fallback_count) < 0:
            raise ValueError("invalid V4 telemetry stage")
        self.sink(
            {
                "event": "v4.stage",
                "correlation_id": str(self.correlation_id),
                "owner_user_id": self.owner_user_id,
                "stage": stage,
                "duration_ms": float(duration_ms),
                "candidate_count": int(candidate_count),
                "fallback_count": int(fallback_count),
            }
        )

    def finish(self, outcome: str, *, context: Mapping[str, Any] | None = None) -> None:
        self.sink(
            {
                "event": "v4.outcome",
                "correlation_id": str(self.correlation_id),
                "owner_user_id": self.owner_user_id,
                "outcome": str(outcome),
            }
        )


class AuditWriter:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def write(
        self,
        *,
        actor_user_id: int | None,
        action: str,
        target_id: str | None,
        outcome: str,
        correlation_id: UUID,
        created_at: datetime,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.repository.append_audit(
            actor_user_id=actor_user_id,
            action=action,
            target_id=target_id,
            outcome=outcome,
            correlation_id=correlation_id,
            created_at=created_at,
        )


__all__ = ("AuditWriter", "StructuredTelemetry")
