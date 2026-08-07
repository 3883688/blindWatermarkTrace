from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from trace_app.v4.telemetry import AuditWriter, StructuredTelemetry


def test_stage_events_and_final_outcome_use_an_explicit_allowlist() -> None:
    events = []
    telemetry = StructuredTelemetry(events.append, correlation_id=UUID(int=1), owner_user_id=7)
    canaries = {
        "token": "bearer-secret",
        "key": "hmac-secret",
        "auth_tag": "deadbeef",
        "image": b"raw-image",
        "path": "D:/server/private.png",
        "host": "postgres.internal",
        "bucket": "private-bucket",
    }

    for stage in (
        "decode", "exact_lookup", "dino_recall", "geometry",
        "observation", "authentication", "total",
    ):
        telemetry.stage(stage, duration_ms=1.25, candidate_count=2, context=canaries)
    telemetry.finish("not_found", context=canaries)

    assert {event["stage"] for event in events[:-1]} == {
        "decode", "exact_lookup", "dino_recall", "geometry",
        "observation", "authentication", "total",
    }
    assert events[-1]["outcome"] == "not_found"
    serialized = json.dumps(events)
    for secret in ("bearer-secret", "hmac-secret", "deadbeef", "raw-image", "D:/server", "postgres.internal", "private-bucket"):
        assert secret not in serialized


def test_audit_writer_discards_details_and_uses_repository_allowlist() -> None:
    calls = []

    class Repository:
        def append_audit(self, **values):
            calls.append(values)

    AuditWriter(Repository()).write(
        actor_user_id=7,
        action="v4.deep.create",
        target_id="job-id",
        outcome="success",
        correlation_id=UUID(int=2),
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        details={"token": "secret-token", "storage_key": "D:/private"},
    )

    assert set(calls[0]) == {
        "actor_user_id", "action", "target_id", "outcome", "correlation_id", "created_at"
    }
    assert "secret-token" not in repr(calls)
