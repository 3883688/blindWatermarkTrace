"""Durable, owner-scoped deep-forensics job lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from sqlalchemy import Engine, and_, insert, or_, select, update

from trace_app.v4.domain import OwnerScope
from trace_app.v4.schema import V4Tables


@dataclass(frozen=True, slots=True)
class DeepJob:
    id: UUID
    owner_user_id: int
    requested_owner_user_id: int | None
    input_media_id: str
    status: str
    progress: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    deadline_at: datetime
    result: Mapping[str, Any] | None


class DeepJobStore:
    def __init__(self, engine: Engine, *, tables: V4Tables | None = None) -> None:
        self.engine = engine
        self.tables = tables or V4Tables.build()

    def create(
        self,
        actor_user_id: int,
        requested_scope: OwnerScope,
        input_media_id: str,
        *,
        now: datetime | None = None,
    ) -> DeepJob:
        created = now or datetime.now(UTC)
        values = {
            "id": uuid4(),
            "owner_user_id": actor_user_id,
            "requested_owner_user_id": requested_scope.query_owner_id,
            "input_media_id": input_media_id,
            "status": "queued",
            "progress": 0,
            "lease_owner": None,
            "lease_expires_at": None,
            "deadline_at": created + timedelta(seconds=1000),
            "result_outcome": None,
            "result_media_id": None,
            "result_evidence_id": None,
            "error_code": None,
            "error_detail": None,
            "created_at": created,
            "updated_at": created,
        }
        with self.engine.begin() as connection:
            connection.execute(insert(self.tables.deep_forensics_jobs).values(**values))
            connection.execute(
                insert(self.tables.audit_events).values(
                    id=uuid4(), actor_user_id=actor_user_id, action="v4.deep.create",
                    target_id=str(values["id"]), outcome="success",
                    correlation_id=values["id"], created_at=created,
                )
            )
        return self._job(values)

    def get(self, scope: OwnerScope, job_id: UUID) -> DeepJob | None:
        table = self.tables.deep_forensics_jobs
        conditions = [table.c.id == job_id]
        if scope.query_owner_id is not None:
            conditions.append(table.c.owner_user_id == scope.query_owner_id)
        with self.engine.connect() as connection:
            row = connection.execute(select(table).where(*conditions)).mappings().first()
        return None if row is None else self._job(row)

    def claim(self, worker_id: str, *, now: datetime, lease_seconds: int) -> DeepJob | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid lease")
        table = self.tables.deep_forensics_jobs
        claimable = and_(
            table.c.deadline_at > now,
            or_(table.c.status == "queued", and_(table.c.status == "running", table.c.lease_expires_at < now)),
        )
        statement = select(table.c.id).where(claimable).order_by(table.c.created_at, table.c.id).limit(1)
        if self.engine.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        with self.engine.begin() as connection:
            job_id = connection.execute(statement).scalar_one_or_none()
            if job_id is None:
                return None
            connection.execute(
                update(table).where(table.c.id == job_id, claimable).values(
                    status="running",
                    lease_owner=worker_id,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            row = connection.execute(select(table).where(table.c.id == job_id)).mappings().one()
        return self._job(row)

    def renew(
        self, job_id: UUID, worker_id: str, *, progress: int, now: datetime, lease_seconds: int
    ) -> bool:
        if not 0 <= progress <= 100:
            raise ValueError("progress must be between 0 and 100")
        table = self.tables.deep_forensics_jobs
        with self.engine.begin() as connection:
            result = connection.execute(
                update(table)
                .where(
                    table.c.id == job_id,
                    table.c.status == "running",
                    table.c.lease_owner == worker_id,
                    table.c.deadline_at > now,
                )
                .values(
                    progress=progress,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
        return bool(result.rowcount)

    def cancel(self, scope: OwnerScope, job_id: UUID, *, now: datetime) -> bool:
        table = self.tables.deep_forensics_jobs
        conditions = [table.c.id == job_id, table.c.status.in_(("queued", "running"))]
        if scope.query_owner_id is not None:
            conditions.append(table.c.owner_user_id == scope.query_owner_id)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(table).where(*conditions).values(
                    status="cancelled", lease_owner=None, lease_expires_at=None, updated_at=now
                )
            )
            if result.rowcount:
                connection.execute(
                    insert(self.tables.audit_events).values(
                        id=uuid4(), actor_user_id=scope.user_id, action="v4.deep.cancel",
                        target_id=str(job_id), outcome="success",
                        correlation_id=job_id, created_at=now,
                    )
                )
        return bool(result.rowcount)

    def finish(
        self, job_id: UUID, worker_id: str, *, outcome: str, result: Mapping[str, Any], now: datetime
    ) -> bool:
        table = self.tables.deep_forensics_jobs
        final_status = "completed" if outcome == "success" else "failed"
        result_media_id = result.get("result_media_id")
        evidence_id = result.get("evidence_id")
        try:
            normalized_evidence_id = None if evidence_id is None else UUID(str(evidence_id))
        except ValueError:
            normalized_evidence_id = None
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(table)
                .where(table.c.id == job_id, table.c.status == "running", table.c.lease_owner == worker_id)
                .values(
                    status=final_status,
                    progress=100,
                    lease_owner=None,
                    lease_expires_at=None,
                    result_outcome=outcome,
                    result_media_id=(
                        str(result_media_id)[:64] if result_media_id is not None else None
                    ),
                    result_evidence_id=normalized_evidence_id,
                    error_code=None if final_status == "completed" else outcome[:64],
                    error_detail=None,
                    updated_at=now,
                )
            )
        return bool(changed.rowcount)

    @staticmethod
    def _job(row: Mapping[str, Any]) -> DeepJob:
        lease_expires_at = row["lease_expires_at"]
        deadline_at = row["deadline_at"]
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
            lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
        if deadline_at.tzinfo is None:
            deadline_at = deadline_at.replace(tzinfo=UTC)
        result = None
        if row.get("result_outcome") is not None:
            result = {"outcome": row["result_outcome"]}
            if row.get("result_media_id") is not None:
                result["result_media_id"] = row["result_media_id"]
            if row.get("result_evidence_id") is not None:
                result["evidence_id"] = str(row["result_evidence_id"])
        return DeepJob(
            id=row["id"],
            owner_user_id=row["owner_user_id"],
            requested_owner_user_id=row["requested_owner_user_id"],
            input_media_id=row["input_media_id"],
            status=row["status"],
            progress=row["progress"],
            lease_owner=row["lease_owner"],
            lease_expires_at=lease_expires_at,
            deadline_at=deadline_at,
            result=result,
        )


class DeepJobWorker:
    def __init__(self, store: DeepJobStore, pool: Any, *, worker_id: str) -> None:
        self.store = store
        self.pool = pool
        self.worker_id = worker_id

    def run_once(self, operation: Callable[[str], Any], *, now: datetime | None = None) -> Any:
        current = now or datetime.now(UTC)
        job = self.store.claim(self.worker_id, now=current, lease_seconds=30)
        if job is None:
            return None
        timeout = max(0.0, min(1000.0, (job.deadline_at - current).total_seconds()))
        result = self.pool.run(operation, job.input_media_id, timeout_seconds=timeout)
        outcome = result.outcome.value
        value = result.value if isinstance(result.value, Mapping) else {}
        self.store.finish(
            job.id,
            self.worker_id,
            outcome=outcome,
            result={"outcome": outcome, **value},
            now=current,
        )
        return result


__all__ = ("DeepJob", "DeepJobStore", "DeepJobWorker")
