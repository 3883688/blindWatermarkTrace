from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import create_engine, insert
from fastapi.testclient import TestClient

from trace_app.application import create_app
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import Settings
from trace_app.dependencies import get_current_user
from trace_app.v4.domain import DetectionOutcome, OwnerScope
from trace_app.v4.jobs import DeepJobStore, DeepJobWorker
from trace_app.v4.schema import V4Tables
from trace_app.v4.workers import WorkerResult


@pytest.fixture
def store() -> DeepJobStore:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    tables = V4Tables.build()
    tables.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(tables.users), [{"id": 1}, {"id": 7}, {"id": 8}])
        connection.execute(
            insert(tables.media_objects),
            {
                "id": "media_opaque",
                "owner_user_id": 7,
                "variant": "original",
                "storage_key": "objects/private.bin",
                "content_type": "image/png",
                "byte_size": 1,
                "sha256": b"x" * 32,
                "status": "active",
            },
        )
    return DeepJobStore(engine, tables=tables)


def test_create_and_read_are_owner_scoped_with_fixed_absolute_deadline(store: DeepJobStore) -> None:
    now = datetime(2026, 7, 29, tzinfo=UTC)
    job = store.create(
        actor_user_id=7,
        requested_scope=OwnerScope(7),
        input_media_id="media_opaque",
        now=now,
    )

    assert job.deadline_at == now + timedelta(seconds=1000)
    assert job.input_media_id == "media_opaque"
    assert store.get(OwnerScope(7), job.id) == job
    assert store.get(OwnerScope(8), job.id) is None
    assert store.get(OwnerScope(1, cross_owner=True), job.id) == job


def test_expired_lease_is_recovered_and_progress_is_bounded(store: DeepJobStore) -> None:
    now = datetime(2026, 7, 29, tzinfo=UTC)
    job = store.create(7, OwnerScope(7), "media_opaque", now=now)
    first = store.claim("worker-a", now=now, lease_seconds=30)

    assert first.id == job.id and first.status == "running"
    assert store.claim("worker-b", now=now + timedelta(seconds=29), lease_seconds=30) is None
    recovered = store.claim("worker-b", now=now + timedelta(seconds=31), lease_seconds=30)
    assert recovered.id == job.id and recovered.lease_owner == "worker-b"
    assert store.renew(job.id, "worker-b", progress=75, now=now + timedelta(seconds=32), lease_seconds=30)
    with pytest.raises(ValueError):
        store.renew(job.id, "worker-b", progress=101, now=now, lease_seconds=30)


def test_cancellation_prevents_further_work(store: DeepJobStore) -> None:
    now = datetime(2026, 7, 29, tzinfo=UTC)
    job = store.create(7, OwnerScope(7), "media_opaque", now=now)
    store.claim("worker", now=now, lease_seconds=30)

    assert store.cancel(OwnerScope(7), job.id, now=now)
    assert not store.renew(job.id, "worker", progress=50, now=now, lease_seconds=30)
    assert store.get(OwnerScope(7), job.id).status == "cancelled"


def test_create_and_cancel_write_redacted_audits(store: DeepJobStore) -> None:
    now = datetime(2026, 7, 29, tzinfo=UTC)
    job = store.create(7, OwnerScope(7), "media_opaque", now=now)
    store.cancel(OwnerScope(7), job.id, now=now)
    with store.engine.connect() as connection:
        rows = connection.execute(store.tables.audit_events.select()).mappings().all()
    assert [(row["action"], row["outcome"]) for row in rows] == [
        ("v4.deep.create", "success"), ("v4.deep.cancel", "success")
    ]
    assert "storage_key" not in repr(rows)


def test_worker_never_runs_past_absolute_1000_second_deadline(store: DeepJobStore) -> None:
    now = datetime(2026, 7, 29, tzinfo=UTC)
    job = store.create(7, OwnerScope(7), "media_opaque", now=now)

    class Pool:
        timeout = None

        def run(self, function, *args, timeout_seconds):
            self.timeout = timeout_seconds
            return WorkerResult(DetectionOutcome.TIMEOUT)

    pool = Pool()
    worker = DeepJobWorker(store, pool, worker_id="worker")
    result = worker.run_once(lambda media_id: media_id, now=now)

    assert result.outcome is DetectionOutcome.TIMEOUT
    assert pool.timeout == 1000
    saved = store.get(OwnerScope(7), job.id)
    assert saved.status == "failed"
    assert saved.result == {"outcome": "timeout"}


def test_deep_job_api_is_authenticated_and_owner_scoped(tmp_path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path, upload_dir="uploads", data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'jobs.sqlite3'}",
        admin_user="admin", admin_pass="secret", environment="test",
    )

    class Jobs:
        scope = None

        def create(self, actor_user_id, requested_scope, input_media_id):
            self.scope = requested_scope
            return type("Job", (), {
                "id": UUID(int=9), "status": "queued", "progress": 0,
                "deadline_at": datetime(2026, 7, 29, tzinfo=UTC), "result": None,
            })()

        def get(self, scope, job_id):
            self.scope = scope
            return self.create(scope.user_id, scope, "media_opaque")

        def cancel(self, scope, job_id, **kwargs):
            self.scope = scope
            return True

    class Media:
        def get_media_or_404(self, media_id):
            return type("Media", (), {"id": media_id, "owner_user_id": 7})()

    jobs = Jobs()
    app = create_app(
        settings=settings, initialize_database=False,
        v4_job_service_factory=lambda: jobs,
        v4_media_service_factory=lambda: Media(),
    )
    client = TestClient(app)
    path = "/api/v4/jobs/00000000-0000-0000-0000-000000000009"
    assert client.post("/api/v4/jobs", data={"media_id": "media_opaque"}).status_code == 401
    assert client.get(path).status_code == 401
    assert client.delete(path).status_code == 401

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(7, "user", "operator")
    assert client.post("/api/v4/jobs", data={"media_id": "media_opaque"}).json()["status"] == "queued"
    assert client.get(path).status_code == 200
    assert client.delete(path).json() == {"cancelled": True}
    assert jobs.scope.query_owner_id == 7
