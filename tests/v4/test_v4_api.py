from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from trace_app.application import create_app
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import Settings
from trace_app.dependencies import get_current_user
from trace_app.v4.domain import DetectionOutcome, DetectionResult
from watermark_v4.payload import CODEC_ID


@dataclass
class FakeGeneration:
    request: object | None = None

    def generate(self, request, deadline):
        self.request = request
        return SimpleNamespace(
            record=_record(request.owner_user_id), source_group_created=True
        )


@dataclass
class FakeDetection:
    request: object | None = None

    def detect(self, request, deadline):
        self.request = request
        return DetectionResult(DetectionOutcome.SUCCESS, _record(request.scope.user_id))


class FakeRecords:
    def __init__(self):
        self.scope = None

    def list_records(self, scope):
        self.scope = scope
        return (_record(scope.user_id),)

    def delete_record(self, scope, record_id):
        self.scope = scope
        self.deleted_id = record_id
        return True


class FakeMedia:
    def issue_url(self, media_id, *, requester_user_id, requester_is_admin):
        return f"/api/media/{media_id}?signed=opaque"


def _record(owner_id: int):
    return SimpleNamespace(
        id=UUID("10000000-0000-0000-0000-000000000001"),
        source_group_id=UUID("20000000-0000-0000-0000-000000000002"),
        owner_user_id=owner_id,
        trace_id="v4-trace",
        codec=CODEC_ID,
        auth_tag=b"secret!!",
        key_id="private-key",
        output_media_id="out_opaque",
        thumbnail_media_id="thumb_opaque",
        storage_key="D:/private/object.bin",
        status="active",
    )


@pytest.fixture
def context(tmp_path: Path):
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}",
        admin_user="admin",
        admin_pass="secret",
        environment="test",
    )
    generation = FakeGeneration()
    detection = FakeDetection()
    records = FakeRecords()
    app = create_app(
        settings=settings,
        initialize_database=False,
        v4_generation_service_factory=lambda: generation,
        v4_detection_service_factory=lambda: detection,
        v4_record_repository_factory=lambda: records,
        v4_capabilities_factory=lambda: {"dinov2": True, "lightglue": False},
        v4_media_service_factory=lambda: FakeMedia(),
        v4_remote_fetch_factory=lambda url, deadline: (b"remote-image", "image/png"),
    )
    yield app, generation, detection, records


def _login_as(app, *, role="operator", user_id=7):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id=user_id, username="tester", role=role
    )


def test_all_v4_routes_require_login(context) -> None:
    app, *_ = context
    client = TestClient(app)
    requests = (
        ("post", "/api/v4/generate", {"files": {"file": ("x.png", b"x", "image/png")}, "data": {"codec": CODEC_ID}}),
        ("post", "/api/v4/detect", {"files": {"file": ("x.png", b"x", "image/png")}}),
        ("post", "/api/v4/detect-url", {"data": {"url": "https://example.test/x.png"}}),
        ("get", "/api/v4/records", {}),
        ("get", "/api/v4/capabilities", {}),
    )
    for method, path, kwargs in requests:
        assert getattr(client, method)(path, **kwargs).status_code == 401


def test_generate_accepts_only_the_exact_v4_codec_and_hides_private_fields(context) -> None:
    app, generation, *_ = context
    _login_as(app)
    client = TestClient(app)
    files = {"file": ("source.png", b"image", "image/png")}

    assert client.post("/api/v4/generate", files=files, data={"codec": "v3"}).status_code == 422
    response = client.post("/api/v4/generate", files=files, data={"codec": CODEC_ID})

    assert response.status_code == 200
    assert generation.request.owner_user_id == 7
    payload = response.json()
    assert payload["codec"] == CODEC_ID
    assert payload["output_access_url"].startswith("/api/media/out_opaque?")
    serialized = str(payload)
    for secret in ("auth_tag", "key_id", "storage_key", "D:/private", "/uploads/"):
        assert secret not in serialized


def test_detection_returns_typed_outcome_and_applies_owner_scope(context) -> None:
    app, _, detection, _ = context
    _login_as(app)
    client = TestClient(app)

    response = client.post(
        "/api/v4/detect", files={"file": ("query.png", b"query", "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["outcome"] == "success"
    assert detection.request.scope.query_owner_id == 7

    app.state.v4_remote_fetch_factory = lambda *_args: pytest.fail(
        "unauthorized cross-owner request must be rejected before remote fetch"
    )
    forbidden = client.post(
        "/api/v4/detect-url",
        data={"url": "https://example.test/x.png", "cross_owner": "true"},
    )
    assert forbidden.status_code == 403


def test_admin_cross_owner_requires_explicit_flag(context) -> None:
    app, _, detection, _ = context
    _login_as(app, role="admin", user_id=1)
    client = TestClient(app)

    client.post("/api/v4/detect-url", data={"url": "https://example.test/x.png"})
    assert detection.request.scope.query_owner_id == 1
    client.post(
        "/api/v4/detect-url",
        data={"url": "https://example.test/x.png", "cross_owner": "true"},
    )
    assert detection.request.scope.query_owner_id is None


def test_records_and_capabilities_are_scoped_and_allowlisted(context) -> None:
    app, _, _, records = context
    _login_as(app)
    client = TestClient(app)

    listed = client.get("/api/v4/records")
    assert listed.status_code == 200
    assert records.scope.query_owner_id == 7
    assert listed.json()["items"][0]["output_media_id"] == "out_opaque"
    assert "private-key" not in str(listed.json())
    assert client.get("/api/v4/capabilities").json() == {
        "codec": CODEC_ID,
        "dinov2": True,
        "lightglue": False,
    }


def test_record_deletion_requires_login_and_owner_scope(context) -> None:
    app, _, _, records = context
    path = "/api/v4/records/10000000-0000-0000-0000-000000000001"
    assert TestClient(app).delete(path).status_code == 401

    _login_as(app)
    response = TestClient(app).delete(path)
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert records.scope.query_owner_id == 7


def test_runtime_registers_original_v4_contract_without_legacy_upload_routes(context) -> None:
    app, *_ = context
    paths = set(app.openapi()["paths"])
    assert "/api/v4/generate" in paths
    for path in (
        "/api/watermark/embed",
        "/api/watermark/extract",
        "/api/watermark/extract-url",
    ):
        assert path in paths
    for path in (
        "/uploads/{media_path:path}",
        "/api/dev/reset",
    ):
        assert path not in paths
