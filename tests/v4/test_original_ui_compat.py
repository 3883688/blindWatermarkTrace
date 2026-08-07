from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from trace_app.application import create_app
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import Settings
from trace_app.dependencies import get_current_user, get_repository
from trace_app.v4.domain import DetectionOutcome, DetectionResult


def _record(owner_id: int = 7):
    return SimpleNamespace(
        id=UUID("10000000-0000-0000-0000-000000000001"),
        source_group_id=UUID("20000000-0000-0000-0000-000000000002"),
        owner_user_id=owner_id,
        trace_id="trace-v4-only",
        evidence_uuid=UUID("12345678-0000-0000-0000-000012345678"),
        output_media_id="opaque-output",
        thumbnail_media_id="opaque-thumb",
        original_filename="source.png",
        created_at=None,
        status="active",
    )


class _Generation:
    def generate(self, request, deadline):
        return SimpleNamespace(record=_record(request.owner_user_id), source_group_created=True)


class _Detection:
    def detect(self, request, deadline):
        return DetectionResult(DetectionOutcome.SUCCESS, _record(request.scope.user_id))


class _Repository:
    def list_records(self, scope):
        return (_record(scope.user_id or 7),)

    def get_source_group(self, scope, source_group_id):
        return SimpleNamespace(original_media_id="opaque-original")

    def get_media(self, media_id):
        sizes = {
            "opaque-original": 1024,
            "opaque-output": 2621440,
            "opaque-thumb": 512,
        }
        return SimpleNamespace(byte_size=sizes[media_id])

    def delete_record(self, scope, record_id):
        return True

    def dashboard_stats(self, scope):
        return {"total": 1, "today": 1, "detected": 1, "success_rate": 100.0}

    def increment_counter(self, owner_user_id, key, delta=1):
        return 1


class _Users:
    def get_user_by_id(self, user_id):
        return {"id": user_id, "username": "operator", "role": "operator"}


class _Media:
    def issue_url(self, media_id, *, requester_user_id, requester_is_admin):
        return f"/api/media/{media_id}?expires=999&signature=signed"


def _client(tmp_path) -> tuple[TestClient, object]:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'compat.sqlite3'}",
        admin_user="admin",
        admin_pass="secret",
        environment="test",
    )
    repository = _Repository()
    app = create_app(
        settings=settings,
        initialize_database=False,
        v4_generation_service_factory=_Generation,
        v4_detection_service_factory=_Detection,
        v4_record_repository_factory=lambda: repository,
        v4_media_service_factory=_Media,
    )
    app.dependency_overrides[get_repository] = _Users
    return TestClient(app), app


def test_original_routes_require_login(tmp_path) -> None:
    client, _app = _client(tmp_path)
    assert client.post("/api/watermark/embed").status_code == 401
    assert client.post("/api/watermark/extract").status_code == 401
    assert client.get("/api/images").status_code == 401
    assert client.get("/api/dashboard-stats").status_code == 401


def test_original_generation_and_images_use_only_opaque_v4_media_urls(tmp_path) -> None:
    client, app = _client(tmp_path)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id=7, username="operator", role="operator"
    )

    generated = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", b"image", "image/png")},
        data={"user_id": "display-user", "mode": "dct"},
    )
    listed = client.get("/api/images")

    assert generated.status_code == 200, generated.text
    assert generated.json()["download_access_url"].startswith("/api/media/opaque-output?")
    assert generated.json()["original_access_url"].startswith("/api/media/opaque-original?")
    assert generated.json()["size"] == "2.5 MB"
    assert listed.status_code == 200
    assert listed.json()["items"][0]["size"] == "2.5 MB"
    assert listed.json()["items"][0]["robust_watermark_version"] == 4
    assert listed.json()["items"][0]["evidence_uuid_head"] == "12345678"
    assert listed.json()["items"][0]["evidence_uuid_tail"] == "12345678"
    assert listed.json()["items"][0]["thumbnail_access_url"].startswith(
        "/api/media/opaque-thumb?"
    )
    assert "/uploads/" not in str(generated.json()) + str(listed.json())


def test_original_detection_and_dashboard_are_backed_by_v4(tmp_path) -> None:
    client, app = _client(tmp_path)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id=7, username="operator", role="operator"
    )

    detected = client.post(
        "/api/watermark/extract",
        files={"file": ("query.png", b"query", "image/png")},
    )
    dashboard = client.get("/api/dashboard-stats")

    assert detected.status_code == 200
    assert detected.json()["user_id"] == "operator"
    assert detected.json()["trace_id"] == "trace-v4-only"
    assert detected.json()["matched_file_access_url"].startswith(
        "/api/media/opaque-original?"
    )
    assert dashboard.json() == {
        "total": 1,
        "today": 1,
        "detected": 1,
        "success_rate": 100.0,
    }


def test_production_entrypoint_and_frontend_keep_original_contract() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    entrypoint = (root / "main.py").read_text(encoding="utf-8")
    frontend = (root / "assets/app/app.js").read_text(encoding="utf-8")

    assert "trace_app import compat" not in entrypoint
    assert "create_app" in entrypoint
    assert "/api/watermark/embed" in frontend
    assert "/api/images" in frontend
    assert "/api/v4/generate" not in frontend
