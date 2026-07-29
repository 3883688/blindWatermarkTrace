from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trace_app.application import create_app
from trace_app.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite3'}",
        admin_user="admin",
        admin_pass="admin-secret",
        environment="test",
    )
    app = create_app(settings=settings, initialize_database=True)
    app.state.runtime.store.create_user("operator", "operator-secret", "operator")
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
        headers={"X-Forwarded-For": "203.0.113.10"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.mark.parametrize(
    ("method", "path"),
        [
            ("get", "/api/users"),
            ("get", "/api/roles"),
            ("get", "/api/v4/capabilities"),
            ("get", "/api/v4/records"),
            ("post", "/api/v4/detect-url"),
            ("post", "/api/v4/detect"),
            ("post", "/api/v4/generate"),
        ],
)
def test_product_operations_reject_anonymous_users(
    client: TestClient, method: str, path: str
) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == 401


def test_user_and_role_administration_requires_admin(client: TestClient) -> None:
    operator = _login(client, "operator", "operator-secret")
    admin = _login(client, "admin", "admin-secret")

    assert client.get("/api/users", headers=operator).status_code == 403
    assert client.get("/api/roles", headers=operator).status_code == 403
    assert client.get("/api/users", headers=admin).status_code == 200
    assert client.get("/api/roles", headers=admin).status_code == 200


def test_logout_revokes_database_session(client: TestClient) -> None:
    admin = _login(client, "admin", "admin-secret")
    assert client.get("/api/roles", headers=admin).status_code == 200
    assert client.post("/auth/logout", headers=admin).json() == {"logged_out": True}
    assert client.get("/api/roles", headers=admin).status_code == 401


def test_login_errors_do_not_disclose_account_existence(client: TestClient) -> None:
    unknown = client.post(
        "/auth/login", data={"username": "missing", "password": "wrong"}
    )
    wrong = client.post(
        "/auth/login", data={"username": "admin", "password": "wrong"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


def test_login_attempts_are_rate_limited(client: TestClient) -> None:
    responses = [
        client.post(
            "/auth/login", data={"username": "missing", "password": "wrong"}
        )
        for _ in range(6)
    ]
    assert [response.status_code for response in responses[:5]] == [401] * 5
    assert responses[5].status_code == 429


def test_production_does_not_register_development_reset(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="postgresql+psycopg://trace:test@db/trace",
        admin_user="admin",
        admin_pass="admin-secret",
        environment="production",
    )
    app = create_app(settings=settings, initialize_database=False)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/dev/reset" not in paths
