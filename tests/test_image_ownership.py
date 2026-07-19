from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trace_app.application import create_app
from trace_app.config import Settings


def _login_headers(
    client: TestClient, username: str, password: str
) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture
def ownership_context(tmp_path: Path):
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'ownership.sqlite3'}",
        admin_user="admin",
        admin_pass="admin-secret",
    )
    app = create_app(settings=settings, initialize_database=True)
    store = app.state.runtime.store
    store.create_user("alice", "alice-secret", "operator")
    alice_id = store.get_user_by_username("alice")["id"]
    admin_id = store.get_user_by_username("admin")["id"]
    store.insert_record(
        {"id": "admin-image", "status": "保护中"},
        owner_user_id=admin_id,
    )
    store.insert_record(
        {"id": "alice-image", "status": "保护中"},
        owner_user_id=alice_id,
    )
    with TestClient(app) as client:
        yield client, store, admin_id, alice_id


def test_non_admin_lists_only_own_images(ownership_context) -> None:
    client, _store, _admin_id, _alice_id = ownership_context

    assert client.get("/api/images").status_code == 401
    alice_response = client.get(
        "/api/images",
        headers=_login_headers(client, "alice", "alice-secret"),
    )
    admin_response = client.get(
        "/api/images",
        headers=_login_headers(client, "admin", "admin-secret"),
    )

    assert [item["id"] for item in alice_response.json()["items"]] == [
        "alice-image"
    ]
    assert alice_response.json()["stats"]["total"] == 1
    assert {item["id"] for item in admin_response.json()["items"]} == {
        "admin-image",
        "alice-image",
    }
    assert admin_response.json()["stats"]["total"] == 2


def test_non_admin_can_delete_only_own_image(ownership_context) -> None:
    client, store, admin_id, alice_id = ownership_context
    headers = _login_headers(client, "alice", "alice-secret")

    forbidden = client.delete("/api/images/admin-image", headers=headers)
    own = client.delete("/api/images/alice-image", headers=headers)

    assert forbidden.status_code == 404
    assert own.json() == {"deleted": True}
    assert store.read_records(owner_user_id=admin_id)[0]["id"] == "admin-image"
    assert store.read_records(owner_user_id=alice_id) == []
