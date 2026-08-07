from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert

from trace_app.application import create_app
from trace_app.config import Settings
from trace_app.v4.media import V4MediaService
from trace_app.v4.repository import V4Repository
from trace_app.v4.schema import V4Tables


@pytest.fixture
def media_service(tmp_path: Path) -> V4MediaService:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    tables = V4Tables.build()
    tables.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(tables.users), [{"id": 7}, {"id": 8}])
    return V4MediaService(
        V4Repository(engine, tables=tables),
        storage_root=tmp_path / "objects",
        signing_key=b"k" * 32,
        public_base_url="https://media.example.test",
    )


def test_media_addresses_are_opaque_and_hide_storage_layout(
    media_service: V4MediaService,
) -> None:
    media = media_service.put_bytes(
        owner_user_id=7,
        variant="watermarked",
        content_type="image/png",
        content=b"private image bytes",
    )
    url = media_service.issue_url(media.id, requester_user_id=7, requester_is_admin=False)

    assert media.id not in media.storage_key
    assert url.startswith(f"https://media.example.test/api/media/{media.id}?")
    assert "/uploads/" not in url
    assert "watermarked" not in url
    assert str(media_service.storage_root) not in url
    assert "expires=" in url and "signature=" in url


def test_media_signature_binds_id_variant_owner_and_expiry(
    media_service: V4MediaService,
) -> None:
    media = media_service.put_bytes(
        owner_user_id=7,
        variant="thumbnail",
        content_type="image/webp",
        content=b"thumbnail",
    )
    expires, signature = media_service.sign(media, ttl_seconds=60, now=100)

    assert media_service.verify(media, expires=expires, signature=signature, now=160)
    assert not media_service.verify(media, expires=expires, signature=signature, now=161)
    tampered = media.__class__(
        id=media.id,
        owner_user_id=8,
        variant=media.variant,
        storage_key=media.storage_key,
        content_type=media.content_type,
        byte_size=media.byte_size,
        sha256=media.sha256,
        status=media.status,
    )
    assert not media_service.verify(tampered, expires=expires, signature=signature, now=100)


def test_non_owner_cannot_issue_media_url(media_service: V4MediaService) -> None:
    media = media_service.put_bytes(
        owner_user_id=7,
        variant="original",
        content_type="image/png",
        content=b"source",
    )
    with pytest.raises(HTTPException) as raised:
        media_service.issue_url(media.id, requester_user_id=8, requester_is_admin=False)
    assert raised.value.status_code == 404


def test_storage_resolver_rejects_traversal_and_symlink_escape(
    media_service: V4MediaService, tmp_path: Path
) -> None:
    with pytest.raises(HTTPException):
        media_service.resolve_storage_key("originals/../outside.bin")

    outside = tmp_path / "outside"
    outside.mkdir()
    link = media_service.storage_root / "originals"
    media_service.storage_root.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(HTTPException):
        media_service.resolve_storage_key("originals/aa/object.bin")


def test_mapped_transfer_route_serves_only_valid_signature(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'media.sqlite3'}",
        admin_user="admin",
        admin_pass="admin-secret",
        environment="test",
    )
    app = create_app(settings=settings, initialize_database=True)
    admin_id = app.state.runtime.store.get_user_by_username("admin")["id"]
    media = app.state.v4_media_service.put_bytes(
        owner_user_id=admin_id,
        variant="thumbnail",
        content_type="image/webp",
        content=b"mapped bytes",
    )
    with TestClient(app) as client:
        login = client.post(
            "/auth/login", data={"username": "admin", "password": "admin-secret"}
        )
        headers = {"Authorization": f"Bearer {login.json()['token']}"}
        issued = client.post(f"/api/media/{media.id}/access", headers=headers)
        assert issued.status_code == 200
        url = issued.json()["url"]
        transfer_path = url.removeprefix("https://media.example.test")
        assert client.get(transfer_path).content == b"mapped bytes"
        assert client.get(transfer_path.split("?", 1)[0]).status_code == 403
