from io import BytesIO
from pathlib import Path
import socket

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from trace_app.application import create_app
from trace_app.config import Settings
from trace_app.imaging.io import load_image_from_url
from trace_app import media
from trace_app.imaging import io as image_io


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
def media_context(tmp_path: Path):
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'media.sqlite3'}",
        admin_user="admin",
        admin_pass="admin-secret",
    )
    app = create_app(settings=settings, initialize_database=True)
    store = app.state.runtime.store
    store.create_user("alice", "alice-secret", "operator")
    store.create_user("bob", "bob-secret", "operator")
    alice_id = store.get_user_by_username("alice")["id"]
    media_path = settings.original_dir / "alice-private.png"
    media_path.write_bytes(b"private-image-bytes")
    watermarked_path = settings.watermarked_dir / "alice-marked.png"
    watermarked_path.write_bytes(b"watermarked-image-bytes")
    thumbnail_path = settings.thumbnail_dir / "alice-thumb.png"
    thumbnail_path.write_bytes(b"thumbnail-image-bytes")
    store.insert_record(
        {
            "id": "alice-image",
            "name": "display-name.png",
            "trace_id": "TR-ALICE",
            "mode": "dct",
            "status": "保护中",
            "confidence": 98,
            "original_url": "/uploads/originals/alice-private.png",
            "download_url": "/uploads/watermarked/alice-marked.png",
            "thumbnail_url": "/uploads/thumbnails/alice-thumb.png",
            "robust_auth_code": "secret-auth-code",
            "feature_index_path": "private/features/alice.npz",
            "original_file_sha256": "private-original-hash",
        },
        owner_user_id=alice_id,
    )
    with TestClient(app) as client:
        yield client


def test_media_file_requires_authentication(media_context: TestClient) -> None:
    response = media_context.get("/uploads/originals/alice-private.png")

    assert response.status_code == 403


def test_media_file_is_available_only_to_owner_or_admin(
    media_context: TestClient,
) -> None:
    alice = _login_headers(media_context, "alice", "alice-secret")
    bob = _login_headers(media_context, "bob", "bob-secret")
    admin = _login_headers(media_context, "admin", "admin-secret")

    for headers in (alice, bob, admin):
        assert media_context.get(
            "/uploads/originals/alice-private.png", headers=headers
        ).status_code == 403


def test_local_upload_reference_cannot_escape_upload_directory(
    tmp_path: Path,
) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (tmp_path / "outside.png").write_bytes(b"not-relevant")

    with pytest.raises(HTTPException) as raised:
        load_image_from_url("/uploads/../outside.png", upload_dir)

    assert raised.value.status_code == 400
    assert raised.value.detail == "图片链接路径无效"


def test_image_list_exposes_only_public_management_fields(
    media_context: TestClient,
) -> None:
    alice = _login_headers(media_context, "alice", "alice-secret")

    response = media_context.get("/api/images", headers=alice)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"items", "stats"}
    record = payload["items"][0]
    assert set(record) == {
        "id",
        "name",
        "trace_id",
        "mode",
        "status",
        "confidence",
        "download_access_url",
        "thumbnail_access_url",
    }
    for field, content in (
        ("download_access_url", b"watermarked-image-bytes"),
        ("thumbnail_access_url", b"thumbnail-image-bytes"),
    ):
        access_url = record[field]
        assert access_url.startswith(f"/api/images/alice-image/{field.split('_')[0]}?")
        assert "expire_time=" in access_url
        assert "&signature=" in access_url
        assert media_context.get(access_url).content == content
        assert media_context.get(access_url.split("?", 1)[0]).status_code == 403
    serialized = response.text
    for private_value in (
        "alice-private.png",
        "secret-auth-code",
        "private/features/alice.npz",
        "private-original-hash",
    ):
        assert private_value not in serialized
def test_signed_media_url_cannot_be_reused_for_another_path(
    media_context: TestClient,
) -> None:
    alice = _login_headers(media_context, "alice", "alice-secret")
    access_url = media.sign_media_url(
        "/uploads/originals/alice-private.png",
        media_context.app.state.media_signing_key,
        ttl_seconds=60,
    )
    tampered = access_url.replace("alice-private.png", "other-private.png")

    assert media_context.get(tampered).status_code == 403


def test_media_signature_expires() -> None:
    sign = getattr(media, "sign_media_url", None)
    verify = getattr(media, "verify_media_signature", None)
    assert callable(sign)
    assert callable(verify)
    key = b"k" * 32
    signed = sign(
        "/uploads/originals/example.png",
        key,
        ttl_seconds=10,
        now=100,
    )
    query = dict(
        item.split("=", 1) for item in signed.split("?", 1)[1].split("&")
    )

    assert verify(
        "/uploads/originals/example.png",
        expires=query["expire_time"],
        signature=query["signature"],
        key=key,
        now=110,
    )
    assert not verify(
        "/uploads/originals/example.png",
        expires=query["expire_time"],
        signature=query["signature"],
        key=key,
        now=111,
    )


def test_watermark_responses_issue_signed_media_access_urls(
    tmp_path: Path,
) -> None:
    class WatermarkSpy:
        async def embed(self, **_kwargs):
            return {
                "id": "image-id",
                "user_id": "alice",
                "trace_id": "TR-TEST",
                "mode": "dct",
                "mode_label": "DCT",
                "created_at": "2026-07-20 16:00:00",
                "status": "保护中",
                "confidence": 98,
                "robust_watermark_version": 4,
                "original_url": "/uploads/originals/source.png",
                "download_url": "/uploads/watermarked/marked.png",
                "thumbnail_url": "/uploads/thumbnails/thumb.png",
                "robust_auth_code": "private-auth-code",
                "original_file_sha256": "private-hash",
                "feature_index_path": "private-index.npz",
            }

        async def extract_upload(self, _file):
            return {
                "matched_file_url": "/uploads/watermarked/marked.png",
            }

        def extract_url(self, _url):
            return {
                "matched_file_url": "/uploads/originals/source.png",
            }

    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )
    app = create_app(
        settings=settings,
        initialize_database=False,
        watermark_service_factory=WatermarkSpy,
    )

    with TestClient(app) as client:
        embedded = client.post(
            "/api/watermark/embed",
            files={"file": ("source.png", b"image", "image/png")},
            data={"user_id": "alice"},
        ).json()
        extracted = client.post(
            "/api/watermark/extract",
            files={"file": ("marked.png", b"image", "image/png")},
        ).json()
        extracted_url = client.post(
            "/api/watermark/extract-url",
            data={"url": "https://example.test/marked.png"},
        ).json()

    assert set(embedded) == {
        "id",
        "user_id",
        "trace_id",
        "mode",
        "mode_label",
        "created_at",
        "status",
        "confidence",
        "robust_watermark_version",
        "original_access_url",
        "download_access_url",
    }
    assert embedded["original_access_url"].startswith(
        "/uploads/originals/source.png?expire_time="
    )
    assert embedded["download_access_url"].startswith(
        "/uploads/watermarked/marked.png?expire_time="
    )
    assert "source.png" not in str({k: v for k, v in embedded.items() if k != "original_access_url"})
    assert "private-auth-code" not in str(embedded)
    assert "private-hash" not in str(embedded)
    assert "private-index.npz" not in str(embedded)
    assert extracted["matched_file_access_url"].startswith(
        "/uploads/watermarked/marked.png?expire_time="
    )
    assert extracted_url["matched_file_access_url"].startswith(
        "/uploads/originals/source.png?expire_time="
    )


def _resolved_ip(address: str):
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 443),
        )
    ]


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str],
        content: bytes = b"",
    ) -> None:
        self.status = status
        self.headers = headers
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self._content[:limit]


def test_remote_image_url_rejects_private_address_before_connecting(
    tmp_path: Path,
) -> None:
    fetch = getattr(image_io, "fetch_remote_image_bytes", None)
    assert callable(fetch)
    calls: list[str] = []

    with pytest.raises(HTTPException) as raised:
        fetch(
            "http://127.0.0.1/private.png",
            resolve_host=lambda *_args, **_kwargs: _resolved_ip("127.0.0.1"),
            open_url=lambda request, **_kwargs: calls.append(request.full_url),
        )

    assert raised.value.status_code == 400
    assert raised.value.detail == "不允许访问内网或本机地址"
    assert calls == []


def test_remote_image_redirect_revalidates_destination() -> None:
    fetch = getattr(image_io, "fetch_remote_image_bytes", None)
    assert callable(fetch)
    calls: list[str] = []

    def resolve(host: str, *_args, **_kwargs):
        return _resolved_ip(
            "93.184.216.34" if host == "public.example" else "10.0.0.7"
        )

    def open_url(request, **_kwargs):
        calls.append(request.full_url)
        return _FakeResponse(
            status=302,
            headers={"location": "http://internal.example/private.png"},
        )

    with pytest.raises(HTTPException) as raised:
        fetch(
            "https://public.example/start",
            resolve_host=resolve,
            open_url=open_url,
        )

    assert raised.value.detail == "不允许访问内网或本机地址"
    assert calls == ["https://public.example/start"]


def test_remote_public_image_remains_supported(tmp_path: Path) -> None:
    fetch = getattr(image_io, "fetch_remote_image_bytes", None)
    assert callable(fetch)
    buffer = BytesIO()
    Image.new("RGB", (3, 2), "blue").save(buffer, format="PNG")

    data, content_type = fetch(
        "https://public.example/image.png",
        resolve_host=lambda *_args, **_kwargs: _resolved_ip("93.184.216.34"),
        open_url=lambda *_args, **_kwargs: _FakeResponse(
            status=200,
            headers={"content-type": "image/png"},
            content=buffer.getvalue(),
        ),
    )

    assert content_type == "image/png"
    assert load_image_from_url(
        "/uploads/originals/local.png",
        _write_local_image(tmp_path),
    ).size == (3, 2)
    assert Image.open(BytesIO(data)).size == (3, 2)


def _write_local_image(tmp_path: Path) -> Path:
    upload_dir = tmp_path / "uploads"
    originals = upload_dir / "originals"
    originals.mkdir(parents=True)
    Image.new("RGB", (3, 2), "blue").save(originals / "local.png")
    return upload_dir
