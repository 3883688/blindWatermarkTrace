from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select

import main
from database_store import DatabaseStore
from tests.test_watermark_v4_features import _feature_image
from trace_app.database.repositories import Repository
from watermark_v4 import V4Config
from watermark_v4.features import load_feature_index


AUTH_KEY = "api-v4-test-key-material-at-least-32-bytes"


def _png_bytes() -> bytes:
    buffer = BytesIO()
    _feature_image((512, 384), seed=808).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolated_v4_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upload_dir = tmp_path / "uploads"
    data_dir = tmp_path / "data"
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "ORIGINAL_DIR", upload_dir / "originals")
    monkeypatch.setattr(main, "WATERMARKED_DIR", upload_dir / "watermarked")
    monkeypatch.setattr(main, "THUMBNAIL_DIR", upload_dir / "thumbnails")
    monkeypatch.setattr(main, "DEFAULT_WATERMARK_AUTH_KEY", AUTH_KEY)
    store = DatabaseStore(
        create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    )
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("test-admin", "admin-password", "admin")
    monkeypatch.setattr(main.runtime, "store", store)
    monkeypatch.setattr(
        main, "repository", Repository(store, ensure_dirs=main.ensure_dirs)
    )
    monkeypatch.setattr(main, "db_store", store, raising=False)
    main.app.state.generated_trace_ids = []
    main.ensure_dirs()


def _embed_v4(client: TestClient):
    return client.post(
        "/api/watermark/embed",
        files={"file": ("v4-source.png", _png_bytes(), "image/png")},
        data={
            "user_id": "pytest-v4",
            "robust_watermark_version": "4",
            "copyright_enabled": "false",
            "small_crop_trace_enabled": "true",
            "dot_matrix_trace_enabled": "true",
        },
    )


def test_version_normalization_accepts_explicit_v4() -> None:
    assert main.normalize_robust_watermark_version("4") == 4
    assert main.normalize_robust_watermark_version(4) == 4


def test_service_limits_opencv_internal_threads() -> None:
    assert main.cv2.getNumThreads() == 1


def test_v4_generation_rejects_missing_auth_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "DEFAULT_WATERMARK_AUTH_KEY", "short")

    response = _embed_v4(TestClient(main.app))

    assert response.status_code == 503
    assert "32" in response.json()["detail"]


def test_v4_generation_persists_strict_codec_tag_and_feature_index() -> None:
    response = _embed_v4(TestClient(main.app))

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["robust_watermark_version"] == 4
    assert record["robust_watermark_codec"] == V4Config().codec
    assert len(record["robust_auth_code"]) == 8
    assert record["robust_auth_code"] == record["robust_auth_code"].lower()
    assert record["small_crop_trace_enabled"] is False
    assert record["dot_matrix_trace_enabled"] is False
    assert record["watermark_layers"] == {
        "dct_authenticated": True,
        "fft_sync": True,
    }
    index_path = main.DATA_DIR / record["feature_index_path"]
    feature_index = load_feature_index(index_path)
    assert feature_index is not None
    assert (feature_index.image_width, feature_index.image_height) == (512, 384)
    assert len(feature_index.descriptors) > 0


def test_v4_generation_does_not_call_legacy_watermark_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy watermark layer must not run for v4")

    for name in (
        "embed_robust_watermark",
        "embed_robust_watermark_v2",
        "embed_robust_watermark_v3",
        "apply_frequency_layers",
        "apply_code_layer",
        "apply_small_crop_trace_layer",
        "apply_dot_matrix_trace_layer",
        "embed_lsb",
    ):
        monkeypatch.setattr(main, name, forbidden)

    response = _embed_v4(TestClient(main.app))

    assert response.status_code == 200, response.text


def _extract_bytes(client: TestClient, content: bytes, name: str = "query.png"):
    return client.post(
        "/api/watermark/extract",
        files={"file": (name, content, "image/png")},
    )


def test_v4_exact_watermarked_fingerprint_succeeds_and_original_rejects() -> None:
    client = TestClient(main.app)
    record = _embed_v4(client).json()
    watermarked_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )

    watermarked_response = _extract_bytes(client, watermarked_path.read_bytes())
    original_response = _extract_bytes(client, _png_bytes(), "original.png")

    assert watermarked_response.status_code == 200
    assert watermarked_response.json()["trace_id"] == record["trace_id"]
    assert watermarked_response.json()["matched_file_type"] == "watermarked"
    assert original_response.status_code == 404


def test_v4_transformed_crop_is_uniquely_attributed() -> None:
    client = TestClient(main.app)
    record = _embed_v4(client).json()
    watermarked_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    with Image.open(watermarked_path) as loaded:
        query = loaded.convert("RGB").crop((64, 64, 448, 320))
    buffer = BytesIO()
    query.save(buffer, format="PNG")

    response = _extract_bytes(client, buffer.getvalue(), "cropped.png")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["trace_id"] == record["trace_id"]
    assert result["code_recovery"]["codec"] == V4Config().codec
    assert result["code_recovery"]["candidate_count"] <= 3
    assert result["code_recovery"]["authenticated_tiles"] >= 2


def test_v4_negative_never_falls_through_to_legacy_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(main.app)
    _embed_v4(client)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("v4 negative must not use a legacy attribution fallback")

    for name in (
        "detect_dot_matrix_trace",
        "detect_aligned_authenticated_watermark",
        "detect_by_visual_match",
        "detect_small_crop_trace",
        "detect_watermark_code",
        "detect_robust_watermark",
        "detect_by_residual_match",
    ):
        monkeypatch.setattr(main, name, forbidden)
    unrelated = BytesIO()
    _feature_image((512, 384), seed=9999).save(unrelated, format="PNG")

    response = _extract_bytes(client, unrelated.getvalue(), "negative.png")

    assert response.status_code == 404


def test_user_management_persists_hashed_users_in_database() -> None:
    client = TestClient(main.app)

    created = client.post(
        "/api/users",
        json={"username": "alice", "password": "user-secret", "role": "operator"},
    )

    assert created.status_code == 200, created.text
    assert main.db_store.authenticate("alice", "user-secret") == "operator"
    with main.db_store.engine.connect() as connection:
        password_hash = connection.execute(
            select(main.db_store.users.c.password_hash).where(
                main.db_store.users.c.username == "alice"
            )
        ).scalar_one()
    assert password_hash.startswith("scrypt$v1$")
    assert "user-secret" not in password_hash
    assert not (main.DATA_DIR / "users.json").exists()

    login = client.post(
        "/auth/login",
        data={"username": "alice", "password": "user-secret"},
    )
    assert login.status_code == 200, login.text
    assert login.json()["role"] == "operator"

    updated = client.put("/api/users/alice", json={"role": "viewer"})
    assert updated.status_code == 200, updated.text
    assert main.db_store.list_users()["alice"]["role"] == "viewer"

    deleted = client.delete("/api/users/alice")
    assert deleted.status_code == 200, deleted.text
    assert main.db_store.authenticate("alice", "user-secret") is None


def test_auth_api_preserves_invalid_login_and_user_crud_errors() -> None:
    client = TestClient(main.app)

    invalid_login = client.post(
        "/auth/login",
        data={"username": "test-admin", "password": "wrong-password"},
    )
    assert invalid_login.status_code == 401
    assert invalid_login.json() == {"detail": "用户名或密码错误"}

    missing_username = client.post("/api/users", json={"password": "secret"})
    assert missing_username.status_code == 400
    assert missing_username.json() == {"detail": "请输入用户名"}

    payload = {"username": "alice", "password": "secret", "role": "operator"}
    assert client.post("/api/users", json=payload).status_code == 200
    duplicate = client.post("/api/users", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "用户已存在"}

    invalid_role = client.put("/api/users/alice", json={"role": "missing"})
    assert invalid_role.status_code == 400
    assert invalid_role.json() == {"detail": "角色不存在"}

    missing_user = client.put("/api/users/missing", json={"role": "viewer"})
    assert missing_user.status_code == 404
    assert missing_user.json() == {"detail": "用户不存在"}

    assert client.delete("/api/users/alice").status_code == 200
    missing_delete = client.delete("/api/users/alice")
    assert missing_delete.status_code == 404
    assert missing_delete.json() == {"detail": "用户不存在"}
