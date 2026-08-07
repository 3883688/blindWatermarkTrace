import os
import shutil
import sys
from io import BytesIO
from pathlib import Path

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/pytest_false_positive_gate/uploads"
os.environ["DATA_DIR"] = "test_output/pytest_false_positive_gate/data"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine

import main
from trace_app.database.store import DatabaseStore
from trace_app.database.repositories import Repository


@pytest.fixture(autouse=True)
def isolate_runtime_paths(monkeypatch, tmp_path):
    runtime_root = ROOT / "test_output" / "pytest_false_positive_gate"
    upload_dir = runtime_root / "uploads"
    data_dir = runtime_root / "data"
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "ORIGINAL_DIR", upload_dir / "originals")
    monkeypatch.setattr(main, "WATERMARKED_DIR", upload_dir / "watermarked")
    monkeypatch.setattr(main, "THUMBNAIL_DIR", upload_dir / "thumbnails")
    store = DatabaseStore(
        create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    )
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    monkeypatch.setattr(main.runtime, "store", store)
    monkeypatch.setattr(
        main, "repository", Repository(store, ensure_dirs=main.ensure_dirs)
    )
    monkeypatch.setattr(main, "db_store", store)


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def crop_fraction(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * ratio)))
    crop_height = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_width) // 3)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def attacked_unwatermarked(source: Path, scale: float, crop_ratio: float) -> Image.Image:
    image = Image.open(source).convert("RGB")
    scaled_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    scaled = image.resize(scaled_size, Image.Resampling.BICUBIC)
    return crop_fraction(scaled, crop_ratio)


def attacked_watermarked(record: dict, scale: float, crop_ratio: float) -> Image.Image:
    path = main.UPLOAD_DIR / record["download_url"].replace("/uploads/", "")
    image = Image.open(path).convert("RGB")
    scaled_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    scaled = image.resize(scaled_size, Image.Resampling.BICUBIC)
    return crop_fraction(scaled, crop_ratio)


def reset_test_state() -> None:
    root = ROOT / "test_output" / "pytest_false_positive_gate"
    if root.exists():
        shutil.rmtree(root)
    if main.database_ready():
        main.db_clear_all()
        main.require_store().replace_roles(main.DEFAULT_ROLES)
    main.app.state.generated_trace_ids = []
    main.ensure_dirs()


def seed_watermark_records(client: TestClient) -> dict[str, dict]:
    records = {}
    for source in sorted((ROOT / "img").glob("*.png")):
        with source.open("rb") as fp:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, fp, "image/png")},
                data={
                    "user_id": "pytest-false-positive-gate",
                    "mode": "dct",
                    "fidelity_level": "0.75",
                    "small_crop_trace_enabled": "true",
                    "small_crop_trace_strength": "1.0",
                    "small_crop_trace_density": "high",
                    "dot_matrix_trace_enabled": "false",
                    "copyright_enabled": "false",
                },
            )
        assert response.status_code == 200
        records[source.name] = response.json()
    return records


def extract(client: TestClient, image: Image.Image, name: str):
    return client.post(
        "/api/watermark/extract",
        files={"file": (name, png_bytes(image), "image/png")},
    )


def test_embed_v2_records_rs_codec_metadata():
    reset_test_state()
    client = TestClient(main.app)
    source = Image.new("RGB", (384, 384), (100, 120, 140))

    response = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", png_bytes(source), "image/png")},
        data={
            "user_id": "pytest-rs-v2",
            "robust_watermark_version": "2",
            "copyright_enabled": "false",
        },
    )

    assert response.status_code == 200
    record = response.json()
    assert record["robust_watermark_version"] == 2
    assert record["robust_watermark_codec"] == "rs_24_8_three_phase"


def test_embed_v3_requires_auth_key(monkeypatch):
    reset_test_state()
    monkeypatch.setattr(main, "DEFAULT_WATERMARK_AUTH_KEY", "")
    client = TestClient(main.app)
    source = Image.new("RGB", (384, 384), (100, 120, 140))

    response = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", png_bytes(source), "image/png")},
        data={"user_id": "pytest-v3-no-key", "robust_watermark_version": "3"},
    )

    assert response.status_code == 503
    assert "WATERMARK_AUTH_KEY" in response.json()["detail"]


def test_embed_v3_records_hmac_metadata(monkeypatch):
    reset_test_state()
    monkeypatch.setattr(main, "DEFAULT_WATERMARK_AUTH_KEY", "v3-test-key-" * 4)
    client = TestClient(main.app)
    source = Image.new("RGB", (384, 384), (100, 120, 140))

    response = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", png_bytes(source), "image/png")},
        data={
            "user_id": "pytest-v3-metadata",
            "robust_watermark_version": "3",
            "robust_watermark_strength": "0.74",
            "copyright_enabled": "false",
        },
    )

    assert response.status_code == 200
    record = response.json()
    assert record["robust_watermark_version"] == 3
    assert record["robust_watermark_codec"] == "hmac64_full_repeat_phase_permutation_v3"
    assert len(record["robust_auth_code"]) == 16
    assert record["feature_index_path"].startswith("feature_index/")
    assert (main.DATA_DIR / record["feature_index_path"]).exists()


def test_unwatermarked_scaled_crops_are_not_traced_to_existing_records():
    reset_test_state()
    client = TestClient(main.app)
    seed_watermark_records(client)

    cases = [
        ("3.png", 0.5, 0.8),
        ("3.png", 1.5, 0.3),
        ("4.png", 0.5, 0.8),
        ("5.png", 1.5, 0.3),
    ]

    false_positives = []
    for filename, scale, crop_ratio in cases:
        image = attacked_unwatermarked(ROOT / "img" / filename, scale, crop_ratio)
        response = extract(client, image, f"negative_{filename}_{scale}_{crop_ratio}.png")
        if response.status_code == 200:
            false_positives.append(response.json())

    assert false_positives == []


def assert_correct_trace_or_safe_miss(response, expected_trace: str) -> None:
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        assert response.json()["trace_id"] == expected_trace


def test_large_scaled_crop_never_returns_a_wrong_trace():
    reset_test_state()
    client = TestClient(main.app)
    records = seed_watermark_records(client)

    image = attacked_watermarked(records["1.png"], scale=2.0, crop_ratio=0.8)
    response = extract(client, image, "watermarked_1_scale_2_crop_08.png")

    assert_correct_trace_or_safe_miss(response, records["1.png"]["trace_id"])


def test_low_texture_extreme_scaled_small_crop_never_returns_a_wrong_trace():
    reset_test_state()
    client = TestClient(main.app)
    records = seed_watermark_records(client)

    image = attacked_watermarked(records["5.png"], scale=2.0, crop_ratio=0.3)
    response = extract(client, image, "watermarked_5_scale_2_crop_03.png")

    assert_correct_trace_or_safe_miss(response, records["5.png"]["trace_id"])


def test_downscaled_small_crop_never_returns_a_wrong_trace():
    reset_test_state()
    client = TestClient(main.app)
    records = seed_watermark_records(client)

    image = attacked_watermarked(records["1.png"], scale=0.5, crop_ratio=0.3)
    response = extract(client, image, "watermarked_1_scale_05_crop_03.png")

    assert_correct_trace_or_safe_miss(response, records["1.png"]["trace_id"])
