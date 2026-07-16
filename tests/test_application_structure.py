import ast
import importlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import main
from database_store import DatabaseStore
from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine
from trace_app.config import Settings
from trace_app.database.repositories import Repository
from trace_app.imaging.fingerprints import file_sha256
from trace_app.imaging.io import load_image_from_bytes
from trace_app.runtime import Runtime
from trace_app.watermark import small_crop as small_crop_module
from trace_app.watermark.lsb import bits_from_bytes, bytes_from_bits
from trace_app.watermark.small_crop import small_trace_short_code

from io import BytesIO
from PIL import Image


EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/site-logo.png"),
    ("GET", "/favicon.ico"),
    ("GET", "/favico.ico"),
    ("POST", "/auth/login"),
    ("GET", "/api/roles"),
    ("PUT", "/api/roles/{role_key}"),
    ("GET", "/api/users"),
    ("POST", "/api/users"),
    ("PUT", "/api/users/{username}"),
    ("DELETE", "/api/users/{username}"),
    ("POST", "/api/watermark/embed"),
    ("POST", "/api/watermark/extract"),
    ("POST", "/api/watermark/extract-url"),
    ("GET", "/api/dashboard-stats"),
    ("GET", "/api/images"),
    ("DELETE", "/api/images/{image_id}"),
    ("POST", "/api/dev/reset"),
}


def test_lsb_byte_bits_round_trip() -> None:
    assert bytes_from_bits(bits_from_bytes(b"trace")) == b"trace"


def test_detection_pipeline_calls_detectors_in_order() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    calls: list[str] = []

    pipeline = detection.DetectionPipeline(
        (
            lambda image: calls.append("first"),
            lambda image: calls.append("second") or {"trace_id": "matched"},
        )
    )

    assert pipeline(Image.new("RGB", (1, 1))) == {"trace_id": "matched"}
    assert calls == ["first", "second"]


def test_detection_pipeline_stops_after_first_match() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    calls: list[str] = []

    pipeline = detection.DetectionPipeline(
        (
            lambda image: calls.append("first") or {"trace_id": "matched"},
            lambda image: calls.append("second") or {"trace_id": "late"},
        )
    )

    assert pipeline(Image.new("RGB", (1, 1))) == {"trace_id": "matched"}
    assert calls == ["first"]


def test_small_trace_short_code_is_deterministic() -> None:
    assert small_trace_short_code("TRACE-20260716") == 14136


def test_robust_code_to_trace_uses_current_records_without_cache(
    monkeypatch,
) -> None:
    trace_id = "TRACE-DYNAMIC-RECORDS"
    code = main.robust_code_from_trace(trace_id)
    monkeypatch.setattr(main, "read_records", lambda: [{"trace_id": trace_id}])

    assert not hasattr(main.robust_code_to_trace, "cache_info")
    assert main.robust_code_to_trace(code) == trace_id

    monkeypatch.setattr(main, "read_records", lambda: [])

    assert main.robust_code_to_trace(code) is None


def test_small_crop_pattern_helpers_cache_by_arguments() -> None:
    calls = [
        (small_crop_module.small_trace_marker_pattern, (16,)),
        (small_crop_module.small_trace_pattern, ("TRACE-CACHE", 16)),
        (small_crop_module.small_trace_code_carriers, (16,)),
        (small_crop_module.small_trace_short_carriers, (16,)),
        (small_crop_module.code_cell_carriers, (0, 16)),
        (small_crop_module.code_tile_carriers, (16,)),
        (small_crop_module.code_trace_pattern, ("TRACE-CACHE", 16)),
        (small_crop_module.code_marker_pattern, (16,)),
    ]

    for helper, args in calls:
        helper.cache_clear()
        first = helper(*args)
        second = helper(*args)
        assert second is first
        assert helper.cache_info().hits == 1


def test_imaging_io_loads_image_from_bytes() -> None:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")

    assert load_image_from_bytes(buffer.getvalue()).size == (1, 1)


def test_imaging_fingerprints_hashes_file_bytes() -> None:
    assert file_sha256(b"abc") == (
        "BA7816BF8F01CFEA414140DE5DAE2223"
        "B00361A396177A9CB410FF61F20015AD"
    )


def test_main_alignment_wrapper_uses_patchable_resize_helper(
    tmp_path: Path, monkeypatch
) -> None:
    upload_dir = tmp_path / "uploads"
    target_path = upload_dir / "watermarked" / "target.png"
    target_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(target_path)

    resize_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        main,
        "resize_for_residual",
        lambda image, max_side=1200: resize_calls.append(image.size) or image,
    )

    main.align_query_to_record(
        Image.new("RGB", (16, 16), "white"),
        {"download_url": "/uploads/watermarked/target.png"},
    )

    assert resize_calls == [(16, 16), (16, 16)]


def test_main_candidate_ranking_wrapper_passes_current_feature_constants(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main.imaging_feature_matching, "rank_aligned_candidates", capture)
    monkeypatch.setattr(main, "FEATURE_MATCH_MIN_GOOD", 101)
    monkeypatch.setattr(main, "FEATURE_RECENT_BACKFILL", 102)
    monkeypatch.setattr(main, "FEATURE_RECENT_RESERVE", 103)

    main.rank_aligned_candidates(Image.new("RGB", (1, 1)), [])

    assert captured["feature_match_min_good"] == 101
    assert captured["feature_recent_backfill"] == 102
    assert captured["feature_recent_reserve"] == 103


def test_main_exposes_expected_routes() -> None:
    actual_routes = Counter(
        (method, route.path)
        for route in main.app.routes
        for method in getattr(route, "methods", set())
    )

    for expected_route in EXPECTED_ROUTES:
        assert actual_routes[expected_route] == 1


def test_main_exposes_required_python_api() -> None:
    required = {
        "ADMIN_PASS",
        "ADMIN_USER",
        "app",
        "ensure_dirs",
        "embed_robust_watermark",
        "embed_robust_watermark_v2",
        "embed_robust_watermark_v3",
        "detect_aligned_authenticated_watermark",
        "extract_watermark_from_image",
        "align_query_to_record",
        "file_sha256",
    }

    assert required <= set(dir(main))


def test_main_still_contains_watermark_endpoint_implementation() -> None:
    source = Path(main.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    top_level_functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "embed_watermark" in top_level_functions


def test_settings_resolves_relative_directories_from_base_dir(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.upload_dir == tmp_path / "uploads"
    assert settings.data_dir == tmp_path / "data"
    assert settings.original_dir == tmp_path / "uploads" / "originals"


def test_settings_direct_construction_uses_default_app_name(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.app_name == "WatermarkSystem"


def test_settings_from_values_reads_app_name_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_NAME", "EnvironmentApp")

    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.app_name == "EnvironmentApp"


def test_config_loads_dotenv_before_constructing_global_settings() -> None:
    script = """
import os
import sys
import types

dotenv = types.ModuleType("dotenv")

def load_dotenv():
    os.environ["APP_NAME"] = "LoadedFromDotenv"

dotenv.load_dotenv = load_dotenv
sys.modules["dotenv"] = dotenv

from trace_app.config import settings

assert settings.app_name == "LoadedFromDotenv", settings.app_name
"""
    env = os.environ.copy()
    env.pop("APP_NAME", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_starts_without_database_state() -> None:
    runtime = Runtime()

    assert runtime.engine is None
    assert runtime.store is None
    assert runtime.generated_trace_ids == []


def test_repository_without_store_reports_database_unavailable() -> None:
    repository = Repository(None)

    with pytest.raises(HTTPException) as exc_info:
        repository.read_records()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "数据库不可用"


def test_repository_replaces_and_reads_records() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    repository = Repository(store)

    repository.replace_records([{"id": "one"}])

    assert repository.read_records() == [{"id": "one"}]


def test_disabled_database_initialization_preserves_existing_state(
    monkeypatch,
) -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    existing_runtime = Runtime(store=store)
    existing_repository = Repository(store)
    monkeypatch.setattr(main, "DB_ENABLED", False)
    monkeypatch.setattr(main, "runtime", existing_runtime)
    monkeypatch.setattr(main, "repository", existing_repository)
    monkeypatch.setattr(main, "db_store", store)

    main.initialize_database()

    assert main.runtime is existing_runtime
    assert main.repository is existing_repository
    assert main.db_store is store


def test_failed_database_initialization_synchronizes_compatibility_state(
    tmp_path: Path, monkeypatch
) -> None:
    original_runtime = main.runtime
    original_repository = main.repository
    original_db_engine = main.db_engine
    original_db_store = main.db_store
    original_db_error = main.db_error
    monkeypatch.setattr(main, "runtime", original_runtime)
    monkeypatch.setattr(main, "repository", original_repository)
    monkeypatch.setattr(main, "db_engine", original_db_engine)
    monkeypatch.setattr(main, "db_store", original_db_store)
    monkeypatch.setattr(main, "db_error", original_db_error)
    unavailable_path = tmp_path / "missing" / "runtime.sqlite3"
    failing_settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{unavailable_path}",
        admin_user="admin",
        admin_pass="secret",
    )
    monkeypatch.setattr(main, "DB_ENABLED", True)
    monkeypatch.setattr(main, "settings", failing_settings)

    with pytest.raises(RuntimeError, match="Database initialization failed"):
        main.initialize_database()

    assert main.runtime.db_error == "OperationalError"
    assert main.runtime.store is None
    assert main.db_store is None
    assert main.db_error == "OperationalError"

    monkeypatch.undo()

    assert main.runtime is original_runtime
    assert main.repository is original_repository
    assert main.db_engine is original_db_engine
    assert main.db_store is original_db_store
    assert main.db_error == original_db_error


def test_repository_ensures_directories_before_writing_stats() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    calls: list[str] = []
    repository = Repository(store, ensure_dirs=lambda: calls.append("called"))

    repository.write_detection_stats({"attempts": 1, "successes": 1})
    repository.write_watermark_stats({"daily": {"2026-07-16": 1}})

    assert calls == ["called", "called"]


def test_main_today_watermark_count_uses_patchable_date_predicate(
    monkeypatch,
) -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    monkeypatch.setattr(main, "repository", Repository(store))
    monkeypatch.setattr(main, "read_watermark_stats", lambda: {"daily": {}})
    monkeypatch.setattr(
        main,
        "is_today_record",
        lambda record: record.get("id") == "selected",
    )

    count = main.today_watermark_count([{"id": "selected"}, {"id": "other"}])

    assert count == 1
