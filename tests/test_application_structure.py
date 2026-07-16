import ast
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
from trace_app.runtime import Runtime


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
