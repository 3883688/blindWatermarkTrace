from pathlib import Path

import pytest

from trace_app.config import Settings


def test_production_v4_requires_postgresql_and_fixed_deadlines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="sqlite+pysqlite:///:memory:",
            admin_user="admin",
            admin_pass="secret",
            environment="production",
        )
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="postgresql+psycopg://trace:test@db/trace",
        admin_user="admin",
        admin_pass="secret",
        environment="production",
    )

    assert settings.v4_sync_p95_seconds == 120
    assert settings.v4_sync_timeout_seconds == 300
    assert settings.v4_deep_timeout_seconds == 1000


@pytest.mark.parametrize(
    "db_url",
    [
        "postgresql-evil://trace:test@db/trace",
        "postgresqlfoo://trace:test@db/trace",
        "postgresql",
        "postgresql://",
        "not a database URL",
    ],
)
def test_production_v4_rejects_malformed_or_non_postgresql_urls(
    tmp_path: Path, db_url: str
) -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url=db_url,
            admin_user="admin",
            admin_pass="secret",
            environment="production",
        )


def test_v4_settings_resolve_relative_paths_and_preserve_test_sqlite(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="sqlite+pysqlite:///:memory:",
        admin_user="admin",
        admin_pass="secret",
        environment="test",
        v4_model_manifest_path="models/v4-manifest.json",
        media_public_base_url="https://media.example.test/assets/",
    )

    assert settings.upload_dir == tmp_path / "uploads"
    assert settings.data_dir == tmp_path / "data"
    assert settings.v4_model_manifest_path == tmp_path / "models" / "v4-manifest.json"
    assert settings.media_public_base_url == "https://media.example.test/assets"


@pytest.mark.parametrize(
    "overrides",
    [
        {"v4_sync_worker_quota": 0},
        {"v4_deep_worker_quota": -1},
        {"v4_sync_p95_seconds": 121},
        {"v4_sync_timeout_seconds": 301},
        {"v4_deep_timeout_seconds": 1001},
        {"v4_sync_p95_seconds": 0},
        {"v4_sync_p95_seconds": 121, "v4_sync_timeout_seconds": 300},
        {"v4_sync_p95_seconds": 121, "v4_sync_timeout_seconds": 120},
        {"v4_sync_timeout_seconds": 119},
        {"v4_sync_timeout_seconds": 301, "v4_deep_timeout_seconds": 300},
        {"v4_deep_timeout_seconds": 299},
    ],
)
def test_v4_settings_reject_invalid_quotas_and_deadlines(
    tmp_path: Path, overrides: dict[str, int]
) -> None:
    values = {
        "base_dir": tmp_path,
        "upload_dir": "uploads",
        "data_dir": "data",
        "db_url": "sqlite+pysqlite:///:memory:",
        "admin_user": "admin",
        "admin_pass": "secret",
        "environment": "test",
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        Settings.from_values(**values)


def test_v4_settings_accept_lower_operational_deadlines_in_order(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="sqlite+pysqlite:///:memory:",
        admin_user="admin",
        admin_pass="secret",
        environment="test",
        v4_sync_p95_seconds=60,
        v4_sync_timeout_seconds=180,
        v4_deep_timeout_seconds=600,
        v4_sync_worker_quota=2,
        v4_deep_worker_quota=1,
    )

    assert (
        settings.v4_sync_p95_seconds,
        settings.v4_sync_timeout_seconds,
        settings.v4_deep_timeout_seconds,
    ) == (60, 180, 600)
    assert (settings.v4_sync_worker_quota, settings.v4_deep_worker_quota) == (2, 1)


def test_settings_repr_does_not_expose_database_or_admin_secrets(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="postgresql+psycopg://trace:database-secret@db/trace",
        admin_user="admin",
        admin_pass="admin-secret",
        environment="production",
    )

    assert "database-secret" not in repr(settings)
    assert "admin-secret" not in repr(settings)
