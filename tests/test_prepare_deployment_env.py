import base64
import subprocess
import sys
from pathlib import Path

import pytest


def _values(path: Path, name: str) -> list[str]:
    prefix = f"{name}="
    return [
        line[len(prefix) :]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]


def test_prepare_environment_generates_v4_key_and_preserves_unrelated_lines(
    tmp_path: Path,
) -> None:
    from tools.prepare_deployment_env import prepare_environment

    path = tmp_path / ".env"
    path.write_text(
        "# keep\n"
        "DB_ENABLED=true\n"
        "DB_URL=mysql+pymysql://REMOVED:secret@127.0.0.1/trace\n"
        "APP_NAME=WatermarkSystem\n"
        "WATERMARK_AUTH_KEY=\n"
        "ROBUST_WATERMARK_VERSION=1\n",
        encoding="utf-8",
    )

    result = prepare_environment(path)

    keys = _values(path, "WATERMARK_AUTH_KEY")
    assert result == {"generated": True, "utf8_bytes": 64, "entries": 1}
    assert len(keys) == 1
    assert len(base64.b64decode(keys[0], validate=True)) == 48
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]
    content = path.read_text(encoding="utf-8")
    assert "# keep" in content
    assert "APP_NAME=WatermarkSystem" in content
    assert keys[0] not in str(result)


def test_prepare_environment_preserves_valid_key_on_repeated_run(
    tmp_path: Path,
) -> None:
    from tools.prepare_deployment_env import prepare_environment

    path = tmp_path / ".env"
    existing = base64.b64encode(b"x" * 48).decode("ascii")
    path.write_text(
        f"DB_ENABLED=false\nWATERMARK_AUTH_KEY={existing}\n"
        "ROBUST_WATERMARK_VERSION=1\n",
        encoding="utf-8",
    )

    first = prepare_environment(path)
    second = prepare_environment(path)

    assert first["generated"] is False
    assert second["generated"] is False
    assert _values(path, "WATERMARK_AUTH_KEY") == [existing]
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]


def test_prepare_environment_replaces_duplicate_managed_entries(
    tmp_path: Path,
) -> None:
    from tools.prepare_deployment_env import prepare_environment

    path = tmp_path / ".env"
    path.write_text(
        "DB_ENABLED=false\n"
        "WATERMARK_AUTH_KEY=first-value-that-is-long-enough-123456\n"
        "WATERMARK_AUTH_KEY=second-value-that-is-long-enough-12345\n"
        "ROBUST_WATERMARK_VERSION=3\n"
        "ROBUST_WATERMARK_VERSION=1\n",
        encoding="utf-8",
    )

    result = prepare_environment(path)

    assert result["generated"] is True
    assert len(_values(path, "WATERMARK_AUTH_KEY")) == 1
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]


def test_prepare_environment_requires_db_url_when_database_enabled(
    tmp_path: Path,
) -> None:
    from tools.prepare_deployment_env import prepare_environment

    path = tmp_path / ".env"
    path.write_text("DB_ENABLED=true\nDB_URL=\n", encoding="utf-8")

    with pytest.raises(ValueError, match="DB_URL"):
        prepare_environment(path)


def test_prepare_environment_cli_never_prints_generated_key(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("DB_ENABLED=false\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/prepare_deployment_env.py",
            "--env-file",
            str(path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    keys = _values(path, "WATERMARK_AUTH_KEY")
    assert len(keys) == 1
    key = keys[0]
    assert "generated=True" in completed.stdout
    assert "utf8_bytes=64" in completed.stdout
    assert key
    assert key not in completed.stdout
    assert key not in completed.stderr
