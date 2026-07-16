from pathlib import Path

import pytest

from tools.build_centos_release import is_release_source, release_files


@pytest.mark.parametrize(
    "relative",
    (
        "trace_app/tests/test_service.py",
        "trace_app/nested/__pycache__/service.pyc",
        "assets/.pytest_cache/state",
        "trace_app/.mypy_cache/state.json",
        "trace_app/nested/.ruff_cache/state",
        "trace_app/nested/.cache/state",
        "trace_app/service.pyc",
        "trace_app/service.pyo",
        "trace_app/.env",
        "trace_app/nested/.env.production",
        "trace_app/nested/.env.example",
        "assets/private.pem",
        "assets/private.p12",
        "assets/private.pfx",
        "trace_app/signing.key",
        "trace_app/id_rsa",
        "trace_app/id_rsa.pub",
        "trace_app/id_dsa",
        "trace_app/id_ecdsa",
        "trace_app/id_ed25519",
        "assets/secrets.json",
        "assets/secrets-production.json",
    ),
)
def test_release_source_filter_rejects_private_and_development_paths(
    relative: str,
) -> None:
    assert not is_release_source(Path(relative))


@pytest.mark.parametrize(
    "relative",
    (
        "password_security.py",
        "trace_app/password_security.py",
        "trace_app/module.py",
        "watermark_v4/config.py",
        "assets/fonts/tabler-icons.woff2",
    ),
)
def test_release_source_filter_allows_runtime_sources(relative: str) -> None:
    assert is_release_source(Path(relative))


def test_release_collector_filters_nested_fixture_tree(tmp_path: Path) -> None:
    allowed = {
        "trace_app/module.py",
        "trace_app/password_security.py",
        "watermark_v4/config.py",
        "assets/fonts/tabler-icons.woff2",
    }
    excluded = {
        "trace_app/tests/test_module.py",
        "trace_app/nested/__pycache__/module.pyc",
        "trace_app/nested/.env.production",
        "trace_app/nested/private.pem",
        "assets/.pytest_cache/state",
        "assets/secrets-release.json",
    }
    for relative in allowed | excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    collected = {path.as_posix() for path in release_files(tmp_path)}

    assert allowed <= collected
    assert not (excluded & collected)
