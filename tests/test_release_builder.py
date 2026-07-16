from pathlib import Path

import pytest

import tools.build_centos_release as builder
from tools.build_centos_release import ROOT_FILES, is_release_source, release_files


def _write_required_root_files(root: Path, *, missing: str | None = None) -> None:
    for relative in ROOT_FILES:
        if relative == missing:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"required")


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
    _write_required_root_files(tmp_path)
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


@pytest.mark.parametrize("missing", ("main.py", "deploy.sh", "requirements.txt"))
def test_release_collector_fails_when_required_root_file_is_missing(
    tmp_path: Path,
    missing: str,
) -> None:
    _write_required_root_files(tmp_path, missing=missing)

    with pytest.raises(FileNotFoundError) as error:
        release_files(tmp_path)

    assert missing in str(error.value)


@pytest.mark.parametrize("linked_name", ("main.py", "trace_app", "trace_app/module.py"))
def test_release_collector_rejects_link_or_junction_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_name: str,
) -> None:
    _write_required_root_files(tmp_path)
    for tree in builder.RECURSIVE_TREES:
        (tmp_path / tree).mkdir(parents=True, exist_ok=True)
    (tmp_path / "trace_app/module.py").write_bytes(b"module")
    monkeypatch.setattr(
        builder,
        "_is_link_or_junction",
        lambda path: path.relative_to(tmp_path).as_posix() == linked_name,
        raising=False,
    )

    with pytest.raises(ValueError, match="link|junction"):
        release_files(tmp_path)


def test_release_collector_rejects_real_external_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_required_root_files(source)
    for tree in builder.RECURSIVE_TREES:
        (source / tree).mkdir(parents=True, exist_ok=True)
    secret = tmp_path / ".env"
    secret.write_text("SECRET=value", encoding="utf-8")
    link = source / "trace_app/external.py"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="link|junction"):
        release_files(source)


def test_build_rejects_external_release_root_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(builder, "RELEASE_ROOT", outside)
    monkeypatch.setattr(builder, "RELEASE_ARCHIVE", tmp_path / "release.zip")
    monkeypatch.setattr(builder, "RELEASE_CHECKSUM", tmp_path / "release.zip.sha256")

    with pytest.raises(ValueError, match="release"):
        builder.build_release()

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "target_name",
    ("RELEASE_PARENT", "RELEASE_ARCHIVE", "RELEASE_CHECKSUM"),
)
def test_build_rejects_external_release_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    target = tmp_path / target_name.casefold()
    monkeypatch.setattr(builder, target_name, target)
    release_main = builder.RELEASE_ROOT / "main.py"
    original = release_main.read_bytes()

    with pytest.raises(ValueError, match="release"):
        builder.build_release()

    assert release_main.read_bytes() == original


@pytest.mark.parametrize("linked_target", ("RELEASE_PARENT", "RELEASE_ROOT"))
def test_build_rejects_linked_release_target_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    linked_target: str,
) -> None:
    target = getattr(builder, linked_target)
    release_main = builder.RELEASE_ROOT / "main.py"
    original = release_main.read_bytes()
    monkeypatch.setattr(
        builder,
        "_is_link_or_junction",
        lambda path: path == target,
        raising=False,
    )

    with pytest.raises(ValueError, match="link|junction"):
        builder.build_release()

    assert release_main.read_bytes() == original
