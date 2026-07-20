import hashlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import zipfile

import pytest

import tools.build_centos_release as builder
from tools.build_centos_release import ROOT_FILES, is_release_source, release_files


BUILD_TIME = datetime(2026, 7, 20, 14, 35, 42)


def test_release_targets_use_current_package_time() -> None:
    targets = builder.release_targets(BUILD_TIME)

    assert targets.name == "trace-v4-centos-20260720-143542"
    assert targets.root == builder.RELEASE_PARENT / targets.name
    assert targets.archive == builder.RELEASE_PARENT / f"{targets.name}.zip"
    assert targets.checksum == targets.archive.with_suffix(".zip.sha256")
    assert targets.zip_timestamp == (2026, 7, 20, 14, 35, 42)


def _write_required_root_files(root: Path, *, missing: str | None = None) -> None:
    for relative in ROOT_FILES:
        if relative == missing:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"required")


def test_build_writes_artifacts_to_the_requested_package_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_required_root_files(source)
    for tree in builder.RECURSIVE_TREES:
        (source / tree).mkdir(parents=True, exist_ok=True)
    release_parent = source / "release"
    release_parent.mkdir()
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "RELEASE_PARENT", release_parent)

    digest = builder.build_release(BUILD_TIME)

    targets = builder.release_targets(BUILD_TIME)
    assert targets.root.is_dir()
    assert targets.archive.is_file()
    assert targets.checksum.read_text(encoding="ascii") == (
        f"{digest}  {targets.archive.name}\n"
    )
    with zipfile.ZipFile(targets.archive) as package:
        assert package.getinfo("main.py").date_time == targets.zip_timestamp


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
        "trace_app/credentials.json",
        "trace_app/.npmrc",
        "trace_app/nested/.hidden.py",
        "trace_app/nested/.git/config",
        "trace_app/nested/.venv/module.py",
        "trace_app/nested/venv/module.py",
        "trace_app/nested/tools/helper.py",
        "assets/customer-export.csv",
        "assets/unknown.bin",
        "assets/metadata.json",
        "trace_app/metadata.json",
        "unknown/module.py",
    ),
)
def test_release_source_filter_rejects_private_and_development_paths(
    relative: str,
) -> None:
    assert not is_release_source(Path(relative))


@pytest.mark.parametrize(
    "relative",
    (
        "trace_app/password_security.py",
        "trace_app/module.py",
        "watermark_v4/config.py",
        "assets/tabler-icons.css",
        "assets/fonts/tabler-icons.ttf",
        "assets/fonts/tabler-icons.woff",
        "assets/fonts/tabler-icons.woff2",
    ),
)
def test_release_source_filter_allows_runtime_sources(relative: str) -> None:
    assert is_release_source(Path(relative))


def test_release_source_filter_allows_compiled_frontend_javascript() -> None:
    assert is_release_source(Path("assets/app/app.js"))


def test_root_password_security_module_remains_explicitly_allowed() -> None:
    assert "password_security.py" in ROOT_FILES


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
        "trace_app/credentials.json",
        "trace_app/.npmrc",
        "trace_app/nested/.git/config",
        "assets/customer-export.csv",
        "assets/unknown.bin",
        "assets/metadata.json",
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
    targets = replace(builder.release_targets(BUILD_TIME), root=outside)

    with pytest.raises(ValueError, match="release"):
        builder.build_release(targets=targets)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "target_name",
    ("parent", "archive", "checksum"),
)
def test_build_rejects_external_release_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    if target_name == "parent":
        monkeypatch.setattr(builder, "RELEASE_PARENT", outside)
        targets = builder.release_targets(BUILD_TIME)
    else:
        targets = replace(
            builder.release_targets(BUILD_TIME),
            **{target_name: outside / f"release.{target_name}"},
        )

    with pytest.raises(ValueError, match="release"):
        builder.build_release(targets=targets)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_build_is_deterministic_and_does_not_modify_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_required_root_files(source)
    recursive_files = {
        "trace_app/module.py": b"trace fixture",
        "watermark_v4/config.py": b"watermark fixture",
        "assets/app.css": b"asset fixture",
        "assets/fonts/icons.woff2": b"font fixture",
    }
    for relative, content in recursive_files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    release_parent = source / "release"
    release_parent.mkdir()
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "RELEASE_PARENT", release_parent)
    targets = builder.release_targets(BUILD_TIME)
    source_paths = release_files(source)
    source_hashes = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in source_paths
    }

    first_digest = builder.build_release(targets=targets)
    first_archive = targets.archive.read_bytes()
    second_digest = builder.build_release(targets=targets)

    assert first_digest == second_digest
    assert targets.archive.read_bytes() == first_archive
    assert {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in source_paths
    } == source_hashes
    with zipfile.ZipFile(targets.archive) as package:
        assert package.read("main.py") == b"required"
        assert package.read("trace_app/module.py") == b"trace fixture"


@pytest.mark.parametrize("linked_target", ("parent", "root"))
def test_build_rejects_linked_release_target_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    linked_target: str,
) -> None:
    targets = builder.release_targets(BUILD_TIME)
    target = builder.RELEASE_PARENT if linked_target == "parent" else targets.root
    release_root = max(
        path
        for path in builder.RELEASE_PARENT.glob("trace-v4-centos-????????-??????")
        if path.is_dir()
    )
    release_main = release_root / "main.py"
    original = release_main.read_bytes()
    monkeypatch.setattr(
        builder,
        "_is_link_or_junction",
        lambda path: path == target,
        raising=False,
    )

    with pytest.raises(ValueError, match="link|junction"):
        builder.build_release(targets=targets)

    assert release_main.read_bytes() == original
