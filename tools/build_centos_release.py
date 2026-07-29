from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

from tests.v4.benchmark_manifest import verify_signed_report


ROOT = Path(__file__).resolve().parents[1]
RELEASE_PREFIX = "trace-v4-centos"
RELEASE_PARENT = ROOT / "release"


@dataclass(frozen=True)
class ReleaseTargets:
    name: str
    root: Path
    archive: Path
    checksum: Path
    zip_timestamp: tuple[int, int, int, int, int, int]


def release_targets(build_time: datetime | None = None) -> ReleaseTargets:
    build_time = datetime.now() if build_time is None else build_time
    name = f"{RELEASE_PREFIX}-{build_time:%Y%m%d-%H%M%S}"
    archive = RELEASE_PARENT / f"{name}.zip"
    return ReleaseTargets(
        name=name,
        root=RELEASE_PARENT / name,
        archive=archive,
        checksum=archive.with_suffix(".zip.sha256"),
        zip_timestamp=(
            build_time.year,
            build_time.month,
            build_time.day,
            build_time.hour,
            build_time.minute,
            build_time.second,
        ),
    )

ROOT_FILES = (
    ".env.example",
    "README_DEPLOY.md",
    "candidate_feature_index.py",
    "database_store.py",
    "deploy.sh",
    "favico.ico",
    "favicon.ico",
    "index.html",
    "logo.png",
    "main.py",
    "password_security.py",
    "requirements.txt",
    "site-logo.png",
    "tools/migrate_json_to_mysql.py",
    "tools/prepare_deployment_env.py",
    "watermark_auth.py",
    "watermark_ecc.py",
)
RECURSIVE_TREES = ("assets", "trace_app", "watermark_v4")
RECURSIVE_SUFFIX_ALLOWLIST = {
    "assets": {".css", ".js", ".ttf", ".woff", ".woff2"},
    "trace_app": {".py"},
    "watermark_v4": {".py"},
}
FORBIDDEN_DIRECTORIES = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "cvs",
    "env",
    "node_modules",
    "secrets",
    "tests",
    "tool",
    "tools",
    "venv",
    "virtualenv",
}


def is_release_source(relative: Path) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    if len(parts) < 2 or any(part.startswith(".") for part in parts):
        return False
    allowed_suffixes = RECURSIVE_SUFFIX_ALLOWLIST.get(parts[0])
    if allowed_suffixes is None:
        return False
    if any(part in FORBIDDEN_DIRECTORIES for part in parts[1:-1]):
        return False
    return relative.suffix.casefold() in allowed_suffixes


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _reject_link_components(path: Path, root: Path) -> None:
    current = root
    if _is_link_or_junction(current):
        raise ValueError(f"Release source link or junction is forbidden: {current}")
    for part in path.relative_to(root).parts:
        current /= part
        if _is_link_or_junction(current):
            raise ValueError(
                f"Release source link or junction is forbidden: {current}"
            )


def _resolve_inside(path: Path, allowed_root: Path, *, strict: bool) -> Path:
    try:
        resolved = path.resolve(strict=strict)
        allowed = allowed_root.resolve(strict=True)
        resolved.relative_to(allowed)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Release path escapes its allowed root: {path}"
        ) from exc
    return resolved


def _validate_source_entry(path: Path, allowed_root: Path) -> Path:
    _reject_link_components(path, allowed_root)
    return _resolve_inside(path, allowed_root, strict=True)


def release_files(root: Path | None = None) -> tuple[Path, ...]:
    root = ROOT if root is None else root
    if not root.is_dir():
        raise FileNotFoundError(f"Release source root is missing: {root}")
    _reject_link_components(root, root)
    real_root = root.resolve(strict=True)
    paths = set()
    for relative in ROOT_FILES:
        source = root / relative
        _reject_link_components(source, root)
        if not source.is_file():
            raise FileNotFoundError(f"Required release source is missing: {relative}")
        _resolve_inside(source, real_root, strict=True)
        paths.add(Path(relative))
    for tree in RECURSIVE_TREES:
        tree_root = root / tree
        _reject_link_components(tree_root, root)
        if not tree_root.is_dir():
            raise FileNotFoundError(f"Required release source tree is missing: {tree}")
        _resolve_inside(tree_root, real_root, strict=True)
        for current_name, directory_names, file_names in os.walk(
            tree_root,
            topdown=True,
            followlinks=False,
        ):
            current = Path(current_name)
            kept_directories = []
            for name in directory_names:
                directory = current / name
                _validate_source_entry(directory, tree_root)
                if (
                    not name.startswith(".")
                    and name.casefold() not in FORBIDDEN_DIRECTORIES
                ):
                    kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in file_names:
                source = current / name
                resolved = _validate_source_entry(source, tree_root)
                if not resolved.is_file():
                    raise ValueError(f"Release source is not a file: {source}")
                relative = source.relative_to(root)
                if is_release_source(relative):
                    paths.add(relative)
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _validate_release_targets(targets: ReleaseTargets) -> None:
    if _is_link_or_junction(ROOT):
        raise ValueError(f"Release root link or junction is forbidden: {ROOT}")
    real_root = ROOT.resolve(strict=True)
    expected_parent = ROOT / "release"
    if Path(os.path.abspath(RELEASE_PARENT)) != Path(os.path.abspath(expected_parent)):
        raise ValueError(f"Invalid release parent: {RELEASE_PARENT}")
    if _is_link_or_junction(RELEASE_PARENT):
        raise ValueError(
            f"Release target link or junction is forbidden: {RELEASE_PARENT}"
        )
    real_parent = _resolve_inside(RELEASE_PARENT, real_root, strict=True)
    if real_parent == real_root:
        raise ValueError(f"Release parent must be inside source root: {RELEASE_PARENT}")

    expected_root = RELEASE_PARENT / targets.name
    if Path(os.path.abspath(targets.root)) != Path(os.path.abspath(expected_root)):
        raise ValueError(f"Invalid release directory: {targets.root}")
    if _is_link_or_junction(targets.root):
        raise ValueError(
            f"Release target link or junction is forbidden: {targets.root}"
        )
    resolved_root = _resolve_inside(targets.root, real_parent, strict=False)
    if resolved_root != real_parent / targets.name or resolved_root == real_parent:
        raise ValueError(f"Invalid release directory: {targets.root}")

    expected_archive = RELEASE_PARENT / f"{targets.name}.zip"
    expected_checksum = expected_archive.with_suffix(".zip.sha256")
    for actual, expected in (
        (targets.archive, expected_archive),
        (targets.checksum, expected_checksum),
    ):
        if Path(os.path.abspath(actual)) != Path(os.path.abspath(expected)):
            raise ValueError(f"Invalid release artifact path: {actual}")
        if _is_link_or_junction(actual):
            raise ValueError(f"Release target link or junction is forbidden: {actual}")
        if _resolve_inside(actual, real_parent, strict=False) != real_parent / expected.name:
            raise ValueError(f"Invalid release artifact path: {actual}")


def _require_release_evidence() -> None:
    report_path = os.getenv("V4_RELEASE_REPORT", "")
    report_key = os.getenv("V4_RELEASE_REPORT_KEY", "").encode()
    if not report_path or not report_key:
        raise RuntimeError("Current signed V4 release evidence is required")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    verify_signed_report(report, report_key, git_commit=current_commit)


def build_release(
    build_time: datetime | None = None,
    *,
    targets: ReleaseTargets | None = None,
) -> str:
    if build_time is not None and targets is not None:
        raise ValueError("Specify either build_time or targets, not both")
    targets = release_targets(build_time) if targets is None else targets
    _validate_release_targets(targets)
    _require_release_evidence()
    files = release_files(ROOT)
    if targets.root.exists():
        shutil.rmtree(targets.root)
    targets.root.mkdir(parents=True)

    for relative in files:
        destination = targets.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    with zipfile.ZipFile(
        targets.archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as package:
        for relative in files:
            info = zipfile.ZipInfo(relative.as_posix(), targets.zip_timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            permissions = 0o755 if relative.as_posix() == "deploy.sh" else 0o644
            info.external_attr = (stat.S_IFREG | permissions) << 16
            package.writestr(info, (ROOT / relative).read_bytes(), compresslevel=9)

    digest = hashlib.sha256(targets.archive.read_bytes()).hexdigest()
    targets.checksum.write_text(
        f"{digest}  {targets.archive.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return digest


if __name__ == "__main__":
    current_time = datetime.now()
    current_targets = release_targets(current_time)
    print(current_targets.archive)
    print(build_release(targets=current_targets))
