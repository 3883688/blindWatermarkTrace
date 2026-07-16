from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAME = "trace-v4-centos-20260715"
RELEASE_PARENT = ROOT / "release"
RELEASE_ROOT = RELEASE_PARENT / RELEASE_NAME
RELEASE_ARCHIVE = RELEASE_PARENT / f"{RELEASE_NAME}.zip"
RELEASE_CHECKSUM = RELEASE_ARCHIVE.with_suffix(".zip.sha256")
ZIP_TIMESTAMP = (2026, 7, 15, 0, 0, 0)

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
EXCLUDED_DIRECTORIES = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "secrets",
    "tests",
}
EXCLUDED_SUFFIXES = {".key", ".p12", ".pem", ".pfx", ".pyc", ".pyo"}
SSH_KEY_NAMES = ("id_dsa", "id_ecdsa", "id_ed25519", "id_rsa")


def is_release_source(relative: Path) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    if any(part in EXCLUDED_DIRECTORIES for part in parts[:-1]):
        return False
    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return False
    if any(name == key or name.startswith(f"{key}.") for key in SSH_KEY_NAMES):
        return False
    if name.startswith("secrets") and name.endswith(".json"):
        return False
    return relative.suffix.casefold() not in EXCLUDED_SUFFIXES


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


def release_files(root: Path = ROOT) -> tuple[Path, ...]:
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
                if name.casefold() not in EXCLUDED_DIRECTORIES:
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


def _validate_release_targets() -> None:
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

    expected_root = RELEASE_PARENT / RELEASE_NAME
    if Path(os.path.abspath(RELEASE_ROOT)) != Path(os.path.abspath(expected_root)):
        raise ValueError(f"Invalid release directory: {RELEASE_ROOT}")
    if _is_link_or_junction(RELEASE_ROOT):
        raise ValueError(
            f"Release target link or junction is forbidden: {RELEASE_ROOT}"
        )
    resolved_root = _resolve_inside(RELEASE_ROOT, real_parent, strict=False)
    if resolved_root != real_parent / RELEASE_NAME or resolved_root == real_parent:
        raise ValueError(f"Invalid release directory: {RELEASE_ROOT}")

    expected_archive = RELEASE_PARENT / f"{RELEASE_NAME}.zip"
    expected_checksum = expected_archive.with_suffix(".zip.sha256")
    for actual, expected in (
        (RELEASE_ARCHIVE, expected_archive),
        (RELEASE_CHECKSUM, expected_checksum),
    ):
        if Path(os.path.abspath(actual)) != Path(os.path.abspath(expected)):
            raise ValueError(f"Invalid release artifact path: {actual}")
        if _is_link_or_junction(actual):
            raise ValueError(f"Release target link or junction is forbidden: {actual}")
        if _resolve_inside(actual, real_parent, strict=False) != real_parent / expected.name:
            raise ValueError(f"Invalid release artifact path: {actual}")


def build_release() -> str:
    _validate_release_targets()
    files = release_files()
    if RELEASE_ROOT.exists():
        shutil.rmtree(RELEASE_ROOT)
    RELEASE_ROOT.mkdir(parents=True)

    for relative in files:
        destination = RELEASE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    with zipfile.ZipFile(
        RELEASE_ARCHIVE,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as package:
        for relative in files:
            info = zipfile.ZipInfo(relative.as_posix(), ZIP_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            permissions = 0o755 if relative.as_posix() == "deploy.sh" else 0o644
            info.external_attr = (stat.S_IFREG | permissions) << 16
            package.writestr(info, (ROOT / relative).read_bytes(), compresslevel=9)

    digest = hashlib.sha256(RELEASE_ARCHIVE.read_bytes()).hexdigest()
    RELEASE_CHECKSUM.write_text(
        f"{digest}  {RELEASE_ARCHIVE.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return digest


if __name__ == "__main__":
    print(build_release())
