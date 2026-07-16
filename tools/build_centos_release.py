from __future__ import annotations

import hashlib
import shutil
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


def release_files(root: Path = ROOT) -> tuple[Path, ...]:
    paths = set()
    for relative in ROOT_FILES:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Required release source is missing: {relative}")
        paths.add(Path(relative))
    for tree in RECURSIVE_TREES:
        paths.update(
            relative
            for path in (root / tree).rglob("*")
            if path.is_file()
            for relative in (path.relative_to(root),)
            if is_release_source(relative)
        )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def build_release() -> str:
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
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if relative.as_posix() == "deploy.sh" else 0o100644
            info.external_attr = mode << 16
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
