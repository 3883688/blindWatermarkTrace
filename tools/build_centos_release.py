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


def release_files() -> tuple[Path, ...]:
    paths = {Path(relative) for relative in ROOT_FILES}
    for tree in RECURSIVE_TREES:
        paths.update(
            path.relative_to(ROOT)
            for path in (ROOT / tree).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
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
