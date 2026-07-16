import hashlib
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cmd",
    ".example",
    ".html",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
}
SENSITIVE_FINGERPRINTS = {
    5: {"edc3aa98eafc7d8938b614366e13aaba6670b949296d71841bbc4a046707887f"},
    6: {"ca58aa965159baf811902e7855c89fdc042449361966e5c3ff985eb4d8d4ed86"},
    10: {"ece1144c57d408f79936d1c64a19fe178fbd6d62744827ef9a9dff0d77c98f7d"},
    18: {"c8702f2fc609f137ffe0668d9cad87282b94026e4e6684fc590a8b38e8d84771"},
}


def _tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in completed.stdout.split(b"\0") if value]


def _contains_sensitive_fingerprint(content: str) -> bool:
    for length, fingerprints in SENSITIVE_FINGERPRINTS.items():
        for index in range(len(content) - length + 1):
            candidate = content[index : index + length]
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            if digest in fingerprints:
                return True
    return False


def test_tracked_text_contains_no_known_credentials() -> None:
    leaked_paths = []
    for path in _tracked_paths():
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _contains_sensitive_fingerprint(content):
            leaked_paths.append(path.relative_to(ROOT).as_posix())
    assert leaked_paths == []


def test_environment_example_has_empty_credentials() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"(?m)^ADMIN_USER=$", content)
    assert re.search(r"(?m)^ADMIN_PASS=$", content)
    assert re.search(r"(?m)^DB_URL=$", content)


def test_source_and_frontend_have_no_credential_defaults() -> None:
    source = (ROOT / "trace_app" / "config.py").read_text(encoding="utf-8")
    frontend = (ROOT / "index.html").read_text(encoding="utf-8")
    assert 'os.getenv("DB_URL", "")' in source
    assert 'os.getenv("ADMIN_USER", "")' in source
    assert 'os.getenv("ADMIN_PASS", "")' in source
    assert "const ADMIN_USER=" not in frontend
    assert not re.search(r'id="loginUsername"[^>]+value=', frontend)


def test_runtime_json_paths_are_ignored_and_untracked() -> None:
    relative_paths = [
        "data/detection_stats.json",
        "data/images.json",
        "data/roles.json",
        "data/users.json",
        "data/watermark_stats.json",
    ]
    tracked = {path.relative_to(ROOT).as_posix() for path in _tracked_paths()}
    assert not tracked.intersection(relative_paths)
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *relative_paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert set(completed.stdout.splitlines()) == set(relative_paths)


def test_release_source_matches_sanitized_root() -> None:
    release = ROOT / "release" / "trace-v4-centos-20260715"
    for relative in (
        ".env.example",
        "README_DEPLOY.md",
        "database_store.py",
        "deploy.sh",
        "index.html",
        "main.py",
        "password_security.py",
        "tools/migrate_json_to_mysql.py",
    ):
        assert (release / relative).read_bytes() == (ROOT / relative).read_bytes()


def test_release_archive_excludes_private_runtime_files_when_present() -> None:
    archive = ROOT / "release" / "trace-v4-centos-20260715.zip"
    if not archive.exists():
        return
    leaked_entries = []
    with zipfile.ZipFile(archive) as package:
        names = {name.replace("\\", "/").lstrip("./") for name in package.namelist()}
        for entry in package.infolist():
            suffix = Path(entry.filename).suffix.lower()
            if entry.is_dir() or suffix not in TEXT_SUFFIXES:
                continue
            try:
                content = package.read(entry).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if _contains_sensitive_fingerprint(content):
                leaked_entries.append(entry.filename)
    assert ".env" not in names
    assert not any(name.startswith("data/") and name.endswith(".json") for name in names)
    assert "tools/migrate_json_to_mysql.py" in names
    assert leaked_entries == []
