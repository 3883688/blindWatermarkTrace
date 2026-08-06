import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOVED_ROOT_MODULES = {
    "candidate_feature_index",
    "database_store",
    "password_security",
    "watermark_auth",
    "watermark_ecc",
}
EXPECTED_MODULE_PATHS = {
    "trace_app/imaging/candidate_feature_index.py",
    "trace_app/database/store.py",
    "trace_app/auth/password_security.py",
    "trace_app/watermark/auth.py",
    "trace_app/watermark/ecc.py",
}


def _python_sources() -> tuple[Path, ...]:
    paths = [ROOT / "main.py"]
    for source_root in (
        ROOT / "trace_app",
        ROOT / "watermark_v4",
        ROOT / "tools",
        ROOT / "tests",
    ):
        paths.extend(source_root.rglob("*.py"))
    return tuple(sorted(paths))


def test_main_is_the_only_root_python_file() -> None:
    root_python_files = {path.name for path in ROOT.glob("*.py")}
    assert root_python_files == {"main.py"}


def test_domain_modules_exist_at_their_owned_paths() -> None:
    assert {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).as_posix() in EXPECTED_MODULE_PATHS
    } == EXPECTED_MODULE_PATHS


def test_stale_import_scan_covers_all_maintained_python_sources() -> None:
    scanned = {path.relative_to(ROOT).as_posix() for path in _python_sources()}
    assert "main.py" in scanned
    assert "watermark_v4/__init__.py" in scanned


def test_python_sources_do_not_import_removed_root_modules() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module.split(".", 1)[0]}
            else:
                continue
            stale = imported & REMOVED_ROOT_MODULES
            if stale:
                relative = path.relative_to(ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}:{sorted(stale)}")
    assert violations == []
