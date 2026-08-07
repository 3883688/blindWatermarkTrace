# Trace App Root Module Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the five root production modules into their owning `trace_app` domains while preserving `main:app` startup and application behavior.

**Architecture:** Keep the three-line root `main.py` as the only top-level Python entrypoint. Move persistence, password security, image indexing, watermark authentication, and watermark ECC into existing domain packages; update every production, tool, test, and release reference to use absolute `trace_app.*` imports.

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy, pytest, Git worktrees, CentOS release builder

---

## File Structure

The implementation creates no new compatibility layer. These tracked moves define the final ownership boundaries:

```text
candidate_feature_index.py -> trace_app/imaging/candidate_feature_index.py
database_store.py          -> trace_app/database/store.py
password_security.py       -> trace_app/auth/password_security.py
watermark_auth.py          -> trace_app/watermark/auth.py
watermark_ecc.py           -> trace_app/watermark/ecc.py
main.py                    -> main.py  # unchanged compatibility entrypoint
```

The package `__init__.py` files remain lightweight. Callers import concrete submodules directly so package initialization does not create circular imports.

### Task 1: Move Database Persistence And Password Security

**Files:**
- Move: `database_store.py` -> `trace_app/database/store.py`
- Move: `password_security.py` -> `trace_app/auth/password_security.py`
- Modify: `trace_app/database/store.py`
- Modify: `trace_app/database/connection.py`
- Modify: `trace_app/database/repositories.py`
- Modify: `trace_app/runtime.py`
- Modify: `trace_app/auth/service.py`
- Modify: `tools/migrate_json_to_mysql.py`
- Modify: `tools/migrate_mysql_to_postgresql.py`
- Modify: `tools/migrate_v4_relational_only.py`
- Modify: `tests/test_application_structure.py`
- Modify: `tests/test_database_store.py`
- Modify: `tests/test_false_positive_gate.py`
- Modify: `tests/test_json_mysql_migration.py`
- Modify: `tests/test_password_security.py`
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `tests/v4/test_relational_only_schema.py`

- [ ] **Step 1: Change persistence and password tests to the new package imports**

Replace the old test imports with these exact imports:

```python
from trace_app.auth.password_security import hash_password, verify_password
from trace_app.database.store import DatabaseStore
```

In `tests/test_password_security.py`, replace the monkeypatch target with:

```python
monkeypatch.setattr(
    "trace_app.auth.password_security.hashlib.scrypt",
    lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("untrusted parameters reached scrypt")
    ),
)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_password_security.py tests/test_database_store.py tests/test_json_mysql_migration.py tests/v4/test_relational_only_schema.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `trace_app.auth.password_security` or `trace_app.database.store` because the files have not moved yet.

- [ ] **Step 3: Move both modules with Git history preserved**

Run:

```powershell
git mv password_security.py trace_app/auth/password_security.py
git mv database_store.py trace_app/database/store.py
```

- [ ] **Step 4: Update production and tool imports**

In `trace_app/database/store.py`, use:

```python
from trace_app.auth.password_security import hash_password, verify_password
```

In `trace_app/database/connection.py`, `trace_app/database/repositories.py`, and the `TYPE_CHECKING` block in `trace_app/runtime.py`, use:

```python
from trace_app.database.store import DatabaseStore
```

In `trace_app/auth/service.py`, use:

```python
from trace_app.auth.password_security import verify_password
```

In `tools/migrate_json_to_mysql.py`, use:

```python
from trace_app.auth.password_security import hash_password, verify_password  # noqa: E402
from trace_app.database.store import DatabaseStore  # noqa: E402
```

In `tools/migrate_mysql_to_postgresql.py`, use:

```python
from trace_app.database.store import DatabaseStore  # noqa: E402
```

In `tools/migrate_v4_relational_only.py`, use:

```python
from trace_app.database.store import DatabaseStore
```

- [ ] **Step 5: Run persistence, authentication, migration, and API regressions**

Run:

```powershell
python -m pytest tests/test_password_security.py tests/test_database_store.py tests/test_json_mysql_migration.py tests/v4/test_relational_only_schema.py tests/test_false_positive_gate.py tests/test_application_structure.py tests/test_watermark_v4_api.py -q
```

Expected: PASS with no collection errors and no behavioral assertion changes.

- [ ] **Step 6: Verify no stale persistence or password imports remain outside release fixtures**

Run:

```powershell
rg -n "from database_store|import database_store|from password_security|import password_security|password_security\.hashlib" trace_app tools tests
```

Expected: no matches.

- [ ] **Step 7: Commit the persistence move**

```powershell
git add trace_app/auth/password_security.py trace_app/database/store.py trace_app/database/connection.py trace_app/database/repositories.py trace_app/runtime.py trace_app/auth/service.py tools/migrate_json_to_mysql.py tools/migrate_mysql_to_postgresql.py tools/migrate_v4_relational_only.py tests/test_application_structure.py tests/test_database_store.py tests/test_false_positive_gate.py tests/test_json_mysql_migration.py tests/test_password_security.py tests/test_watermark_v4_api.py tests/v4/test_relational_only_schema.py
git commit -m "refactor: move persistence modules into trace_app"
```

### Task 2: Move The Candidate Feature Index

**Files:**
- Move: `candidate_feature_index.py` -> `trace_app/imaging/candidate_feature_index.py`
- Modify: `trace_app/imaging/candidate_feature_index.py`
- Modify: `trace_app/imaging/feature_matching.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/compat.py`
- Modify: `tests/test_candidate_feature_index.py`

- [ ] **Step 1: Change the feature-index test to the new import**

In `tests/test_candidate_feature_index.py`, use:

```python
from trace_app.imaging.candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
```

- [ ] **Step 2: Run the feature-index test and verify RED**

Run:

```powershell
python -m pytest tests/test_candidate_feature_index.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trace_app.imaging.candidate_feature_index'`.

- [ ] **Step 3: Move the module and update consumers**

Run:

```powershell
git mv candidate_feature_index.py trace_app/imaging/candidate_feature_index.py
```

In `trace_app/imaging/feature_matching.py` and `trace_app/compat.py`, use:

```python
from trace_app.imaging.candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
```

In `trace_app/watermark/default_operations.py`, use:

```python
from trace_app.imaging.candidate_feature_index import (
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
```

Update prose references from `candidate_feature_index` to
`trace_app.imaging.candidate_feature_index` in the moved module and
`trace_app/imaging/feature_matching.py`.

- [ ] **Step 4: Run feature matching and application regressions**

Run:

```powershell
python -m pytest tests/test_candidate_feature_index.py tests/test_application_structure.py tests/test_false_positive_gate.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify no stale feature-index imports remain**

Run:

```powershell
rg -n "from candidate_feature_index|import candidate_feature_index" trace_app tools tests
```

Expected: no matches.

- [ ] **Step 6: Commit the imaging move**

```powershell
git add trace_app/imaging/candidate_feature_index.py trace_app/imaging/feature_matching.py trace_app/watermark/default_operations.py trace_app/compat.py tests/test_candidate_feature_index.py
git commit -m "refactor: move feature index into imaging package"
```

### Task 3: Move Watermark Authentication And ECC

**Files:**
- Move: `watermark_auth.py` -> `trace_app/watermark/auth.py`
- Move: `watermark_ecc.py` -> `trace_app/watermark/ecc.py`
- Modify: `trace_app/watermark/auth.py`
- Modify: `trace_app/watermark/ecc.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/watermark/robust.py`
- Modify: `trace_app/compat.py`
- Modify: `tests/test_watermark_auth.py`
- Modify: `tests/test_watermark_ecc.py`

- [ ] **Step 1: Change watermark unit tests to the new imports**

In `tests/test_watermark_auth.py`, use:

```python
from trace_app.watermark.auth import (
    auth_code_from_trace,
    candidate_radius_probability,
    inverse_permutation,
    permuted_code_bits,
    phase_permutation,
)
```

In `tests/test_watermark_ecc.py`, use:

```python
from trace_app.watermark.ecc import (
    RS_CODEWORD_BYTES,
    RS_DATA_BYTES,
    codeword_phase,
    decode_expected_codeword,
    encode_codeword,
    tile_phase,
)
```

- [ ] **Step 2: Run both tests and verify RED**

Run:

```powershell
python -m pytest tests/test_watermark_auth.py tests/test_watermark_ecc.py -q
```

Expected: collection fails because `trace_app.watermark.auth` and
`trace_app.watermark.ecc` do not exist yet.

- [ ] **Step 3: Move both modules with Git history preserved**

Run:

```powershell
git mv watermark_auth.py trace_app/watermark/auth.py
git mv watermark_ecc.py trace_app/watermark/ecc.py
```

- [ ] **Step 4: Update watermark consumers and documentation links**

In `trace_app/compat.py`, use:

```python
from trace_app.watermark.auth import (
    auth_code_from_trace,
    inverse_permutation,
    permuted_code_bits,
    phase_permutation,
)
from trace_app.watermark.ecc import (
    codeword_phase,
    decode_expected_codeword,
    encode_codeword,
    tile_phase,
)
```

In `trace_app/watermark/default_operations.py`, use:

```python
from trace_app.watermark.auth import auth_code_from_trace
```

In `trace_app/watermark/robust.py`, use:

```python
from trace_app.watermark.auth import permuted_code_bits, phase_permutation
from trace_app.watermark.ecc import (
    codeword_phase,
    decode_expected_codeword,
    encode_codeword,
    tile_phase,
)
```

Update module documentation references from
`watermark_auth.py` and `watermark_ecc.py` to `trace_app.watermark.auth` and
`trace_app.watermark.ecc`.

- [ ] **Step 5: Run watermark and application regressions**

Run:

```powershell
python -m pytest tests/test_watermark_auth.py tests/test_watermark_ecc.py tests/test_application_structure.py tests/test_false_positive_gate.py -q
```

Expected: PASS.

- [ ] **Step 6: Verify no stale watermark imports remain**

Run:

```powershell
rg -n "from watermark_auth|import watermark_auth|from watermark_ecc|import watermark_ecc" trace_app tools tests
```

Expected: no matches.

- [ ] **Step 7: Commit the watermark move**

```powershell
git add trace_app/watermark/auth.py trace_app/watermark/ecc.py trace_app/watermark/default_operations.py trace_app/watermark/robust.py trace_app/compat.py tests/test_watermark_auth.py tests/test_watermark_ecc.py
git commit -m "refactor: move watermark codecs into watermark package"
```

### Task 4: Enforce The Root Layout And Update Release Packaging

**Files:**
- Create: `tests/test_package_layout.py`
- Modify: `tools/build_centos_release.py`
- Modify: `tests/test_release_builder.py`
- Modify: `tests/test_secret_hygiene.py`
- Modify: `tests/test_source_backup_contract.py`

- [ ] **Step 1: Add the package-layout contract**

Create `tests/test_package_layout.py` with:

```python
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


def test_main_is_the_only_root_python_file() -> None:
    root_python_files = {path.name for path in ROOT.glob("*.py")}
    assert root_python_files == {"main.py"}


def test_domain_modules_exist_at_their_owned_paths() -> None:
    assert {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if path.relative_to(ROOT).as_posix() in EXPECTED_MODULE_PATHS
    } == EXPECTED_MODULE_PATHS


def test_python_sources_do_not_import_removed_root_modules() -> None:
    violations: list[str] = []
    for source_root in (ROOT / "trace_app", ROOT / "tools", ROOT / "tests"):
        for path in source_root.rglob("*.py"):
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
```

- [ ] **Step 2: Add the new release expectation and verify RED**

Replace `test_root_password_security_module_remains_explicitly_allowed` in
`tests/test_release_builder.py` with:

```python
def test_moved_domain_modules_are_recursive_release_sources() -> None:
    moved = {
        "trace_app/imaging/candidate_feature_index.py",
        "trace_app/database/store.py",
        "trace_app/auth/password_security.py",
        "trace_app/watermark/auth.py",
        "trace_app/watermark/ecc.py",
    }
    assert moved <= {path.as_posix() for path in release_files()}
    assert not {
        "candidate_feature_index.py",
        "database_store.py",
        "password_security.py",
        "watermark_auth.py",
        "watermark_ecc.py",
    } & set(ROOT_FILES)
```

Run:

```powershell
python -m pytest tests/test_package_layout.py tests/test_release_builder.py::test_moved_domain_modules_are_recursive_release_sources -q
```

Expected: the layout tests pass after Tasks 1-3, but the release expectation fails
because the old paths are still listed in `ROOT_FILES`.

- [ ] **Step 3: Remove the old root entries from the release builder**

In `tools/build_centos_release.py`, remove these entries from `ROOT_FILES`:

```python
"candidate_feature_index.py",
"database_store.py",
"password_security.py",
"watermark_auth.py",
"watermark_ecc.py",
```

Keep `main.py` in `ROOT_FILES`. Do not add the moved paths there because
`RECURSIVE_TREES = ("assets", "trace_app", "watermark_v4")` already collects them.

- [ ] **Step 4: Update release and hygiene fixtures to the new paths**

In `tests/test_release_builder.py`, replace old fixture paths with:

```python
"trace_app/auth/password_security.py"
"trace_app/database/store.py"
"trace_app/imaging/candidate_feature_index.py"
"trace_app/watermark/auth.py"
"trace_app/watermark/ecc.py"
```

Update `tests/test_secret_hygiene.py` so its sanitized source expectations list
`main.py` as the only root Python source and include the five paths above through
the recursive `trace_app` source set.

In `tests/test_source_backup_contract.py`, change the included fixture path from:

```python
"watermark_auth.py"
```

to:

```python
"trace_app/watermark/auth.py"
```

- [ ] **Step 5: Run layout, release, secret, and backup contracts**

Run:

```powershell
python -m pytest tests/test_package_layout.py tests/test_release_builder.py tests/test_secret_hygiene.py tests/test_source_backup_contract.py tests/test_centos_deploy_contract.py -q
```

Expected: PASS, including release collection from the new package paths.

- [ ] **Step 6: Scan the entire maintained source tree for stale path references**

Run:

```powershell
rg -n --glob '!release/**' --glob '!docs/superpowers/**' "candidate_feature_index\.py|database_store\.py|password_security\.py|watermark_auth\.py|watermark_ecc\.py|from candidate_feature_index|from database_store|from password_security|from watermark_auth|from watermark_ecc" .
```

Expected: no matches.

- [ ] **Step 7: Commit the layout and packaging contract**

```powershell
git add tests/test_package_layout.py tools/build_centos_release.py tests/test_release_builder.py tests/test_secret_hygiene.py tests/test_source_backup_contract.py
git commit -m "refactor: enforce trace_app package boundaries"
```

### Task 5: Run Full Relevant Verification And Restart The Service

**Files:**
- Verify only; modify files only to fix a demonstrated regression within this refactor

- [ ] **Step 1: Run the complete affected backend suite**

Run:

```powershell
python -m pytest tests/test_package_layout.py tests/test_application_structure.py tests/test_candidate_feature_index.py tests/test_database_store.py tests/test_false_positive_gate.py tests/test_json_mysql_migration.py tests/test_password_security.py tests/test_watermark_auth.py tests/test_watermark_ecc.py tests/test_watermark_v4_api.py tests/v4/test_relational_only_schema.py tests/test_release_builder.py tests/test_secret_hygiene.py tests/test_source_backup_contract.py tests/test_centos_deploy_contract.py -q
```

Expected: PASS with no import, response, migration, or packaging regressions.

- [ ] **Step 2: Verify import and whitespace hygiene**

Run:

```powershell
python -c "from trace_app.auth.password_security import hash_password; from trace_app.database.store import DatabaseStore; from trace_app.imaging.candidate_feature_index import extract_feature_descriptors; from trace_app.watermark.auth import auth_code_from_trace; from trace_app.watermark.ecc import encode_codeword; print('trace_app imports: ok')"
git diff --check
git status --short
```

Expected: the import command prints `trace_app imports: ok`, `git diff --check`
reports no errors, and status contains only intentional commits plus pre-existing
untracked runtime/release artifacts.

- [ ] **Step 3: Stop the old worktree service on port 8000**

Resolve the exact listener before stopping it:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" |
    Select-Object ProcessId, CommandLine
  Stop-Process -Id $listener.OwningProcess
}
```

Expected: only the verified Uvicorn process from this worktree is stopped.

- [ ] **Step 4: Start the refactored application with the existing isolated local database**

Set the same local runtime values used by the current development service, keeping
secrets out of command output, and start a hidden background process:

```powershell
$worktree = (Resolve-Path '.').Path
$commonGit = (Resolve-Path (git rev-parse --git-common-dir)).Path
$mainRoot = Split-Path $commonGit
$configured = @{}
foreach ($line in Get-Content (Join-Path $mainRoot '.env')) {
    if ($line -match '^(ADMIN_USER|ADMIN_PASS|WATERMARK_AUTH_KEY)=(.*)$') {
        $configured[$matches[1]] = $matches[2]
    }
}
$keys = @(
    'DB_URL', 'ADMIN_USER', 'ADMIN_PASS', 'WATERMARK_AUTH_KEY',
    'ENVIRONMENT', 'V4_MODEL_MANIFEST_PATH'
)
$previous = @{}
foreach ($key in $keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$env:DB_URL = "sqlite+pysqlite:///$($worktree.Replace('\', '/'))/.vite/local-service.db"
$env:ADMIN_USER = $configured['ADMIN_USER']
$env:ADMIN_PASS = $configured['ADMIN_PASS']
$env:WATERMARK_AUTH_KEY = $configured['WATERMARK_AUTH_KEY']
$env:ENVIRONMENT = 'development'
$env:V4_MODEL_MANIFEST_PATH = (Join-Path $mainRoot 'models/v4-manifest.json')
try {
    $process = Start-Process -FilePath (Get-Command python).Source `
        -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $worktree `
        -RedirectStandardOutput (Join-Path $worktree '.vite/trace-service-8000.stdout.log') `
        -RedirectStandardError (Join-Path $worktree '.vite/trace-service-8000.stderr.log') `
        -WindowStyle Hidden -PassThru
    Write-Output "PID=$($process.Id)"
} finally {
    foreach ($key in $keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
```

Expected: a PID is printed, while administrator credentials and the watermark key
are never written to command output or logs.

- [ ] **Step 5: Verify the live service**

Run:

```powershell
curl.exe -sS --max-time 5 -o NUL -w "ROOT=%{http_code}" http://127.0.0.1:8000/
curl.exe -sS --max-time 5 -o NUL -w "CAPABILITIES=%{http_code}" http://127.0.0.1:8000/api/v4/capabilities
```

Expected: `ROOT=200`; the unauthenticated capabilities endpoint returns
`CAPABILITIES=401`.

- [ ] **Step 6: Record final branch history and status**

Run:

```powershell
git log --oneline --decorate 7e545c7..HEAD
git status --short --branch
```

Expected: the design commit and four focused refactor commits appear after
`7e545c7`; only pre-existing untracked artifacts remain.

## Self-Review Record

- Spec coverage: Tasks 1-3 implement every module mapping; Task 4 enforces the root,
  release, and no-shim contracts; Task 5 covers startup and behavioral verification.
- Placeholder scan: the plan contains no deferred implementation markers or unnamed
  files; each edit and command names its exact target.
- Type consistency: all consumers use the same concrete symbols from the new absolute
  package paths; no re-export API is introduced.
- Scope control: database contents, API behavior, algorithms, frontend code, models,
  environment files, and untracked release artifacts remain outside this refactor.
