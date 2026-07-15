# Database Migration and Secret Purge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all runtime JSON data to MySQL, make user management database-backed with salted password hashes, remove tracked credentials, and purge leaked data and secrets from Git history.

**Architecture:** Add focused password and SQLAlchemy storage modules, then make `main.py` use the database store as its only persistence layer. A reusable, idempotent migration CLI validates the five private JSON inputs, imports and verifies them in one transaction, and removes inputs only after a successful commit. Repository and release sanitization precedes a final `git-filter-repo` rewrite and object scan.

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy Core, MySQL/MariaDB with PyMySQL, SQLite for isolated tests, pytest, git-filter-repo.

---

## File Structure

- Create `password_security.py`: versioned scrypt password hashing and verification only.
- Create `database_store.py`: schema definitions and database persistence operations only.
- Create `tools/migrate_json_to_mysql.py`: JSON validation, transactional import, verification, backup, and cleanup CLI.
- Create `tests/test_password_security.py`: password hashing contract.
- Create `tests/test_database_store.py`: database schema and CRUD contract using SQLite.
- Create `tests/test_json_mysql_migration.py`: migration validation, idempotency, rollback, and cleanup contract.
- Create `tests/test_secret_hygiene.py`: tracked source and release credential exclusions.
- Modify `main.py`: required configuration, database-only runtime adapters, hashed login, and user endpoints.
- Modify `tests/test_watermark_v4_api.py`: isolated SQLite database fixture and user API coverage.
- Modify `tests/test_centos_deploy_contract.py`: require secret-free deployment assets and migration command.
- Modify `deploy.sh`: remove credential defaults and add an explicit `migrate-data` command.
- Modify `.env.example`: retain variable names with empty secret values.
- Modify `.gitignore`: ignore the five runtime JSON paths and migration backups.
- Modify `README_DEPLOY.md`: document private `.env` setup and one-time migration.
- Remove `data/*.json`: remove the five migrated runtime datasets from the current tree.
- Remove `release/trace-v4-centos-20260715.zip` and its checksum: remove leaked binary artifacts.
- Synchronize sanitized source files under `release/trace-v4-centos-20260715/` before rebuilding the release.

### Task 1: Versioned Password Hashing

**Files:**
- Create: `password_security.py`
- Create: `tests/test_password_security.py`

- [ ] **Step 1: Write failing password tests**

```python
from password_security import hash_password, verify_password


def test_hash_is_salted_versioned_and_not_plaintext() -> None:
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first.startswith("scrypt$v1$")
    assert first != second
    assert "same-password" not in first


def test_verify_accepts_only_the_original_password() -> None:
    encoded = hash_password("correct-password")
    assert verify_password("correct-password", encoded) is True
    assert verify_password("wrong-password", encoded) is False
    assert verify_password("correct-password", "invalid") is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_password_security.py -q`

Expected: FAIL because `password_security` does not exist.

- [ ] **Step 3: Implement the minimal hashing module**

```python
import base64
import hashlib
import hmac
import secrets

N = 2**14
R = 8
P = 1
DKLEN = 32


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=DKLEN)
    return "$".join(("scrypt", "v1", str(N), str(R), str(P),
                     base64.b64encode(salt).decode(), base64.b64encode(derived).decode()))


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, version, n, r, p, salt_text, digest_text = encoded.split("$")
        if (algorithm, version) != ("scrypt", "v1"):
            return False
        salt = base64.b64decode(salt_text, validate=True)
        expected = base64.b64decode(digest_text, validate=True)
        actual = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest tests/test_password_security.py -q`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add password_security.py tests/test_password_security.py
git commit -m "feat: add versioned password hashing"
```

### Task 2: Database Store and Relational Users

**Files:**
- Create: `database_store.py`
- Create: `tests/test_database_store.py`

- [ ] **Step 1: Write failing SQLite-backed store tests**

```python
from sqlalchemy import create_engine
from database_store import DatabaseStore


def test_user_crud_stores_only_hashes() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.create_user("alice", "secret", "operator")
    assert store.authenticate("alice", "secret") == "operator"
    assert store.authenticate("alice", "wrong") is None
    assert store.list_users() == {"alice": {"role": "operator"}}
    store.update_user_role("alice", "viewer")
    assert store.list_users()["alice"]["role"] == "viewer"
    store.delete_user("alice")
    assert store.list_users() == {}


def test_records_and_json_documents_round_trip() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.replace_records([{"id": "b"}, {"id": "a"}])
    store.set_json("roles", {"roles": {"admin": {"menus": []}}})
    assert [item["id"] for item in store.read_records()] == ["b", "a"]
    assert store.get_json("roles", {})["roles"]["admin"]["menus"] == []
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_database_store.py -q`

Expected: FAIL because `database_store` does not exist.

- [ ] **Step 3: Implement SQLAlchemy Core schema and store methods**

Define `image_records`, `app_json_store`, and `app_users` with SQLAlchemy `MetaData`/`Table`. Implement `create_schema`, `replace_records`, `read_records`, `get_json`, `set_json`, `create_user`, `upsert_user_hash`, `list_users`, `update_user_role`, `delete_user`, `authenticate`, `clear_all`, and transaction-aware import helpers. `create_user` must call `hash_password`; `authenticate` must call `verify_password`; no method may return `password_hash`.

Use portable SQLAlchemy `select`, `insert`, `update`, and `delete` statements so the same code runs on SQLite tests and MySQL production. Implement upserts as select-then-insert/update inside the caller's transaction instead of dialect-specific SQL.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest tests/test_database_store.py tests/test_password_security.py -q`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add database_store.py tests/test_database_store.py
git commit -m "feat: add database persistence store"
```

### Task 3: Make FastAPI Runtime Database-Only

**Files:**
- Modify: `main.py:50-490`
- Modify: `main.py:3689-3755`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Add failing runtime and user API tests**

Update the autouse fixture to create a temporary SQLite file, assign a `DatabaseStore` to `main.db_store`, call `create_schema`, and seed roles plus the administrator. Add tests that POST `/api/users`, verify the database row contains a scrypt hash rather than plaintext, log in with the created password, update its role, and delete it. Add a test that replaces `main.db_store` with `None` and asserts persistence calls raise HTTP 503 without creating any JSON file.

- [ ] **Step 2: Run the focused API tests and verify RED**

Run: `pytest tests/test_watermark_v4_api.py -k "user or database" -q`

Expected: FAIL because runtime user operations still use JSON dictionaries and plaintext comparison.

- [ ] **Step 3: Replace JSON fallback adapters in `main.py`**

Implement these runtime rules:

```python
DB_URL = os.getenv("DB_URL", "").strip()
ADMIN_USER = os.getenv("ADMIN_USER", "").strip()
ADMIN_PASS = os.getenv("ADMIN_PASS", "")


def require_store() -> DatabaseStore:
    if db_store is None:
        raise HTTPException(status_code=503, detail="数据库不可用")
    return db_store
```

Create the engine and schema only when a non-empty `DB_URL` is present; production startup must fail with the missing variable name or connection error. Replace record, role, and statistics functions with calls to `DatabaseStore`. Replace user dictionary operations with direct store CRUD. Seed `ADMIN_USER` through `create_user` only when absent, using `ADMIN_PASS`, and remove every built-in credential and plaintext login branch. Add `DELETE /api/users/{username}` backed by `delete_user`.

Remove `RECORD_FILE`, `DETECTION_STATS_FILE`, `WATERMARK_STATS_FILE`, `ROLE_FILE`, `USER_FILE`, `read_json_file`, `write_json_file`, `migrate_json_to_database`, and all runtime JSON creation. Keep `DATA_DIR` only for feature-index files.

- [ ] **Step 4: Run API and existing data-loading tests**

Run: `pytest tests/test_watermark_v4_api.py tests/test_homepage_data_loading.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add main.py tests/test_watermark_v4_api.py
git commit -m "feat: use database for runtime users and data"
```

### Task 4: Idempotent Server Migration CLI

**Files:**
- Create: `tools/migrate_json_to_mysql.py`
- Create: `tests/test_json_mysql_migration.py`

- [ ] **Step 1: Write failing migration tests**

Build five representative JSON files under `tmp_path`. Test that `load_source_data` rejects a missing or malformed file before engine creation. Run `migrate` against a temporary SQLite database and assert exact records, statistics, roles, user authentication, and removal of source JSON only after verification. Run it twice using restored identical inputs and assert no duplicates. Inject a verification failure and assert the database transaction rolls back and source files remain.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_json_mysql_migration.py -q`

Expected: FAIL because the migration module does not exist.

- [ ] **Step 3: Implement validation and transactional import**

Expose:

```python
def load_source_data(data_dir: Path) -> SourceData: ...
def migrate(engine: Engine, source: SourceData, data_dir: Path, backup_dir: Path) -> MigrationResult: ...
def main(argv: list[str] | None = None) -> int: ...
```

Validate the five top-level shapes and unique image IDs. In one `engine.begin()` block, create schema, replace image rows, set the three JSON documents, and upsert users with freshly generated hashes. Verify all source counts and keys through the same connection before leaving the transaction. After commit, copy the five inputs into `backup_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")`, verify copied bytes, and unlink only the original five files.

CLI arguments: `--env-file`, `--data-dir`, and `--backup-dir`. Load `DB_URL` from the selected `.env` without printing it. Print only dataset counts, backup path, and success/failure status.

- [ ] **Step 4: Run migration tests and verify GREEN**

Run: `pytest tests/test_json_mysql_migration.py tests/test_database_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add tools/migrate_json_to_mysql.py tests/test_json_mysql_migration.py
git commit -m "feat: add transactional JSON database migration"
```

### Task 5: Secret-Free Configuration and Deployment

**Files:**
- Modify: `.env.example`
- Modify: `deploy.sh`
- Modify: `README_DEPLOY.md`
- Modify: `tests/test_centos_deploy_contract.py`
- Create: `tests/test_secret_hygiene.py`

- [ ] **Step 1: Replace old credential assertions with failing hygiene tests**

Assert `.env.example` contains `ADMIN_PASS=` and `DB_URL=` with empty values, `deploy.sh` has no `DB_PASS` fallback, and `main.py` has no credential-bearing URL default. Read tracked text files and fail on the three known leaked values. Assert the release ZIP, when present, excludes `.env`, the five runtime JSON files, and known leaked values.

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `pytest tests/test_centos_deploy_contract.py tests/test_secret_hygiene.py -q`

Expected: FAIL on current tracked credentials.

- [ ] **Step 3: Sanitize configuration and add deployment migration command**

Set secret fields in `.env.example` to empty. Remove `DB_PASS` defaults from `deploy.sh`; parse only `.env`. Make `write_env` stop after creating `.env` and instruct the operator to populate required values. Add:

```sh
migrate_data() {
  select_python
  "${PYTHON_BIN}" "${ROOT}/tools/migrate_json_to_mysql.py" \
    --env-file "${ROOT}/.env" \
    --data-dir "${ROOT}/data" \
    --backup-dir "${ROOT}/../trace-private-migration-backups"
}
```

Expose it as `./deploy.sh migrate-data`. Document that migration runs before service restart and that real values stay only in `.env`.

- [ ] **Step 4: Run the deployment and hygiene tests**

Run: `pytest tests/test_centos_deploy_contract.py tests/test_prepare_deployment_env.py tests/test_secret_hygiene.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add .env.example deploy.sh README_DEPLOY.md tests/test_centos_deploy_contract.py tests/test_secret_hygiene.py
git commit -m "security: remove credential defaults"
```

### Task 6: Remove Runtime JSON and Leaked Release Artifacts

**Files:**
- Modify: `.gitignore`
- Remove: `data/detection_stats.json`
- Remove: `data/images.json`
- Remove: `data/roles.json`
- Remove: `data/users.json`
- Remove: `data/watermark_stats.json`
- Remove: `release/trace-v4-centos-20260715.zip`
- Remove: `release/trace-v4-centos-20260715.zip.sha256`
- Synchronize: `release/trace-v4-centos-20260715/`

- [ ] **Step 1: Add failing ignore and release-content assertions**

Extend `tests/test_secret_hygiene.py` to assert all five runtime JSON paths are ignored and absent from `git ls-files`, while `data/feature_index/` remains trackable. Assert every release source copy matches its sanitized root counterpart. ZIP-content assertions must skip when the old archive is intentionally absent and run once Task 7 rebuilds it.

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_secret_hygiene.py -q`

Expected: FAIL because runtime JSON files and the old release are tracked.

- [ ] **Step 3: Remove JSON inputs and leaked release artifacts**

Before removal, copy the five JSON files to `D:\WWW\python\trace-private-migration-input-20260715\` and verify byte-for-byte equality. Add exact JSON paths and `trace-private-migration-backups/` to `.gitignore`. Remove the tracked JSON files and the leaked ZIP/checksum. Synchronize sanitized deployable files to `release/trace-v4-centos-20260715/`. Do not rebuild the ZIP until after Task 7 removes the old binary object from history.

- [ ] **Step 4: Verify release and run the relevant suite**

Run: `pytest tests/test_secret_hygiene.py tests/test_centos_deploy_contract.py tests/test_watermark_v4_api.py tests/test_json_mysql_migration.py -q`

Expected: PASS with no credential values printed; ZIP assertions are skipped because the old archive is absent.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore data release tests/test_secret_hygiene.py
git commit -m "security: remove runtime data from repository"
```

### Task 7: Full Verification, Git History Purge, and Sanitized Release

**Files:**
- Rewrite: Git refs and objects; no source API changes in this task.

- [ ] **Step 1: Run fresh verification before rewriting history**

Run:

```powershell
pytest tests/test_password_security.py tests/test_database_store.py tests/test_json_mysql_migration.py tests/test_secret_hygiene.py tests/test_centos_deploy_contract.py tests/test_prepare_deployment_env.py tests/test_homepage_data_loading.py tests/test_watermark_v4_api.py -q
git diff --check
git status --short
```

Expected: all tests pass, no whitespace errors, clean worktree.

- [ ] **Step 2: Create an external safety mirror**

Run from the repository parent:

```powershell
git clone --mirror D:\WWW\python\trace D:\WWW\python\trace-before-secret-purge.git
git -C D:\WWW\python\trace remote get-url origin | Set-Content D:\WWW\python\trace-origin-url.txt
git -C D:\WWW\python\trace rev-parse origin/master | Set-Content D:\WWW\python\trace-origin-master.txt
```

Expected: a mirror plus private remote metadata outside the worktree; none of these files are pushed or shared.

- [ ] **Step 3: Rewrite all refs**

Ensure `git filter-repo` is installed, then remove the five JSON paths plus the old binary release artifacts from all history. Use `--replace-text` with `D:\WWW\python\trace-secret-replacements.txt`, a private file outside the repository containing one `literal:old-value==>REMOVED` line for each of the three previously inventoried password values. Keep this file only until Step 4 finishes the object scan.

```powershell
git filter-repo --force `
  --path data/detection_stats.json `
  --path data/images.json `
  --path data/roles.json `
  --path data/users.json `
  --path data/watermark_stats.json `
  --path release/trace-v4-centos-20260715.zip `
  --path release/trace-v4-centos-20260715.zip.sha256 `
  --invert-paths `
  --replace-text D:\WWW\python\trace-secret-replacements.txt
```

Because `git-filter-repo` removes `origin` by default, restore it without printing a possibly credential-bearing URL:

```powershell
$originUrl = (Get-Content D:\WWW\python\trace-origin-url.txt -Raw).Trim()
git remote add origin $originUrl
```

- [ ] **Step 4: Scan reachable history and current text artifacts**

Use the same private replacement file to scan every reachable revision without placing password values in the plan or shell history:

```powershell
$secrets = Get-Content D:\WWW\python\trace-secret-replacements.txt | ForEach-Object {
  ($_ -replace '^literal:', '') -split '==>', 2 | Select-Object -First 1
}
$revisions = git rev-list --all
foreach ($secret in $secrets) {
  if (git log --all --format='%H' -S $secret) { throw 'Secret remains in commit history' }
  foreach ($revision in $revisions) {
    if (git grep -I -l -F $secret $revision -- 2>$null) { throw 'Secret remains in reachable blob' }
  }
}
$removedPaths = @(
  'data/detection_stats.json', 'data/images.json', 'data/roles.json',
  'data/users.json', 'data/watermark_stats.json',
  'release/trace-v4-centos-20260715.zip',
  'release/trace-v4-centos-20260715.zip.sha256'
)
foreach ($path in $removedPaths) {
  if (git log --all --format='%H' -- $path) { throw "Removed path remains: $path" }
}
Remove-Item -LiteralPath D:\WWW\python\trace-secret-replacements.txt
```

Expected: no secret or removed path is found. The private replacement file is deleted only after the scan succeeds.

- [ ] **Step 5: Build and scan the sanitized release**

Build `release/trace-v4-centos-20260715.zip` only from `release/trace-v4-centos-20260715/`, regenerate its SHA-256 file, and run:

```powershell
pytest tests/test_secret_hygiene.py tests/test_centos_deploy_contract.py -q
git add release/trace-v4-centos-20260715.zip release/trace-v4-centos-20260715.zip.sha256
git commit -m "build: regenerate sanitized CentOS release"
```

Expected: release-content scanning passes and the new archive is committed only after the old archive object has been removed from rewritten history.

- [ ] **Step 6: Force-push the sanitized history**

```powershell
$oldRemoteMaster = (Get-Content D:\WWW\python\trace-origin-master.txt -Raw).Trim()
git push --force-with-lease=refs/heads/master:$oldRemoteMaster origin master
git push --force origin --tags
Remove-Item -LiteralPath D:\WWW\python\trace-origin-url.txt,D:\WWW\python\trace-origin-master.txt
```

Expected: remote accepts rewritten `master` and affected tags. If lease protection rejects because remote moved, stop and inspect the new remote commits; do not override them blindly.

- [ ] **Step 7: Final verification**

Run:

```powershell
git fetch origin
git status --short --branch
pytest tests/test_password_security.py tests/test_database_store.py tests/test_json_mysql_migration.py tests/test_secret_hygiene.py tests/test_centos_deploy_contract.py tests/test_watermark_v4_api.py -q
```

Expected: local branch matches rewritten `origin/master`, worktree is clean, and all focused tests pass.
