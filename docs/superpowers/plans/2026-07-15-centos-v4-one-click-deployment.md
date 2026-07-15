# CentOS V4 One-Click Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing CentOS/systemd deployment flow so one command preserves current data and configuration, prepares a stable V4 authentication key, validates the existing database, deploys V4, verifies HTTP health, and presents V4 evidence accurately in the frontend.

**Architecture:** Keep `deploy.sh` as the only public deployment entry point. Move deterministic `.env` mutation into a testable standard-library Python helper, keep the existing database and administrator defaults, and validate deployment shell behavior with contract tests because the development host is Windows. Frontend changes remain inside the existing single-file UI and conditionally render V4 evidence without changing API requests.

**Tech Stack:** Bash, CentOS/RHEL systemd, Python 3 standard library, pytest, FastAPI/Uvicorn, MySQL/MariaDB client, HTML/JavaScript.

**Repository note:** `D:\WWW\python\trace` is not a Git repository. Replace commit steps with documented verification checkpoints; do not initialize Git as part of this work.

---

### Task 1: Atomic V4 Environment Preparation

**Files:**
- Create: `tools/prepare_deployment_env.py`
- Create: `tests/test_prepare_deployment_env.py`

- [ ] **Step 1: Write failing tests for generation, preservation, duplicate collapse, and DB validation**

Create tests that exercise the helper as real code:

```python
import base64
from pathlib import Path

import pytest

from tools.prepare_deployment_env import prepare_environment


def _values(path: Path, name: str) -> list[str]:
    prefix = f"{name}="
    return [line[len(prefix):] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]


def test_prepare_environment_generates_v4_key_and_preserves_unrelated_lines(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# keep\nDB_ENABLED=true\nDB_URL=mysql+pymysql://REMOVED:secret@127.0.0.1/trace\nAPP_NAME=WatermarkSystem\nWATERMARK_AUTH_KEY=\nROBUST_WATERMARK_VERSION=1\n", encoding="utf-8")

    result = prepare_environment(path)

    key = _values(path, "WATERMARK_AUTH_KEY")
    assert result["generated"] is True
    assert len(key) == 1
    assert len(base64.b64decode(key[0], validate=True)) == 48
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]
    assert "# keep" in path.read_text(encoding="utf-8")
    assert "APP_NAME=WatermarkSystem" in path.read_text(encoding="utf-8")
    assert key[0] not in str(result)


def test_prepare_environment_preserves_valid_key_on_repeated_run(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    existing = base64.b64encode(b"x" * 48).decode("ascii")
    path.write_text(f"DB_ENABLED=false\nWATERMARK_AUTH_KEY={existing}\nROBUST_WATERMARK_VERSION=1\n", encoding="utf-8")

    first = prepare_environment(path)
    second = prepare_environment(path)

    assert first["generated"] is False
    assert second["generated"] is False
    assert _values(path, "WATERMARK_AUTH_KEY") == [existing]
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]


def test_prepare_environment_replaces_duplicate_key_entries(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("DB_ENABLED=false\nWATERMARK_AUTH_KEY=first-value-that-is-long-enough-123456\nWATERMARK_AUTH_KEY=second-value-that-is-long-enough-12345\nROBUST_WATERMARK_VERSION=3\nROBUST_WATERMARK_VERSION=1\n", encoding="utf-8")

    result = prepare_environment(path)

    assert result["generated"] is True
    assert len(_values(path, "WATERMARK_AUTH_KEY")) == 1
    assert _values(path, "ROBUST_WATERMARK_VERSION") == ["4"]


def test_prepare_environment_requires_db_url_when_database_enabled(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("DB_ENABLED=true\nDB_URL=\n", encoding="utf-8")

    with pytest.raises(ValueError, match="DB_URL"):
        prepare_environment(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_prepare_deployment_env.py
```

Expected: collection fails because `tools.prepare_deployment_env` does not exist.

- [ ] **Step 3: Implement the minimal atomic helper**

Implement `prepare_environment(path: Path) -> dict[str, object]` with these exact behaviors:

```python
import argparse
import base64
import os
import secrets
import tempfile
from pathlib import Path

MANAGED_KEYS = ("ROBUST_WATERMARK_VERSION", "WATERMARK_AUTH_KEY")


def _assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    name, value = stripped.split("=", 1)
    return name.strip(), value


def prepare_environment(path: Path) -> dict[str, object]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    assignments: dict[str, list[str]] = {}
    retained: list[str] = []
    for line in lines:
        parsed = _assignment(line)
        if parsed is not None:
            assignments.setdefault(parsed[0], []).append(parsed[1])
        if parsed is None or parsed[0] not in MANAGED_KEYS:
            retained.append(line)

    db_enabled = (assignments.get("DB_ENABLED", ["true"])[-1].strip().lower() not in {"0", "false", "no", "off"})
    db_url = assignments.get("DB_URL", [""])[-1].strip()
    if db_enabled and not db_url:
        raise ValueError("DB_URL is required when DB_ENABLED=true")

    existing = assignments.get("WATERMARK_AUTH_KEY", [])
    valid = len(existing) == 1 and len(existing[0].encode("utf-8")) >= 32
    key = existing[0] if valid else base64.b64encode(secrets.token_bytes(48)).decode("ascii")
    output = retained + ["ROBUST_WATERMARK_VERSION=4", f"WATERMARK_AUTH_KEY={key}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(output).rstrip("\n") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if os.name != "nt":
        os.chmod(path, 0o600)
    return {"generated": not valid, "utf8_bytes": len(key.encode("utf-8")), "entries": 1}
```

Add a CLI accepting `--env-file`, printing only `generated`, `utf8_bytes`, and `entries`; never print `key`.

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    try:
        result = prepare_environment(args.env_file)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"environment preparation failed: {exc}\n")
    print(
        "deployment environment prepared: "
        f"generated={result['generated']} "
        f"utf8_bytes={result['utf8_bytes']} entries={result['entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests/test_prepare_deployment_env.py
```

Expected: `4 passed`.

- [ ] **Step 5: Record checkpoint**

Record the RED/GREEN output in the session; no commit is possible because the workspace has no Git metadata.

### Task 2: Idempotent CentOS Deployment Contract

**Files:**
- Create: `tests/test_centos_deploy_contract.py`
- Modify: `deploy.sh`

- [ ] **Step 1: Write failing deployment contract tests**

Read `deploy.sh` as text and assert the externally important contract:

```python
from pathlib import Path

SCRIPT = Path("deploy.sh").read_text(encoding="utf-8")


def test_install_service_prepares_v4_and_backs_up_before_restart() -> None:
    install_body = SCRIPT[SCRIPT.index("install_service()") : SCRIPT.index("run_server()")]
    assert "prepare_deployment_env.py" in SCRIPT
    assert "backup_runtime_state" in SCRIPT
    assert install_body.index("backup_runtime_state") < install_body.index("prepare_deployment_env.py")
    assert install_body.index("prepare_deployment_env.py") < install_body.index("check_database")
    assert install_body.index("check_database") < install_body.index('systemctl restart "${SERVICE_NAME}"')


def test_install_service_uses_existing_database_without_server_mutation() -> None:
    install_body = SCRIPT[SCRIPT.index("install_service()") : SCRIPT.index("run_server()")]
    assert "check_database" in install_body
    assert "CREATE DATABASE" not in SCRIPT
    assert "CREATE USER" not in SCRIPT
    assert "ALTER USER" not in SCRIPT


def test_install_service_polls_local_http_health() -> None:
    assert "wait_for_http_health" in SCRIPT
    assert "http://127.0.0.1:${PORT}/" in SCRIPT
    assert "journalctl" in SCRIPT


def test_existing_database_and_admin_defaults_are_retained() -> None:
    assert 'DB_PASS=""' in SCRIPT
    assert "ADMIN_PASS=" in ENV_EXAMPLE
```

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_centos_deploy_contract.py
```

Expected: failures for missing backup, V4 preparation, health polling, and database mutation removal.

- [ ] **Step 3: Refactor `deploy.sh` around bounded functions**

Keep parameters and existing credentials, but replace database initialization with:

```bash
check_database() {
  load_env_config
  if ! command -v mysql >/dev/null 2>&1; then
    echo "mysql client is required to validate the existing database" >&2
    return 1
  fi
  MYSQL_PWD="${DB_PASS}" mysql -u"${DB_USER}" -h"${DB_HOST}" -P"${DB_PORT}" "${DB_NAME}" -e "SELECT 1" >/dev/null
}
```

Add backup before any environment mutation:

```bash
backup_runtime_state() {
  local stamp archive
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${ROOT}/backups/deploy"
  archive="${ROOT}/backups/deploy/pre-deploy-${stamp}.tar.gz"
  local items=()
  [ -f "${ROOT}/.env" ] && items+=(".env")
  [ -d "${ROOT}/data" ] && items+=("data")
  [ -d "${ROOT}/uploads" ] && items+=("uploads")
  if [ "${#items[@]}" -gt 0 ]; then
    tar -czf "${archive}" "${items[@]}"
    LAST_BACKUP="${archive}"
  else
    LAST_BACKUP="none (fresh deployment)"
  fi
}
```

Add bounded health polling:

```bash
wait_for_http_health() {
  local attempt
  for attempt in $(seq 1 60); do
    if "${ROOT}/.venv/bin/python" -c "import urllib.request; response=urllib.request.urlopen('http://127.0.0.1:${PORT}/', timeout=2); raise SystemExit(0 if response.status == 200 else 1)" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  systemctl --no-pager status "${SERVICE_NAME}" || true
  journalctl -u "${SERVICE_NAME}" -n 80 --no-pager || true
  echo "Deployment health check failed. Backup: ${LAST_BACKUP}" >&2
  return 1
}
```

Make `install_service` execute in this order: root check, packages, backup, create `.env` only if absent, environment helper, database check, venv/dependencies, directories, systemd unit, restart, active check, HTTP health check. Remove all SQL DDL and MySQL root-password flows. Keep `run` and `status`; replace `init-db` with `check-db`.

Use this concrete orchestration, extracting the current systemd heredoc into `write_systemd_service` without changing its service name, user, working directory, restart policy, host, or port:

```bash
write_env() {
  if [ ! -f "${ROOT}/.env" ]; then
    cp "${ROOT}/.env.example" "${ROOT}/.env"
  fi
}

install_service() {
  need_root install-service
  install_system_packages
  backup_runtime_state
  write_env
  python3 "${ROOT}/tools/prepare_deployment_env.py" --env-file "${ROOT}/.env"
  check_database
  install_python_environment
  write_systemd_service
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"
  if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
    systemctl --no-pager status "${SERVICE_NAME}" || true
    journalctl -u "${SERVICE_NAME}" -n 80 --no-pager || true
    echo "Service failed to start. Backup: ${LAST_BACKUP}" >&2
    return 1
  fi
  wait_for_http_health
  if systemctl is-active --quiet firewalld; then
    firewall-cmd --add-port="${PORT}/tcp" --permanent || true
    firewall-cmd --reload || true
  fi
  echo "Installed ${SERVICE_NAME}."
  echo "URL: http://<server-ip>:${PORT}"
  echo "Backup: ${LAST_BACKUP}"
  echo "Status: systemctl status ${SERVICE_NAME}"
}
```

- [ ] **Step 4: Run contract tests and syntax checks**

Run:

```powershell
python -m pytest -q tests/test_centos_deploy_contract.py tests/test_prepare_deployment_env.py
```

Expected: all tests pass. If `bash` is available, also run `bash -n deploy.sh`; otherwise state that CentOS execution remains an environment verification item.

- [ ] **Step 5: Record checkpoint**

Record contract and syntax results; no commit is possible in this workspace.

### Task 3: V4 Deployment Defaults And Operator Documentation

**Files:**
- Modify: `.env.example`
- Modify: `README_DEPLOY.md`
- Modify: `docs/commercial_watermark_upgrade_plan.md`
- Test: `tests/test_centos_deploy_contract.py`

- [ ] **Step 1: Extend failing tests for defaults and documentation**

Assert `.env.example` contains exactly one `ROBUST_WATERMARK_VERSION=4`, one blank `WATERMARK_AUTH_KEY=`, and retains existing `ADMIN_PASS` and `DB_URL`. Assert the README uses `sudo ./deploy.sh install-service`, documents preservation/backup, existing DB validation, generated key preservation, V4, health checks, and `check-db` without claiming MySQL installation.

- [ ] **Step 2: Run the new tests and verify RED**

Run `python -m pytest -q tests/test_centos_deploy_contract.py` and confirm the old V1/default documentation fails.

- [ ] **Step 3: Update defaults and deployment guide**

Change only these environment defaults:

```dotenv
ROBUST_WATERMARK_STRENGTH=0.74
ROBUST_WATERMARK_VERSION=4
WATERMARK_AUTH_KEY=
```

Retain current `ADMIN_USER`, `ADMIN_PASS`, and `DB_URL`. Rewrite deployment instructions around the existing DB and one command. Add explicit backup location, idempotent key behavior, health deadline, diagnostics, and recovery commands. Add a dated deployment-readiness entry to the commercial plan without marking real-route evidence complete.

- [ ] **Step 4: Verify documentation contract**

Run `python -m pytest -q tests/test_centos_deploy_contract.py` and expect all deployment documentation tests to pass.

### Task 4: Frontend V4 Evidence Rendering

**Files:**
- Modify: `index.html:1073-1166`
- Modify: `tests/test_frontend_watermark_version.py`

- [ ] **Step 1: Write failing frontend contract tests**

Add assertions that the HTML contains dedicated `v4GenerationEvidence` and `v4RecoveryEvidence` helpers, checks `robust_watermark_version === 4`, renders `authenticated_tiles`, `phase_count`, `corrected_symbols`, `bit_errors`, `sync_confidence`, and `elapsed_ms`, and does not route V4 through the legacy `频域评分` row.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest -q tests/test_frontend_watermark_version.py
```

Expected: failures for the missing V4 evidence helpers.

- [ ] **Step 3: Implement conditional escaped rendering**

Add helpers that return HTML rows only for V4. Generation should show `认证 DCT + FFT 同步`; extraction should read `r.code_recovery || {}` and show bounded numeric values. Keep the existing legacy frequency-score row for non-V4 records. Use `esc(...)` for strings and `Number(...)` plus finite fallbacks for numeric values.

- [ ] **Step 4: Run frontend and API regressions**

Run:

```powershell
python -m pytest -q tests/test_frontend_watermark_version.py tests/test_frontend_control_contrast.py tests/test_watermark_v4_api.py
```

Expected: all tests pass.

### Task 5: Final Verification And Deployment Handoff

**Files:**
- Verify all files above
- Update: `docs/commercial/phase-0b-v4-results.md`

- [ ] **Step 1: Run focused deployment and V4 verification**

```powershell
python -m pytest -q tests/test_prepare_deployment_env.py tests/test_centos_deploy_contract.py tests/test_frontend_watermark_version.py tests/test_watermark_v4_api.py tests/test_watermark_v4_detector.py
python -m compileall -q tools watermark_v4 main.py
```

Expected: zero failures and compilation exit code 0.

- [ ] **Step 2: Run full regression**

```powershell
python -m pytest -q
```

Expected: zero failures; report the exact pass/skip/warning counts rather than copying historical counts.

- [ ] **Step 3: Perform a non-secret environment smoke test**

Copy `.env.example` to a temporary directory, set `DB_ENABLED=false`, run the helper twice, and verify the first run reports `generated=True`, the second `generated=False`, V4 appears once, the key appears once, and no key value is printed.

- [ ] **Step 4: Verify service and documentation state**

Confirm the local V4 service still returns HTTP 200 after any restart. Update Phase 0B results with the one-click deployment status while keeping `REAL_ROUTE_PENDING` unchanged.

- [ ] **Step 5: Hand off CentOS command**

Provide:

```bash
chmod +x deploy.sh
sudo ./deploy.sh install-service
```

Also provide `systemctl status trace-system`, `journalctl -u trace-system -f`, backup location, and `sudo ./deploy.sh check-db`. State clearly that actual CentOS execution must be performed on the target server even though contract, syntax, helper, and application tests passed locally.
