from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "deploy.sh").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
README = (ROOT / "README_DEPLOY.md").read_text(encoding="utf-8")


def _install_body() -> str:
    return SCRIPT[SCRIPT.index("install_service()") : SCRIPT.index("run_server()")]


def test_install_service_prepares_v4_and_backs_up_before_restart() -> None:
    install_body = _install_body()

    assert "prepare_deployment_env.py" in SCRIPT
    assert "prepare_deployment_environment" in install_body
    assert "backup_runtime_state" in install_body
    assert install_body.index("backup_runtime_state") < install_body.index(
        "prepare_deployment_environment"
    )
    assert install_body.index("prepare_deployment_environment") < install_body.index(
        "check_database"
    )
    assert install_body.index("check_database") < install_body.index(
        'systemctl restart "${SERVICE_NAME}"'
    )


def test_install_service_uses_existing_database_without_server_mutation() -> None:
    install_body = _install_body()

    assert "check_database" in install_body
    assert "CREATE DATABASE" not in SCRIPT
    assert "CREATE USER" not in SCRIPT
    assert "ALTER USER" not in SCRIPT
    assert "MYSQL_ROOT_PASS" not in SCRIPT


def test_install_service_polls_local_http_health_and_reports_logs() -> None:
    assert "wait_for_http_health" in SCRIPT
    assert "http://127.0.0.1:${PORT}/" in SCRIPT
    assert "journalctl" in SCRIPT
    assert "systemctl is-active" in SCRIPT


def test_runtime_backup_preserves_environment_data_and_uploads() -> None:
    assert '".env"' in SCRIPT
    assert '"data"' in SCRIPT
    assert '"uploads"' in SCRIPT
    assert "backups/deploy" in SCRIPT
    assert "tar -czf" in SCRIPT


def test_existing_database_and_admin_defaults_are_retained() -> None:
    assert 'DB_PASS="${DB_PASS:-REMOVED_PASSWORD}"' in SCRIPT
    assert "ADMIN_PASS=REMOVED_PASSWORD" in ENV_EXAMPLE


def test_non_root_foreground_run_does_not_attempt_chown() -> None:
    assert '[ "$(id -u)" -eq 0 ]' in SCRIPT


def test_deployment_requires_and_reuses_python_310_or_newer() -> None:
    assert "select_python" in SCRIPT
    assert "sys.version_info >= (3, 10)" in SCRIPT
    assert '"${PYTHON_BIN}" "${ROOT}/tools/prepare_deployment_env.py"' in SCRIPT
    assert '"${PYTHON_BIN}" -m venv' in SCRIPT


def test_environment_example_selects_v4_and_keeps_existing_credentials() -> None:
    assert ENV_EXAMPLE.count("ROBUST_WATERMARK_VERSION=4") == 1
    assert ENV_EXAMPLE.count("WATERMARK_AUTH_KEY=") == 1
    assert "WATERMARK_AUTH_KEY=\n" in ENV_EXAMPLE
    assert "ROBUST_WATERMARK_STRENGTH=0.74" in ENV_EXAMPLE
    assert "ADMIN_USER=REMOVED_ADMIN_USER" in ENV_EXAMPLE
    assert "ADMIN_PASS=REMOVED_PASSWORD" in ENV_EXAMPLE
    assert "DB_URL=mysql+pymysql://REMOVED:REMOVED_PASSWORD@127.0.0.1:3306/trace" in ENV_EXAMPLE


def test_readme_documents_preserving_v4_one_click_deployment() -> None:
    for expected in (
        "sudo ./deploy.sh install-service",
        "现有 MySQL",
        "不会建库",
        "ROBUST_WATERMARK_VERSION=4",
        "WATERMARK_AUTH_KEY",
        "保持不变",
        "backups/deploy",
        "HTTP 200",
        "./deploy.sh check-db",
        "data/",
        "uploads/",
    ):
        assert expected in README
    assert "MYSQL_ROOT_PASS" not in README
