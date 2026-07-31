import zipfile
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "deploy.sh").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
README = (ROOT / "README_DEPLOY.md").read_text(encoding="utf-8")
RELEASE_ROOT = max(
    path
    for path in (ROOT / "release").glob("trace-v4-centos-????????-??????")
    if path.is_dir()
)
RELEASE_ARCHIVE = RELEASE_ROOT.with_suffix(".zip")


def _install_body() -> str:
    return SCRIPT[SCRIPT.index("install_service()") : SCRIPT.index("run_server()")]


def _python_install_body() -> str:
    return SCRIPT[
        SCRIPT.index("install_python_environment()") : SCRIPT.index(
            "write_systemd_service()"
        )
    ]


def test_shell_scripts_are_published_with_unix_line_endings() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes
    assert b"\r\n" not in (ROOT / "deploy.sh").read_bytes()
    assert b"\r\n" not in (RELEASE_ROOT / "deploy.sh").read_bytes()
    with zipfile.ZipFile(RELEASE_ARCHIVE) as package:
        assert b"\r\n" not in package.read("deploy.sh")


def test_release_archive_records_unix_file_permissions() -> None:
    with zipfile.ZipFile(RELEASE_ARCHIVE) as package:
        for entry in package.infolist():
            if entry.is_dir():
                continue
            assert entry.create_system == 3
            expected = 0o755 if entry.filename == "deploy.sh" else 0o644
            mode = entry.external_attr >> 16
            assert stat.S_IFMT(mode) == stat.S_IFREG
            assert stat.S_IMODE(mode) == expected


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


def test_database_and_admin_credentials_have_no_defaults() -> None:
    assert 'DB_PASS=""' in SCRIPT
    assert "${DB_PASS:-" not in SCRIPT
    assert "ADMIN_USER=\n" in ENV_EXAMPLE
    assert "ADMIN_PASS=\n" in ENV_EXAMPLE
    assert "DB_URL=\n" in ENV_EXAMPLE


def test_non_root_foreground_run_does_not_attempt_chown() -> None:
    assert '[ "$(id -u)" -eq 0 ]' in SCRIPT


def test_deployment_requires_and_reuses_python_310_or_newer() -> None:
    assert "select_python" in SCRIPT
    assert "sys.version_info >= (3, 10)" in SCRIPT
    assert '"${PYTHON_BIN}" "${ROOT}/tools/prepare_deployment_env.py"' in SCRIPT
    assert '"${PYTHON_BIN}" -m venv' in SCRIPT


def test_cpu_requirements_install_before_optional_gpu_detection() -> None:
    install_body = _python_install_body()
    cpu = 'pip" install -r "${ROOT}/requirements.txt"'
    gpu = "tools/install_optional_gpu.py"

    assert cpu in install_body
    assert gpu in install_body
    assert install_body.index(cpu) < install_body.index(gpu)


def test_environment_example_selects_v4_and_requires_private_credentials() -> None:
    assert ENV_EXAMPLE.count("ROBUST_WATERMARK_VERSION=4") == 1
    assert ENV_EXAMPLE.count("WATERMARK_AUTH_KEY=") == 1
    assert "WATERMARK_AUTH_KEY=\n" in ENV_EXAMPLE
    assert "ROBUST_WATERMARK_STRENGTH=0.74" in ENV_EXAMPLE
    assert "ADMIN_USER=\n" in ENV_EXAMPLE
    assert "ADMIN_PASS=\n" in ENV_EXAMPLE
    assert "DB_URL=\n" in ENV_EXAMPLE
    assert ENV_EXAMPLE.count("TRACE_COMPUTE_DEVICE=auto") == 1


def test_deployment_exposes_explicit_json_migration_command() -> None:
    assert "migrate-data" in SCRIPT
    assert "migrate_json_to_mysql.py" in SCRIPT
    assert "trace-private-migration-backups" in SCRIPT


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


def test_readme_documents_adaptive_gpu_install_and_cpu_isolation() -> None:
    for expected in (
        "TRACE_COMPUTE_DEVICE=auto|cpu|cuda",
        "CPU 主机不会安装 GPU 软件包",
        "自动回退到 CPU",
    ):
        assert expected in README


def test_readme_uses_the_timestamped_release_filename() -> None:
    assert "trace-v4-centos-YYYYMMDD-HHMMSS.zip" in README
    assert "trace-v4-centos-20260717.zip" not in README
