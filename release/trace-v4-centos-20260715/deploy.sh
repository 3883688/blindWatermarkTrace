#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-trace-system}"
PORT="${PORT:-6868}"
HOST_ADDRESS="${HOST_ADDRESS:-0.0.0.0}"
APP_NAME="${APP_NAME:-WatermarkSystem}"
DB_NAME=""
DB_USER=""
DB_PASS=""
DB_HOST=""
DB_PORT=""
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
PYTHON_BIN="${PYTHON_BIN:-}"
LAST_BACKUP="none (fresh deployment)"
cd "$ROOT"

usage() {
  cat <<EOF
Usage:
  sudo ./deploy.sh install-service   Back up runtime data, prepare V4, and restart systemd
  ./deploy.sh run                    Run V4 uvicorn in the foreground
  ./deploy.sh check-db               Validate the existing MySQL/MariaDB database
  ./deploy.sh migrate-data            Import five JSON files into database tables
  sudo ./deploy.sh status            Show service status

Defaults:
  SERVICE_NAME=${SERVICE_NAME}
  HOST_ADDRESS=${HOST_ADDRESS}
  PORT=${PORT}
  Database settings are read from .env only.
EOF
}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root: sudo $0 $*" >&2
    exit 1
  fi
}

pkg_install() {
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y "$@"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "$@"
  else
    echo "Neither dnf nor yum is available. This script targets CentOS/RHEL." >&2
    exit 1
  fi
}

install_system_packages() {
  pkg_install python3 python3-pip firewalld tar
}

select_python() {
  local candidate
  local candidates=()
  if [ -n "${PYTHON_BIN}" ]; then
    candidates=("${PYTHON_BIN}")
  else
    candidates=(python3.13 python3.12 python3.11 python3.10 python3)
  fi
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
      PYTHON_BIN="$(command -v "$candidate")"
      echo "Using Python: ${PYTHON_BIN}"
      return 0
    fi
  done
  echo "Python 3.10 or newer is required. Install it or set PYTHON_BIN." >&2
  return 1
}

write_env() {
  if [ ! -f "${ROOT}/.env" ]; then
    cp "${ROOT}/.env.example" "${ROOT}/.env"
    echo "Created .env from .env.example"
  fi
}

prepare_deployment_environment() {
  "${PYTHON_BIN}" "${ROOT}/tools/prepare_deployment_env.py" --env-file "${ROOT}/.env"
  if [ "$(id -u)" -eq 0 ] && id "${SERVICE_USER}" >/dev/null 2>&1; then
    chown "${SERVICE_USER}:$(id -gn "${SERVICE_USER}")" "${ROOT}/.env"
  fi
}

load_env_config() {
  local parsed
  parsed="$(ENV_FILE="${ROOT}/.env" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

values = {}
for line in Path(os.environ["ENV_FILE"]).read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        name, value = stripped.split("=", 1)
        values[name.strip()] = value

url = urlparse(values.get("DB_URL", ""))
print(unquote(url.username or ""))
print(unquote(url.password or ""))
print(url.hostname or "")
print(url.port or 3306)
print((url.path or "").lstrip("/"))
print(values.get("APP_NAME", ""))
PY
)"
  DB_USER="$(printf '%s\n' "$parsed" | sed -n '1p')"
  DB_PASS="$(printf '%s\n' "$parsed" | sed -n '2p')"
  DB_HOST="$(printf '%s\n' "$parsed" | sed -n '3p')"
  DB_PORT="$(printf '%s\n' "$parsed" | sed -n '4p')"
  DB_NAME="$(printf '%s\n' "$parsed" | sed -n '5p')"
  local app_name_from_env
  app_name_from_env="$(printf '%s\n' "$parsed" | sed -n '6p')"
  if [ -n "$app_name_from_env" ]; then
    APP_NAME="$app_name_from_env"
  fi
}

check_database() {
  load_env_config
  if ! command -v mysql >/dev/null 2>&1; then
    echo "mysql client is required to validate the existing database" >&2
    return 1
  fi
  if ! MYSQL_PWD="${DB_PASS}" mysql \
    -u"${DB_USER}" \
    -h"${DB_HOST}" \
    -P"${DB_PORT}" \
    "${DB_NAME}" \
    -e "SELECT 1" >/dev/null 2>&1; then
    echo "Existing database is not reachable: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}" >&2
    return 1
  fi
  echo "Existing database connection verified: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
}

migrate_data() {
  select_python
  write_env
  prepare_deployment_environment
  check_database
  install_python_environment
  "${ROOT}/.venv/bin/python" "${ROOT}/tools/migrate_json_to_mysql.py" \
    --env-file "${ROOT}/.env" \
    --data-dir "${ROOT}/data" \
    --backup-dir "${ROOT}/../trace-private-migration-backups"
}

backup_runtime_state() {
  local stamp archive
  local items=()
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${ROOT}/backups/deploy"
  archive="${ROOT}/backups/deploy/pre-deploy-${stamp}.tar.gz"
  [ -f "${ROOT}/.env" ] && items+=(".env")
  [ -d "${ROOT}/data" ] && items+=("data")
  [ -d "${ROOT}/uploads" ] && items+=("uploads")
  if [ "${#items[@]}" -gt 0 ]; then
    tar -czf "$archive" "${items[@]}"
    LAST_BACKUP="$archive"
  fi
  echo "Runtime backup: ${LAST_BACKUP}"
}

install_python_environment() {
  "${PYTHON_BIN}" -m venv "${ROOT}/.venv"
  "${ROOT}/.venv/bin/python" -m pip install --upgrade pip
  "${ROOT}/.venv/bin/pip" install -r "${ROOT}/requirements.txt"
  mkdir -p \
    "${ROOT}/data" \
    "${ROOT}/uploads/originals" \
    "${ROOT}/uploads/watermarked" \
    "${ROOT}/uploads/thumbnails" \
    "${ROOT}/logs"
  if [ "$(id -u)" -eq 0 ] && id "${SERVICE_USER}" >/dev/null 2>&1; then
    chown -R "${SERVICE_USER}:$(id -gn "${SERVICE_USER}")" \
      "${ROOT}/data" "${ROOT}/uploads" "${ROOT}/logs"
  fi
}

write_systemd_service() {
  local service_file="/etc/systemd/system/${SERVICE_NAME}.service"
  cat > "$service_file" <<EOF
[Unit]
Description=Trace System FastAPI Service
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${ROOT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${ROOT}/.venv/bin/python -m uvicorn main:app --host ${HOST_ADDRESS} --port ${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
}

wait_for_http_health() {
  local attempt
  for attempt in $(seq 1 60); do
    if "${ROOT}/.venv/bin/python" -c \
      "import urllib.request; response=urllib.request.urlopen('http://127.0.0.1:${PORT}/', timeout=2); raise SystemExit(0 if response.status == 200 else 1)" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  systemctl --no-pager status "${SERVICE_NAME}" || true
  journalctl -u "${SERVICE_NAME}" -n 80 --no-pager || true
  echo "Deployment health check failed. Backup: ${LAST_BACKUP}" >&2
  return 1
}

install_service() {
  need_root install-service
  install_system_packages
  select_python
  backup_runtime_state
  write_env
  prepare_deployment_environment
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

run_server() {
  select_python
  write_env
  prepare_deployment_environment
  install_python_environment
  echo "Starting ${APP_NAME} V4 at http://${HOST_ADDRESS}:${PORT}"
  exec "${ROOT}/.venv/bin/python" -m uvicorn main:app --host "${HOST_ADDRESS}" --port "${PORT}"
}

show_status() {
  systemctl --no-pager status "${SERVICE_NAME}" || true
}

case "${1:-install-service}" in
  install-service)
    install_service
    ;;
  check-db)
    select_python
    write_env
    prepare_deployment_environment
    check_database
    ;;
  migrate-data)
    migrate_data
    ;;
  run)
    run_server
    ;;
  status)
    show_status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
