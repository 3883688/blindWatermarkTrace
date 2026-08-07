#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"

usage() {
  cat <<EOF
Usage:
  ./pg.sh --dry-run   Validate the MySQL source and PostgreSQL target settings
  ./pg.sh             Migrate MySQL data to PostgreSQL
  ./pg.sh [options]   Pass options to the migration script

Required in ${ENV_FILE}:
  DB_URL=...          MySQL source connection
  POSTGRES_URL=...    PostgreSQL target connection
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    python_bin="${ROOT}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
  else
    echo "Python 3 is required." >&2
    exit 1
  fi
fi

if ! "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi

if ! "${python_bin}" -c 'import dotenv, psycopg, sqlalchemy' >/dev/null 2>&1; then
  echo "Python dependencies are missing. Run: ${python_bin} -m pip install -r ${ROOT}/requirements.txt" >&2
  exit 1
fi

if ! "${python_bin}" - "${ENV_FILE}" <<'PY'
import sys
from pathlib import Path

names = {"DB_URL": False, "POSTGRES_URL": False}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() in names:
        names[key.strip()] = bool(value.strip())
missing = [key for key, present in names.items() if not present]
if missing:
    print("Missing required settings: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 1
fi

exec "${python_bin}" "${ROOT}/tools/migrate_mysql_to_postgresql.py" \
  --env-file "${ENV_FILE}" "$@"
