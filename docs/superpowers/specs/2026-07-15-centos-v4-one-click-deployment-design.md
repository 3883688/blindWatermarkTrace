# CentOS V4 One-Click Deployment Design

## Goal

Keep the existing V1-era CentOS deployment workflow (`deploy.sh`, existing MySQL/MariaDB, systemd, direct HTTP port) while deploying the current V4 watermark implementation safely with one command:

```bash
sudo ./deploy.sh install-service
```

Repeated deployment must preserve the existing database, `.env`, `data/`, and `uploads/` content.

## Scope

This iteration covers a single CentOS/RHEL server with an existing reachable MySQL or MariaDB instance. It does not install or reset the database server, add Docker, configure Nginx/TLS, or automate real social-platform evidence collection.

The remaining commercial gates stay recorded in `docs/commercial_watermark_upgrade_plan.md`: real WeChat, browser, and target-platform route collection; broader real-image coverage; production load and recovery testing; managed-secret migration; and production monitoring/security controls.

## Deployment Architecture

`deploy.sh` remains the public entry point and installs the Python environment, validates configuration, checks the existing database, creates required directories, writes the systemd unit, restarts the service, and verifies HTTP availability.

A focused Python helper, `tools/prepare_deployment_env.py`, owns `.env` mutation. It uses the standard library rather than shell text replacement so duplicate keys are removed deterministically and the file is atomically replaced. The helper:

- preserves all unrelated existing values;
- sets exactly one `ROBUST_WATERMARK_VERSION=4` entry;
- preserves a valid existing `WATERMARK_AUTH_KEY`;
- generates 48 CSPRNG bytes and stores their 64-character Base64 representation when the key is missing, blank, duplicated, or shorter than 32 UTF-8 bytes;
- never prints secret values;
- sets file permissions to `0600` on Linux;
- requires a nonempty `DB_URL` when `DB_ENABLED=true`.

`.env.example` keeps the existing database and administrator defaults, selects V4, and leaves `WATERMARK_AUTH_KEY` blank so each deployment receives a unique generated key. Existing server `.env` files are not replaced, and their database and administrator values are preserved.

## Preservation And Backup

Before mutating configuration or restarting the service, `deploy.sh` creates a timestamped archive under `backups/deploy/` containing the existing `.env`, `data/`, and `uploads/` paths that are present. It does not delete old backups automatically.

The script never drops, recreates, migrates, or clears the existing database. It only parses `DB_URL` and runs `SELECT 1` using the application credentials. An unreachable database is a deployment failure.

Existing business data stays in place throughout deployment. Directory creation uses idempotent operations and does not change existing file contents.

## Deployment Flow

1. Require root for systemd installation and verify CentOS/RHEL package tooling.
2. Install only missing runtime packages: Python, Python venv support where available, pip, and firewalld support. Do not install MySQL/MariaDB server.
3. Create the timestamped configuration/data archive.
4. Create `.env` from the existing template only when absent, then run `prepare_deployment_env.py`.
5. Parse `DB_URL` without logging its password and verify the existing database with `SELECT 1`.
6. Create or update `.venv`, install `requirements.txt`, and create runtime directories.
7. Install the systemd unit using the invoking non-root user where available. Run Uvicorn with `ROBUST_WATERMARK_VERSION=4` supplied by `.env`.
8. Restart the service and poll `http://127.0.0.1:<port>/` until it returns HTTP 200 or a fixed startup deadline expires.
9. On success, print the service name, URL, backup path, and operational commands. Do not print `WATERMARK_AUTH_KEY`.
10. On failure, show `systemctl status`, recent journal output, the backup path, and explicit recovery commands; return nonzero.

## Error Handling

Deployment stops before service restart when Python, `.env`, V4 key preparation, dependency installation, or database validation fails. The existing running service is not stopped during these preflight stages.

After restart, failure to become active or return HTTP 200 is an error. The script reports diagnostics and leaves the timestamped backup intact. It does not silently switch to V1, disable the database, weaken authentication, or generate a second key on every run.

## Frontend Alignment

The frontend already submits `robust_watermark_version=4`, so no API workflow change is required. Result presentation will be aligned with V4:

- V4 generation results show authenticated DCT carrier plus FFT synchronization instead of zero-valued legacy DCT/DWT/FFT scores.
- V4 extraction results show authenticated tile count, phase count, corrected symbols, bit errors, synchronization confidence, and elapsed detection time when present.
- Legacy records retain their current score display.
- All response values continue through existing HTML escaping helpers.

## Tests

Test-driven changes will cover:

- missing key generation without revealing its value;
- valid key preservation across repeated deployment preparation;
- duplicate key collapse and forced V4 selection;
- unrelated `.env` values and comments remaining intact;
- rejection of enabled database configuration without `DB_URL`;
- deployment script contract: no fixed watermark authentication key, no database server installation/reset, backup before restart, V4 preparation, database check, systemd restart, and HTTP health polling;
- frontend V4 evidence rendering and suppression of misleading legacy frequency scores;
- existing V4 API, detector, frontend, and full pytest regressions.

## Acceptance Criteria

- On a CentOS server with the current `.env` and reachable database, `sudo ./deploy.sh install-service` finishes with exit code 0 and HTTP 200.
- Existing `.env`, database records, `data/`, and `uploads/` remain present.
- `.env` contains exactly one nonempty `WATERMARK_AUTH_KEY` of at least 32 UTF-8 bytes and exactly one `ROBUST_WATERMARK_VERSION=4` entry.
- Re-running the command preserves the same authentication key.
- Existing fixed database and administrator deployment defaults remain unchanged; no fixed `WATERMARK_AUTH_KEY` is added to `deploy.sh`, `.env.example`, or `README_DEPLOY.md`.
- The systemd service runs the V4 application and does not fall back to V1.
- V4 frontend results display V4 evidence accurately.
