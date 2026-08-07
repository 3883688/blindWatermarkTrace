# V4 Original UI Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the original UI backed exclusively by relational V4 generation, detection, media, authentication, and management services.

**Architecture:** Keep the original browser API contract while replacing its implementations with adapters over `V4Repository`, `V4MediaService`, verified CPU ONNX models, and the HMAC64/RS V4 watermark. Production identity data uses normalized role-menu rows; production image, statistics, job, and metadata persistence contains no JSON columns or JSON full scans.

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL 16, pgvector, ONNX Runtime CPU, OpenCV, Pillow, pytest, Docker Compose.

---

### Task 1: Normalize Production Persistence

**Files:**
- Modify: `database_store.py`
- Modify: `trace_app/database/connection.py`
- Modify: `trace_app/v4/schema.py`
- Modify: `trace_app/v4/repository.py`
- Modify: `trace_app/v4/generation.py`
- Modify: `trace_app/v4/jobs.py`
- Test: `tests/v4/test_relational_only_schema.py`

- [ ] Write failing schema tests asserting production tables have no JSON/JSONB/Text-encoded record columns, role menus use `role_menus`, records use `original_filename`, and job results use explicit columns.
- [ ] Run `py -m pytest -q tests/v4/test_relational_only_schema.py` and confirm the JSON-column assertions fail.
- [ ] Add normalized role-menu, record, and job columns; make production startup create identity-only legacy storage plus V4 tables.
- [ ] Run persistence, auth, repository, and job tests until green.
- [ ] Commit the relational persistence change.

### Task 2: Build Verified CPU V4 Runtime

**Files:**
- Create: `trace_app/v4/production.py`
- Modify: `trace_app/v4/repository.py`
- Test: `tests/v4/test_production_runtime.py`

- [ ] Write failing tests for verified model loading, DINO/pgvector group artifacts, HMAC64/RS embedding, owner-scoped exact detection, and opaque media storage.
- [ ] Run the focused runtime tests and confirm missing production factories fail.
- [ ] Implement decode, embed, DINO multi-view artifacts, ORB feature serialization, LightGlue geometry confirmation, and V4 service factories with concurrency one.
- [ ] Run focused runtime and five-real-image smoke tests.
- [ ] Commit the production runtime.

### Task 3: Preserve Original HTTP and UI Contract

**Files:**
- Create: `trace_app/api/compat_v4.py`
- Modify: `trace_app/application.py`
- Modify: `trace_app/dependencies.py`
- Replace: `assets/app/app.js` with the original release asset
- Test: `tests/v4/test_original_ui_compat.py`

- [ ] Write failing tests for original `/api/watermark`, `/api/images`, and `/api/dashboard-stats` paths backed by V4 services and opaque `/api/media/{id}` URLs.
- [ ] Confirm tests fail because compatibility routes and production services are absent.
- [ ] Implement authenticated original-contract adapters and register them instead of legacy JSON-backed routes.
- [ ] Restore the original JS asset and assert it contains original endpoints and no `/api/v4` endpoint dependency.
- [ ] Run API, authentication, ownership, and frontend contract tests.
- [ ] Commit original UI compatibility.

### Task 4: Destructive V4-Only Deployment Migration

**Files:**
- Create: `tools/migrate_v4_relational_only.py`
- Test: `tests/v4/test_relational_migration.py`

- [ ] Write a PostgreSQL migration test with legacy JSON tables and preserved users/roles.
- [ ] Confirm it fails before the migration exists.
- [ ] Implement backup-gated migration: normalize role menus, drop legacy algorithm/JSON tables and columns, create V4/pgvector schema, verify identities unchanged.
- [ ] Run migration twice to prove idempotency.
- [ ] Commit the migration.

### Task 5: Verification and Full Update Package

**Files:**
- Create: `release/trace-v4-original-ui-relational-<version>/update.sh`
- Create: `release/trace-v4-original-ui-relational-<version>/FILES.txt`
- Create: `release/trace-v4-original-ui-relational-<version>/UPDATE_NOTES.md`

- [ ] Run the complete test suite and focused PostgreSQL tests.
- [ ] Run core algorithms against available files under `img/`.
- [ ] Build a CPU-only Tencent-mirror package excluding `.env`, models, database dumps, images, and uploads.
- [ ] Verify archive contents, script syntax, SHA-256, rollback behavior, and original frontend checksum.
- [ ] Copy the final archive and checksum to the main `release` directory.
