# PostgreSQL Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command, transactional MySQL-to-PostgreSQL migration with normalized role/menu tables.

**Architecture:** Read and validate the complete MySQL source before opening a PostgreSQL write transaction. Keep compatibility `roles.menus` JSON while adding `menus` and `role_menus`, then verify exact projections before commit. The script never mutates the source.

**Tech Stack:** Python, SQLAlchemy, PyMySQL, psycopg, PostgreSQL.

---

### Task 1: Migration normalization tests

**Files:**
- Create: `tests/test_mysql_to_postgresql_migration.py`

- [x] Write tests for menu-key normalization, custom-role preservation, source validation, and target row projection.
- [x] Run `pytest tests/test_mysql_to_postgresql_migration.py -q`; expected failure because the migration module does not exist.

### Task 2: Implement migration script

**Files:**
- Create: `tools/migrate_mysql_to_postgresql.py`

- [x] Implement source extraction and validation from `DB_URL`.
- [x] Implement PostgreSQL schema creation, normalized menu tables, transactional replacement, verification, and `--dry-run`.
- [x] Run the focused tests and confirm they pass.

### Task 3: Driver and operator configuration

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

- [x] Add `psycopg[binary]` and document `POSTGRES_URL` without putting credentials in the repository.
- [x] Run dependency import and script help smoke tests.

### Task 4: Live dry-run and migration

**Files:**
- No additional files.

- [x] Run the script with `--dry-run` against the configured MySQL source.
- [x] Run the script against the local PostgreSQL target only after dry-run validation passes.
- [x] Verify counts, role menus, users, image records, and stats from PostgreSQL.
