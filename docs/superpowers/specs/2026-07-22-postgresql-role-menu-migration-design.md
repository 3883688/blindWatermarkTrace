# PostgreSQL Role/Menu Migration Design

**Date:** 2026-07-22

## Goal

Migrate the complete application dataset from the current MySQL database to a
PostgreSQL database without deleting or changing the MySQL source, while
normalizing role/menu permissions and preserving the existing API contract.

This phase is limited to database migration. Visual retrieval, pgvector image
indexes, and watermark detection changes are subsequent work.

## Current Data

The current MySQL database contains `image_records`, `roles`, `users`, and
`stats`. The `roles.menus` column is JSON text. The application currently has
the system roles `admin`, `operator`, and `viewer`; custom role keys must also
be preserved and supported.

## Target Schema

PostgreSQL keeps the existing application tables and adds normalized permission
tables:

- `roles(role_key primary key, label, is_system, created_at, updated_at)`
- `menus(menu_key primary key, label, sort_order, enabled)`
- `role_menus(role_key, menu_key, primary key(role_key, menu_key), foreign keys)`

`users.role_key` references `roles.role_key`. Existing `image_records` JSON
payloads remain intact so application record fields do not change during this
phase. Existing `stats` keys and JSON values are copied unchanged.

Menu keys found in old role JSON are all inserted into `menus`. Known keys use
the application labels; unknown keys use the key as a temporary label so no
permission disappears during migration. System roles are marked by membership
in the existing default role set; imported custom roles are marked
`is_system = false`.

## Migration Script

Create `tools/migrate_mysql_to_postgresql.py` with two explicit connection
inputs:

- `DB_URL`: read-only MySQL source.
- `POSTGRES_URL`: PostgreSQL target, normally using the `psycopg` SQLAlchemy
  driver.

The script must:

1. Validate both URLs and read all source rows before opening a target write
   transaction.
2. Create the PostgreSQL schema and `vector` extension when available; vector
   tables are not populated in this phase.
3. Validate source IDs, role keys, user role references, menu shapes, and JSON
   payloads before writing.
4. In one PostgreSQL transaction, upsert roles and menus, replace role-menu
   links, copy users with their existing password hashes, copy image records
   with their position indexes, and copy stats.
5. Verify counts, role labels, menu assignments, user role assignments,
   password-hash equality, image IDs, position indexes, and exact stats JSON
   through the same transaction connection.
6. Commit only after verification succeeds. Never delete or mutate MySQL
   rows, and never remove local source files.
7. Support `--dry-run` to validate and report counts without writing.

The migration is repeatable: a later run replaces the target dataset inside a
transaction and produces the same normalized role/menu state. It must not
create duplicate role-menu links or users.

## Application Compatibility

The repository will read role menus through a join and continue returning the
current shape:

```json
{
  "roles": {
    "operator": {"label": "操作员", "menus": ["watermark", "trace"]}
  }
}
```

Role creation and deletion can be added after migration using the normalized
tables. Deletion must reject roles referenced by users and must never delete a
system role. Existing role-menu update behavior remains available.

## Configuration and Cutover

The migration uses `POSTGRES_URL` without changing `DB_URL`. After successful
verification, a separate cutover change will point `DB_URL` at PostgreSQL and
run the application contract tests. The migration script itself does not
restart services or change `.env` automatically.

## Testing

Add unit tests for source validation, menu normalization, dry-run behavior,
transaction rollback on verification failure, and idempotent reruns using
temporary SQLAlchemy databases. Add an integration test that seeds a source
fixture containing system and custom roles, migrates it, and verifies exact
role/menu/user/image/statistics projections from PostgreSQL.

Before execution, run the focused migration tests and a dry run against the
configured MySQL source. Only after the dry run passes should the script be
run against the local PostgreSQL target.
