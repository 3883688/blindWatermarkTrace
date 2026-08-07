# Trace App Root Module Organization Design

## Context

The `v4-production-hardening` branch at `7e545c7` keeps five production modules
at the repository root alongside the three-line `main.py` compatibility entrypoint.
Those modules belong to existing `trace_app` domains, and their root placement
forces production code, migration tools, tests, and release packaging to rely on
top-level imports and special root-file rules.

This change organizes the modules under `trace_app` without changing application
behavior, API contracts, database behavior, deployment commands, or algorithms.

## Goals

- Leave `main.py` as the only Python file at the repository root.
- Preserve `uvicorn main:app` and the existing deployment entrypoint.
- Place each production module in the `trace_app` domain that owns it.
- Replace all internal, tool, and test imports with explicit package imports.
- Package the moved modules through the existing recursive `trace_app` release rule.
- Preserve Git history through tracked file moves.

## Non-Goals

- No API, schema, data migration, environment, or algorithm changes.
- No changes to the contents or behavior of the moved modules beyond import paths.
- No root-level compatibility shims for the five old module names.
- No changes to untracked images, models, release archives, or runtime data.

## Module Mapping

| Current path | New path | Ownership |
| --- | --- | --- |
| `candidate_feature_index.py` | `trace_app/imaging/candidate_feature_index.py` | Image feature indexing |
| `database_store.py` | `trace_app/database/store.py` | Database persistence |
| `password_security.py` | `trace_app/auth/password_security.py` | Password hashing and verification |
| `watermark_auth.py` | `trace_app/watermark/auth.py` | Watermark authentication coding |
| `watermark_ecc.py` | `trace_app/watermark/ecc.py` | Watermark error correction |

`main.py` remains at the repository root and continues to alias
`trace_app.compat`, preserving the current ASGI startup contract.

## Import And Dependency Migration

All production imports use absolute `trace_app` package paths after the move.
The same rule applies to scripts in `tools/`, tests, and monkeypatch target strings.
No `sys.path` manipulation, dynamic module aliases, or duplicate source files are
introduced to hide stale imports.

The refactor must not introduce a package cycle. If a moved module exposes an
existing low-level dependency, its callers import it from the owning package;
the moved module must not import a higher-level service merely to preserve an old
path.

Old external imports such as `import database_store` are intentionally no longer
supported. The stable external compatibility surface is the application entrypoint
`main:app`, not the historical root module layout.

## Release And Deployment

`tools/build_centos_release.py` stops listing the five moved modules in
`ROOT_FILES`. They are included by the existing recursive `trace_app` collection.
`main.py` remains an explicit root release file.

Release, source-filter, secret-hygiene, and backup-contract tests are updated to
refer to the new paths. Deployment scripts continue to execute
`python -m uvicorn main:app`, so server commands and systemd configuration do not
change.

## Failure Handling

The change is complete only when there are no stale imports or packaging references.
Import errors, circular imports, missing release files, or a failed ASGI startup are
blocking failures. They are fixed at the package boundary rather than hidden by
compatibility shims.

No database migration is run as part of this refactor. Runtime smoke tests use an
isolated local database and must not mutate the configured production database.

## Verification

Tests are added or updated to prove:

1. `main.py` is the only root-level Python file.
2. Each moved module imports from its new package path.
3. Source and tests contain no old top-level imports or monkeypatch paths.
4. `uvicorn main:app` remains a valid startup contract.
5. Database, authentication, watermark, migration, and application-structure tests
   retain their behavior.
6. Release collection includes every moved module through `trace_app` and no longer
   requires the old root paths.
7. The refactored `7e545c7` application starts from its worktree and returns HTTP 200.

The test sequence follows a red-green workflow: introduce the new structure/import
contract first, confirm that it fails against the old layout, then move files and
update consumers until the focused and regression suites pass.

## Acceptance Criteria

- The repository root contains only `main.py` among Python files.
- Existing deployment commands remain unchanged.
- No old root module import remains in production code, tools, or tests.
- No special root packaging entry remains for the five moved modules.
- Relevant backend, migration, security, release, and startup checks pass.
- No unrelated tracked or untracked file is modified.
