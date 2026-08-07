# Database Migration and Secret Purge Design

## Goal

Move the five runtime JSON datasets under `data/` into MySQL, make MySQL the only runtime data source, store user passwords as salted hashes, remove plaintext credentials from tracked artifacts, and purge the leaked datasets and credentials from Git history.

The deployed credentials remain unchanged for now. Their values must exist only in the ignored `.env` file and must never be copied into source, examples, documentation, tests, migration payloads, release archives, or Git history.

## Scope

The migration covers:

- `data/images.json`
- `data/detection_stats.json`
- `data/watermark_stats.json`
- `data/roles.json`
- `data/users.json`
- User creation, update, deletion, listing, and login
- Database and administrator credentials currently embedded in tracked files
- Generated release directories and archives that contain leaked credentials
- All reachable local Git history, followed by the rewritten `master` branch on `origin`

Feature-index files and uploaded image files are outside this data migration. They remain filesystem-backed because image records refer to their paths.

## Storage Design

### Image records

Continue using the existing `image_records` table. Each source record is stored with its stable ID, ordering index, serialized record data, and timestamps. The migration must preserve record order and every source field.

### Users

Add a `users` table with:

- `username` as the primary key
- `password_hash` containing a versioned salted password hash
- `role_key` containing the assigned role key
- `created_at` and `updated_at` timestamps

Use the Python standard library's `hashlib.scrypt` with a cryptographically random per-user salt. Store the algorithm name, parameters, salt, and derived key in one versioned encoded value so the verifier can evolve without changing the schema. Existing plaintext passwords are hashed during migration; the passwords users enter do not change.

All user-management operations and login verification use `users`. The application never returns password hashes through an API.

### Roles and statistics

Add a `roles` table with:

- `role_key` as the primary key
- `label` containing the display name
- `menus` containing the JSON-encoded list of allowed menu keys
- `created_at` and `updated_at` timestamps

The `users.role_key` column references `roles.role_key`. Role management reads and updates rows in `roles` while preserving the current API response shape.

Add a `stats` table with:

- `stat_key` as the primary key
- `data` containing the JSON-encoded statistic document
- `updated_at` timestamp

The two rows are keyed as `detection_stats` and `watermark_stats`. The old `app_json_store` table is no longer used.

## Configuration and Startup

Load configuration from `.env` through the existing `python-dotenv` integration. `DB_URL`, `ADMIN_USER`, and `ADMIN_PASS` have no real defaults in source code. The application fails clearly during startup when required configuration is missing or the database cannot be initialized.

`.env` stays ignored. `.env.example` contains names and non-secret placeholders only. Deployment scripts, documentation, tests, and generated release artifacts must not contain real credential values.

JSON fallback writes are removed. A database outage must produce an explicit service error instead of recreating or modifying JSON files.

## Migration Script

Provide an idempotent command-line script for execution on the server. It reads `DB_URL` from `.env` and accepts a data directory argument, defaulting to `data/`.

The script performs these steps:

1. Validate all five JSON files and their expected top-level shapes without changing the database or filesystem.
2. Connect to MySQL and create the required tables if they do not exist.
3. Begin one database transaction.
4. Import image records, roles, statistics, and users; hash every imported user password.
5. Re-read the database and verify image count, usernames, role keys, and exact statistics against the validated source.
6. Commit only after all verification succeeds.
7. Rename each source JSON to a private timestamped backup outside the repository data path, then remove the originals from `data/`.

On validation, connection, import, or verification failure, the transaction rolls back and all JSON inputs remain untouched. Re-running the script produces the same database state and does not duplicate records.

The repository version of the script contains migration logic only. It never embeds source records or passwords.

## Runtime Data Flow

- Image APIs read and write `image_records`.
- Dashboard counters read and write the corresponding `stats` rows.
- Role management reads and writes `roles` rows.
- User management reads and writes `users` rows.
- Login loads the user by username and verifies the submitted password against its scrypt hash.
- The configured administrator account is seeded into `users` when absent, using `ADMIN_PASS` from `.env` and a generated salt.

No runtime path reads or writes the five migrated JSON files.

## Git and Release Sanitization

Remove the five JSON paths from the current tree and add the runtime JSON paths to `.gitignore`. Remove credential defaults and values from all tracked text files. Delete the old release ZIP and checksum because binary archive contents cannot be safely text-replaced; regenerate a sanitized release only after verification.

Rewrite every reachable branch and tag to:

- Remove the five runtime JSON files from all commits.
- Remove the leaked release ZIP and checksum from all commits.
- Replace leaked credential values in remaining textual history with non-secret placeholders.

After rewriting, scan every reachable object for the known leaked values and data paths. Only after the scan is clean should the rewritten `master` branch and affected tags be force-pushed to `origin`. Collaborators must re-clone or hard-reset onto the rewritten history.

History removal does not revoke credentials from existing clones, caches, or logs. Credential rotation remains recommended even though it is explicitly deferred for this change.

## Error Handling

- Missing required environment variables: abort startup or migration with the variable name, never its value.
- Invalid JSON shape: abort before opening a write transaction.
- Database failure: roll back and preserve source files.
- Hashing or user-row failure: roll back the entire import.
- Verification mismatch: roll back and report the mismatched dataset without printing credentials.
- Filesystem cleanup failure after commit: report the committed database state and exact remaining files; a rerun may complete cleanup safely.

## Testing and Verification

Automated tests cover:

- Password hashes are salted, versioned, verifiable, and never equal plaintext.
- Existing user passwords still authenticate after migration.
- User create, update, delete, list, and login operate through the database repository.
- Database failures never create JSON fallback files.
- Migration rejects malformed input without database or filesystem changes.
- Migration is idempotent and only removes inputs after successful verification.
- Configuration examples and release artifacts contain no known credentials.
- The release package excludes `.env` and runtime JSON data.

Final verification includes the focused test suite, the full relevant test suite, a dry-run migration against an isolated test database or SQLAlchemy-compatible test fixture, a clean release-content scan, and a Git object scan after history rewriting.
