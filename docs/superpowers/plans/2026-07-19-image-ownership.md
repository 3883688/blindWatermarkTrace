# Image Ownership and Management Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store each image's uploader as `image_records.user_id -> users.id`, then let administrators manage all images while every other authenticated user can manage only their own.

**Architecture:** Bring the SQLAlchemy metadata in line with the deployed relational schema, add transactional owner-aware repository operations, and backfill null historical owners to the configured administrator. Keep the watermark document's existing string `data.user_id` separate from the relational owner, validate the existing login token through a shared process-local session registry, and apply the authenticated numeric user ID at the API boundary.

**Tech Stack:** Python 3.10+, FastAPI dependencies, SQLAlchemy Core, SQLite/MySQL-compatible migrations, pytest, Vue 3, Vitest, Vite.

---

## File Structure

**Create:**

- `tests/test_image_ownership_api.py`: real SQLite/FastAPI integration coverage for list, statistics, deletion, and upload ownership.
- `frontend/src/auth-session.js`: the one owner of cached login state, Bearer headers, and invalid-session events.

**Modify:**

- `database_store.py`: map numeric user IDs and relational image owners; implement identity lookup, backfill validation, scoped reads, inserts, and deletes.
- `trace_app/database/repositories.py`: expose owner-aware database operations to services.
- `trace_app/database/connection.py`: seed the administrator, backfill legacy image owners, and validate ownership at startup.
- `trace_app/auth/schemas.py`: define the immutable authenticated-user value.
- `trace_app/auth/service.py`: retain login sessions, resolve current users, and prevent deletion of users that own images.
- `trace_app/runtime.py`: hold the shared process-local token-to-user mapping.
- `trace_app/dependencies.py`: validate Bearer tokens and provide the current user to routes.
- `trace_app/application.py`: wire the shared session mapping into the application auth service.
- `trace_app/compat.py`: preserve shared sessions through the legacy compatibility factories.
- `trace_app/management/service.py`: filter image records and authorize deletion by numeric owner ID.
- `trace_app/api/images.py`: require current-user identity for list and delete routes.
- `trace_app/api/watermark.py`: require current-user identity when creating an image.
- `trace_app/watermark/service.py`: persist the authenticated owner separately from the watermark payload user ID.
- `frontend/src/api/client.js`: attach Bearer tokens and invalidate stale cached sessions on HTTP 401.
- `frontend/src/api/trace.js`: mark login as the only unauthenticated API request.
- `frontend/src/state/app.js`: use the shared auth-session storage functions.
- `frontend/src/App.vue`: react to invalid-session events by reopening the login overlay.
- `tests/test_database_store.py`: unit tests for identity, owner-aware storage, and integrity backfill.
- `tests/test_application_structure.py`: auth-session, runtime, repository, and user-deletion tests.
- `tests/test_watermark_v4_api.py`: authenticate existing protected embed requests and assert relational ownership.
- `tests/test_false_positive_gate.py`: seed and authenticate a test administrator before protected embeds.
- `tests/commercial_benchmark_config.py`: provide one benchmark client authentication helper.
- `tests/commercial_attack_benchmark.py`: authenticate before benchmark embeds.
- `tests/commercial_quality_benchmark.py`: authenticate before benchmark embeds.
- `tests/commercial_negative_benchmark.py`: authenticate before benchmark embeds.
- `tests/commercial_trace_benchmark.py`: authenticate before benchmark embeds.
- `tests/video_platform_smoke_benchmark.py`: authenticate before smoke benchmark embeds.
- `frontend/tests/api-contract.test.js`: test Bearer headers, unauthenticated login, and HTTP 401 invalidation.
- `frontend/tests/app-state.test.js`: retain state persistence coverage through the shared storage module.
- `README.md`: document the Bearer requirement and image visibility rule.
- `assets/app/app.js`, `assets/app/app.css`: rebuilt production frontend artifacts.

### Task 1: Map Numeric User Identity

**Files:**

- Modify: `database_store.py:27-114`
- Modify: `database_store.py:231-306`
- Modify: `trace_app/database/repositories.py:95-112`
- Test: `tests/test_database_store.py`

- [ ] **Step 1: Write the failing user-identity test**

Add this test after `test_user_crud_stores_only_hashes`:

```python
def test_users_have_numeric_ids_and_authentication_returns_identity(
    store: DatabaseStore,
) -> None:
    store.replace_roles(
        {"operator": {"label": "Operator", "menus": ["watermark"]}}
    )
    store.create_user("alice", "secret", "operator")

    identity = store.authenticate_user("alice", "secret")

    assert identity is not None
    assert identity == {
        "id": identity["id"],
        "username": "alice",
        "role": "operator",
    }
    assert isinstance(identity["id"], int)
    assert store.get_user_by_id(identity["id"]) == {
        "id": identity["id"],
        "username": "alice",
        "role": "operator",
    }
    assert store.get_user_by_username("alice") == identity
    assert store.authenticate_user("alice", "wrong") is None

    repository = Repository(store)
    assert repository.get_user_by_id(identity["id"]) == identity
    assert repository.get_user_by_username("alice") == identity
    assert repository.authenticate_user("alice", "secret") == identity

    columns = {column["name"] for column in inspect(store.engine).get_columns("users")}
    assert columns == {
        "id",
        "username",
        "password_hash",
        "role_key",
        "created_at",
        "updated_at",
    }
    assert inspect(store.engine).get_pk_constraint("users")["constrained_columns"] == [
        "id"
    ]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_database_store.py::test_users_have_numeric_ids_and_authentication_returns_identity -q
```

Expected: FAIL because `DatabaseStore` does not map `users.id` and has no `authenticate_user`, `get_user_by_id`, or `get_user_by_username` method.

- [ ] **Step 3: Implement the numeric user model and lookup API**

Change the `users` table declaration to:

```python
self.users = Table(
    "users",
    self.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(128), nullable=False, unique=True),
    Column("password_hash", String(512), nullable=False),
    Column(
        "role_key",
        String(64),
        ForeignKey("roles.role_key"),
        nullable=False,
    ),
    Column(
        "created_at",
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    ),
)
```

Add these methods and retain `authenticate()` as the compatibility projection:

```python
def get_user_by_id(
    self, user_id: int, connection: Connection | None = None
) -> dict[str, Any] | None:
    with self._transaction(connection) as conn:
        row = conn.execute(
            select(
                self.users.c.id,
                self.users.c.username,
                self.users.c.role_key,
            ).where(self.users.c.id == user_id)
        ).mappings().first()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "role": str(row["role_key"]),
    }

def get_user_by_username(
    self, username: str, connection: Connection | None = None
) -> dict[str, Any] | None:
    with self._transaction(connection) as conn:
        row = conn.execute(
            select(
                self.users.c.id,
                self.users.c.username,
                self.users.c.role_key,
            ).where(self.users.c.username == username)
        ).mappings().first()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "role": str(row["role_key"]),
    }

def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
    with self.engine.connect() as connection:
        row = connection.execute(
            select(
                self.users.c.id,
                self.users.c.username,
                self.users.c.password_hash,
                self.users.c.role_key,
            ).where(self.users.c.username == username)
        ).mappings().first()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "role": str(row["role_key"]),
    }

def authenticate(self, username: str, password: str) -> str | None:
    identity = self.authenticate_user(username, password)
    return None if identity is None else str(identity["role"])
```

Import `Repository` in the test and add these repository projections:

```python
def authenticate_user(
    self, username: str, password: str
) -> dict[str, Any] | None:
    return self.store.authenticate_user(username, password)

def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
    return self.store.get_user_by_id(user_id)

def get_user_by_username(self, username: str) -> dict[str, Any] | None:
    return self.store.get_user_by_username(username)
```

Change `clear_all()` to delete in foreign-key-safe order:

```python
def clear_all(self, connection: Connection | None = None) -> None:
    with self._transaction(connection) as conn:
        conn.execute(delete(self.image_records))
        conn.execute(delete(self.users))
        conn.execute(delete(self.roles))
        conn.execute(delete(self.stats))
```

- [ ] **Step 4: Run the database-store tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_database_store.py -q
```

Expected: all tests PASS, including the existing role-only `authenticate()` contract.

- [ ] **Step 5: Commit the numeric identity model**

```powershell
git add database_store.py trace_app/database/repositories.py tests/test_database_store.py
git commit -m "feat: map numeric user identities"
```

### Task 2: Add Owner-Aware Image Storage

**Files:**

- Modify: `database_store.py:31-150`
- Modify: `trace_app/database/repositories.py:27-40`
- Modify: `trace_app/compat.py:277-283`
- Test: `tests/test_database_store.py`
- Test: `tests/test_application_structure.py:1350-1380`

- [ ] **Step 1: Write the failing owner-aware storage tests**

Add to `tests/test_database_store.py`:

```python
def test_image_records_can_be_scoped_inserted_and_deleted_by_owner(
    store: DatabaseStore,
) -> None:
    store.replace_roles(
        {"operator": {"label": "Operator", "menus": ["watermark"]}}
    )
    store.create_user("alice", "alice-secret", "operator")
    store.create_user("bob", "bob-secret", "operator")
    alice_id = store.get_user_by_username("alice")["id"]
    bob_id = store.get_user_by_username("bob")["id"]
    alice_record = {"id": "alice-image", "user_id": "payload-alice"}
    bob_record = {"id": "bob-image", "user_id": "payload-bob"}

    store.insert_record(alice_record, owner_user_id=alice_id)
    store.insert_record(bob_record, owner_user_id=bob_id)

    assert store.read_records() == [bob_record, alice_record]
    assert store.read_records(owner_user_id=alice_id) == [alice_record]
    assert store.read_records(owner_user_id=bob_id) == [bob_record]
    assert store.delete_record("bob-image", owner_user_id=alice_id) is None
    assert store.read_records() == [bob_record, alice_record]
    assert store.delete_record("alice-image", owner_user_id=alice_id) == alice_record
    assert store.read_records() == [bob_record]
    assert store.user_has_images(alice_id) is False
    assert store.user_has_images(bob_id) is True

    with store.engine.connect() as connection:
        stored_owner = connection.execute(
            select(store.image_records.c.user_id).where(
                store.image_records.c.id == "bob-image"
            )
        ).scalar_one()
    assert stored_owner == bob_id
```

Extend the repository regression test in `tests/test_application_structure.py`:

```python
def test_repository_scopes_image_records_by_owner() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("alice", "secret", "operator")
    user_id = store.get_user_by_username("alice")["id"]
    repository = Repository(store)

    repository.add_record({"id": "one"}, owner_user_id=user_id)

    assert repository.read_records(owner_user_id=user_id) == [{"id": "one"}]
    assert repository.delete_record("one", owner_user_id=user_id) == {"id": "one"}
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_database_store.py::test_image_records_can_be_scoped_inserted_and_deleted_by_owner tests/test_application_structure.py::test_repository_scopes_image_records_by_owner -q
```

Expected: FAIL because the owner column and scoped storage methods are missing.

- [ ] **Step 3: Implement transactional owner-aware store methods**

Add this column immediately after the image-record primary key:

```python
Column(
    "user_id",
    Integer,
    ForeignKey("users.id"),
    nullable=True,
),
```

Update `replace_records()` to accept a migration owner and include it in each row:

```python
def replace_records(
    self,
    records: list[dict[str, Any]],
    connection: Connection | None = None,
    *,
    owner_user_id: int | None = None,
) -> None:
    with self._transaction(connection) as conn:
        conn.execute(delete(self.image_records))
        rows = []
        for index, source in enumerate(records):
            record = dict(source)
            record_id = str(record.get("id") or uuid.uuid4().hex)
            record["id"] = record_id
            rows.append(
                {
                    "id": record_id,
                    "user_id": owner_user_id,
                    "position_index": index,
                    "data": json.dumps(record, ensure_ascii=False),
                    "created_at": str(record.get("created_at") or ""),
                }
            )
        if rows:
            conn.execute(insert(self.image_records), rows)
```

Replace `read_records()` and add the three focused operations:

```python
def read_records(
    self,
    connection: Connection | None = None,
    *,
    owner_user_id: int | None = None,
) -> list[dict[str, Any]]:
    query = select(self.image_records.c.data).order_by(
        self.image_records.c.position_index
    )
    if owner_user_id is not None:
        query = query.where(self.image_records.c.user_id == owner_user_id)
    with self._transaction(connection) as conn:
        rows = conn.execute(query).scalars()
        return [json.loads(data) for data in rows]

def insert_record(
    self,
    record: dict[str, Any],
    *,
    owner_user_id: int | None = None,
) -> None:
    source = dict(record)
    record_id = str(source.get("id") or uuid.uuid4().hex)
    source["id"] = record_id
    with self.engine.begin() as connection:
        connection.execute(
            update(self.image_records).values(
                position_index=self.image_records.c.position_index + 1
            )
        )
        connection.execute(
            insert(self.image_records).values(
                id=record_id,
                user_id=owner_user_id,
                position_index=0,
                data=json.dumps(source, ensure_ascii=False),
                created_at=str(source.get("created_at") or ""),
            )
        )

def delete_record(
    self, image_id: str, *, owner_user_id: int | None = None
) -> dict[str, Any] | None:
    condition = self.image_records.c.id == image_id
    if owner_user_id is not None:
        condition = condition & (self.image_records.c.user_id == owner_user_id)
    with self.engine.begin() as connection:
        row = connection.execute(
            select(
                self.image_records.c.data,
                self.image_records.c.position_index,
            ).where(condition)
        ).mappings().first()
        if row is None:
            return None
        connection.execute(delete(self.image_records).where(condition))
        connection.execute(
            update(self.image_records)
            .where(self.image_records.c.position_index > row["position_index"])
            .values(position_index=self.image_records.c.position_index - 1)
        )
    return json.loads(row["data"])

def user_has_images(self, user_id: int) -> bool:
    with self.engine.connect() as connection:
        return connection.execute(
            select(self.image_records.c.id)
            .where(self.image_records.c.user_id == user_id)
            .limit(1)
        ).first() is not None
```

Update `Repository` with matching wrappers:

```python
def read_records(
    self, *, owner_user_id: int | None = None
) -> list[dict[str, Any]]:
    return self.store.read_records(owner_user_id=owner_user_id)

def add_record(
    self,
    record: dict[str, Any],
    *,
    owner_user_id: int | None = None,
) -> None:
    self.store.insert_record(record, owner_user_id=owner_user_id)

def delete_record(
    self, image_id: str, *, owner_user_id: int | None = None
) -> dict[str, Any] | None:
    return self.store.delete_record(image_id, owner_user_id=owner_user_id)

def user_has_images(self, user_id: int) -> bool:
    return self.store.user_has_images(user_id)
```

Keep `replace_records()` and `write_records()` as migration/compatibility
operations. The optional insert owner keeps intermediate legacy callers
working, while the completed upload route always supplies an authenticated
numeric owner and the compatibility wrapper below supplies the administrator.

Preserve the legacy `main.add_record()` helper by assigning its records to the
configured administrator:

```python
def add_record(record: dict[str, Any]) -> None:
    identity = repository.get_user_by_username(ADMIN_USER)
    if identity is None:
        raise HTTPException(status_code=503, detail="管理员账户不可用")
    repository.add_record(record, owner_user_id=int(identity["id"]))
```

- [ ] **Step 4: Run owner storage and migration regression tests**

Run:

```powershell
python -m pytest tests/test_database_store.py tests/test_application_structure.py::test_repository_replaces_and_reads_records tests/test_application_structure.py::test_repository_scopes_image_records_by_owner tests/test_json_mysql_migration.py -q
```

Expected: all selected tests PASS and the JSON migration still preserves serialized image data.

- [ ] **Step 5: Commit owner-aware storage**

```powershell
git add database_store.py trace_app/database/repositories.py trace_app/compat.py tests/test_database_store.py tests/test_application_structure.py
git commit -m "feat: add owner-aware image storage"
```

### Task 3: Backfill and Validate Historical Owners

**Files:**

- Modify: `database_store.py`
- Modify: `trace_app/database/connection.py:9-58`
- Test: `tests/test_database_store.py`
- Test: `tests/test_application_structure.py:500-580`

- [ ] **Step 1: Write failing backfill and startup tests**

Add to `tests/test_database_store.py`:

```python
def test_initialize_image_ownership_backfills_only_null_rows(
    store: DatabaseStore,
) -> None:
    store.replace_roles(
        {
            "admin": {"label": "Admin", "menus": ["manage"]},
            "operator": {"label": "Operator", "menus": ["manage"]},
        }
    )
    store.create_user("admin", "admin-secret", "admin")
    store.create_user("alice", "alice-secret", "operator")
    admin_id = store.get_user_by_username("admin")["id"]
    alice_id = store.get_user_by_username("alice")["id"]
    store.replace_records([{"id": "legacy"}])
    store.insert_record({"id": "alice-image"}, owner_user_id=alice_id)

    assert store.initialize_image_ownership("admin") == 1
    assert store.initialize_image_ownership("admin") == 0
    assert store.read_records(owner_user_id=admin_id) == [{"id": "legacy"}]
    assert store.read_records(owner_user_id=alice_id) == [{"id": "alice-image"}]


def test_initialize_image_ownership_rejects_an_orphaned_owner(
    store: DatabaseStore,
) -> None:
    store.replace_roles({"admin": {"label": "Admin", "menus": ["manage"]}})
    store.create_user("admin", "admin-secret", "admin")
    store.insert_record({"id": "orphan"}, owner_user_id=999999)

    with pytest.raises(ValueError, match="invalid owner: orphan"):
        store.initialize_image_ownership("admin")


def test_initialize_image_ownership_rejects_duplicate_legacy_usernames() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "username VARCHAR(128) NOT NULL, "
            "password_hash VARCHAR(512) NOT NULL, "
            "role_key VARCHAR(64) NOT NULL, "
            "created_at DATETIME, updated_at DATETIME)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE image_records ("
            "id VARCHAR(64) PRIMARY KEY, user_id INTEGER, "
            "position_index INTEGER NOT NULL, data TEXT NOT NULL, "
            "created_at VARCHAR(32), updated_at DATETIME)"
        )
        connection.exec_driver_sql(
            "INSERT INTO users (username, password_hash, role_key) VALUES "
            "('admin', 'hash-one', 'admin'), ('admin', 'hash-two', 'admin')"
        )
    legacy_store = DatabaseStore(engine)

    with pytest.raises(ValueError, match="duplicate values"):
        legacy_store.initialize_image_ownership("admin")
```

Add this startup regression to `tests/test_application_structure.py`:

```python
def test_runtime_backfills_legacy_image_owners(tmp_path: Path) -> None:
    database_path = tmp_path / "ownership.sqlite3"
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{database_path}",
        admin_user="admin",
        admin_pass="admin-secret",
    )
    engine = create_engine(settings.db_url)
    store = DatabaseStore(engine)
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("admin", "admin-secret", "admin")
    store.replace_records([{"id": "legacy"}])
    engine.dispose()

    runtime = create_runtime(settings)
    admin_id = runtime.store.get_user_by_username("admin")["id"]

    assert runtime.store.read_records(owner_user_id=admin_id) == [{"id": "legacy"}]
    dispose_runtime(runtime)
```

Import `create_runtime` and `dispose_runtime` from their existing modules in the test file.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_database_store.py::test_initialize_image_ownership_backfills_only_null_rows tests/test_database_store.py::test_initialize_image_ownership_rejects_an_orphaned_owner tests/test_database_store.py::test_initialize_image_ownership_rejects_duplicate_legacy_usernames tests/test_application_structure.py::test_runtime_backfills_legacy_image_owners -q
```

Expected: FAIL because startup does not initialize image ownership.

- [ ] **Step 3: Implement idempotent backfill and integrity validation**

Add this store method:

```python
def initialize_image_ownership(self, admin_username: str) -> int:
    with self.engine.begin() as connection:
        duplicate = connection.execute(
            select(self.users.c.username)
            .group_by(self.users.c.username)
            .having(func.count(self.users.c.id) > 1)
            .limit(1)
        ).scalar_one_or_none()
        if duplicate is not None:
            raise ValueError("users.username contains duplicate values")

        admin_ids = list(
            connection.execute(
                select(self.users.c.id).where(
                    self.users.c.username == admin_username
                )
            ).scalars()
        )
        if len(admin_ids) != 1:
            raise ValueError("configured administrator could not be resolved")

        result = connection.execute(
            update(self.image_records)
            .where(self.image_records.c.user_id.is_(None))
            .values(user_id=int(admin_ids[0]))
        )
        orphan = connection.execute(
            select(self.image_records.c.id)
            .outerjoin(self.users, self.users.c.id == self.image_records.c.user_id)
            .where(self.users.c.id.is_(None))
            .limit(1)
        ).scalar_one_or_none()
        if orphan is not None:
            raise ValueError(f"image record has an invalid owner: {orphan}")
        return int(result.rowcount or 0)
```

Call it after administrator seeding in `create_runtime()`:

```python
runtime.store = DatabaseStore(runtime.engine)
runtime.store.create_schema()
seed_database_defaults(runtime.store, settings)
runtime.store.initialize_image_ownership(settings.admin_user)
```

Expand the initialization exception handling without exposing secrets:

```python
except (SQLAlchemyError, ValueError) as exc:
    runtime.db_error = type(exc).__name__
    runtime.store = None
    dispose_engine(runtime.engine)
    error = RuntimeError("Database initialization failed")
    setattr(error, "runtime", runtime)
    raise error from exc
```

- [ ] **Step 4: Run startup and database tests**

Run:

```powershell
python -m pytest tests/test_database_store.py tests/test_application_structure.py::test_runtime_backfills_legacy_image_owners tests/test_application_structure.py::test_application_lifespan_disposes_sqlite_engine tests/test_application_structure.py::test_failed_runtime_creation_disposes_sqlite_engine -q
```

Expected: all selected tests PASS; the second initialization updates zero rows.

- [ ] **Step 5: Commit the ownership migration**

```powershell
git add database_store.py trace_app/database/connection.py tests/test_database_store.py tests/test_application_structure.py
git commit -m "feat: backfill historical image owners"
```

### Task 4: Validate Login Tokens into Current Users

**Files:**

- Modify: `trace_app/auth/schemas.py`
- Modify: `trace_app/auth/service.py:10-105`
- Modify: `trace_app/runtime.py:13-18`
- Modify: `trace_app/dependencies.py:1-31`
- Modify: `trace_app/application.py:140-165`
- Modify: `trace_app/compat.py:221-230`
- Modify: `trace_app/compat.py:317-319`
- Test: `tests/test_application_structure.py:581-690`
- Test: `tests/test_application_structure.py:1335-1350`

- [ ] **Step 1: Write failing session and deletion-integrity tests**

Add to `tests/test_application_structure.py`:

```python
def test_auth_service_resolves_tokens_to_current_database_identity() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("alice", "secret", "operator")
    sessions: dict[str, int] = {}
    service = AuthService(Repository(store), sessions=sessions)

    login = service.login("alice", "secret")
    current = service.resolve_token(login["token"])

    assert current.id == store.get_user_by_username("alice")["id"]
    assert current.username == "alice"
    assert current.role == "operator"
    store.update_user_role("alice", "viewer")
    assert service.resolve_token(login["token"]).role == "viewer"

    with pytest.raises(HTTPException) as invalid:
        service.resolve_token("missing-token")
    assert (invalid.value.status_code, invalid.value.detail) == (
        401,
        "登录已失效，请重新登录",
    )


def test_auth_service_refuses_to_delete_an_image_owner() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("alice", "secret", "operator")
    user_id = store.get_user_by_username("alice")["id"]
    store.insert_record({"id": "owned"}, owner_user_id=user_id)
    service = AuthService(Repository(store))

    with pytest.raises(HTTPException) as conflict:
        service.delete_user("alice")

    assert (conflict.value.status_code, conflict.value.detail) == (
        409,
        "该用户仍有图片，无法删除",
    )
    assert store.get_user_by_username("alice") is not None
```

Extend `test_runtime_starts_without_database_state` with:

```python
assert runtime.auth_sessions == {}
```

- [ ] **Step 2: Run the session tests and verify RED**

Run:

```powershell
python -m pytest tests/test_application_structure.py::test_auth_service_resolves_tokens_to_current_database_identity tests/test_application_structure.py::test_auth_service_refuses_to_delete_an_image_owner tests/test_application_structure.py::test_runtime_starts_without_database_state -q
```

Expected: FAIL because sessions, numeric identity projection, and owner deletion checks do not exist.

- [ ] **Step 3: Add the authenticated-user type and shared session state**

Add to `trace_app/auth/schemas.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: int
    username: str
    role: str
```

Add to `Runtime`:

```python
auth_sessions: dict[str, int] = field(default_factory=dict)
```

- [ ] **Step 4: Implement session issuance, resolution, and delete protection**

Update the `AuthService` constructor and login flow:

```python
def __init__(
    self,
    repository: Repository | None = None,
    *,
    sessions: dict[str, int] | None = None,
) -> None:
    self.repository = repository
    self.sessions = {} if sessions is None else sessions

def login(self, username: str, password: str) -> dict[str, Any]:
    repository = self._require_repository()
    identity = repository.authenticate_user(username, password)
    if identity is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    roles = repository.read_roles()["roles"]
    role = str(identity["role"])
    menus = self.allowed_menu_keys(roles.get(role, {}).get("menus", []))
    token = f"local-{uuid.uuid4().hex}"
    self.sessions[token] = int(identity["id"])
    return {
        "token": token,
        "username": str(identity["username"]),
        "role": role,
        "menus": menus,
    }

def resolve_token(self, token: str) -> AuthenticatedUser:
    user_id = self.sessions.get(token)
    identity = (
        None
        if user_id is None
        else self._require_repository().get_user_by_id(user_id)
    )
    if identity is None:
        self.sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return AuthenticatedUser(
        id=int(identity["id"]),
        username=str(identity["username"]),
        role=str(identity["role"]),
    )
```

Import `AuthenticatedUser` from `trace_app.auth.schemas`.

Protect user deletion before calling the existing store delete:

```python
def delete_user(self, username: str) -> dict[str, Any]:
    repository = self._require_repository()
    identity = repository.get_user_by_username(username)
    if identity is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if repository.user_has_images(int(identity["id"])):
        raise HTTPException(status_code=409, detail="该用户仍有图片，无法删除")
    repository.delete_user(username)
    return {
        "users": repository.list_users(),
        "roles": repository.read_roles()["roles"],
    }
```

- [ ] **Step 5: Add the Bearer-token dependency and share sessions across factories**

Add to `trace_app/dependencies.py`:

```python
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trace_app.auth.schemas import AuthenticatedUser


bearer_auth = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_auth),
    service: AuthService = Depends(get_auth_service),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="请先登录")
    return service.resolve_token(credentials.credentials)
```

Wire each production and compatibility service to the same dictionary:

```python
app.state.auth_service = AuthService(
    repository,
    sessions=runtime.auth_sessions,
)
```

```python
def get_auth_service() -> AuthService:
    return AuthService(repository, sessions=runtime.auth_sessions)
```

Use the same constructor in `_sync_application_state()`.

- [ ] **Step 6: Run authentication and application-structure tests**

Run:

```powershell
python -m pytest tests/test_application_structure.py -q
```

Expected: all tests PASS; existing login response fields remain unchanged and auth factories share runtime sessions.

- [ ] **Step 7: Commit server-side session identity**

```powershell
git add trace_app/auth/schemas.py trace_app/auth/service.py trace_app/runtime.py trace_app/dependencies.py trace_app/application.py trace_app/compat.py tests/test_application_structure.py
git commit -m "feat: resolve login tokens to database users"
```

### Task 5: Scope Image Management and Deletion

**Files:**

- Create: `tests/test_image_ownership_api.py`
- Modify: `trace_app/management/service.py:73-114`
- Modify: `trace_app/api/images.py:1-24`

- [ ] **Step 1: Create the failing image-visibility integration tests**

Create `tests/test_image_ownership_api.py` with this foundation and list test:

```python
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from trace_app.application import create_app
from trace_app.config import Settings


def login_headers(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture
def ownership_context(tmp_path: Path):
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{tmp_path / 'ownership.sqlite3'}",
        admin_user="admin",
        admin_pass="admin-secret",
    )
    app = create_app(settings=settings, initialize_database=True)
    store = app.state.runtime.store
    store.create_user("alice", "alice-secret", "operator")
    store.create_user("bob", "bob-secret", "viewer")
    ids = {
        username: store.get_user_by_username(username)["id"]
        for username in ("admin", "alice", "bob")
    }
    store.insert_record(
        {"id": "admin-image", "status": "保护中", "user_id": "payload-admin"},
        owner_user_id=ids["admin"],
    )
    store.insert_record(
        {"id": "alice-image", "status": "保护中", "user_id": "payload-alice"},
        owner_user_id=ids["alice"],
    )
    store.insert_record(
        {"id": "bob-image", "status": "泄露预警", "user_id": "payload-bob"},
        owner_user_id=ids["bob"],
    )
    with TestClient(app) as client:
        yield client, store, settings, ids


def test_image_list_requires_login_and_scopes_items_and_stats(ownership_context):
    client, _store, _settings, _ids = ownership_context

    assert client.get("/api/images").status_code == 401
    alice = client.get(
        "/api/images",
        headers=login_headers(client, "alice", "alice-secret"),
    )
    admin = client.get(
        "/api/images",
        headers=login_headers(client, "admin", "admin-secret"),
    )

    assert [item["id"] for item in alice.json()["items"]] == ["alice-image"]
    assert alice.json()["stats"]["total"] == 1
    assert alice.json()["stats"]["protected"] == 1
    assert alice.json()["stats"]["leaks"] == 0
    assert {item["id"] for item in admin.json()["items"]} == {
        "admin-image",
        "alice-image",
        "bob-image",
    }
    assert admin.json()["stats"]["total"] == 3
```

- [ ] **Step 2: Add the failing deletion and user-deletion test**

Append:

```python
def test_non_admin_cannot_delete_another_users_image_or_files(ownership_context):
    client, store, settings, ids = ownership_context
    protected_file = settings.watermarked_dir / "bob.png"
    protected_file.write_bytes(b"bob-image")
    store.delete_record("bob-image", owner_user_id=ids["bob"])
    store.insert_record(
        {
            "id": "bob-image",
            "status": "泄露预警",
            "download_url": "/uploads/watermarked/bob.png",
        },
        owner_user_id=ids["bob"],
    )
    alice_headers = login_headers(client, "alice", "alice-secret")

    forbidden = client.delete("/api/images/bob-image", headers=alice_headers)

    assert forbidden.status_code == 404
    assert store.read_records(owner_user_id=ids["bob"])[0]["id"] == "bob-image"
    assert protected_file.read_bytes() == b"bob-image"
    assert client.delete("/api/users/bob").status_code == 409


def test_non_admin_can_delete_own_image(ownership_context):
    client, store, _settings, ids = ownership_context

    response = client.delete(
        "/api/images/alice-image",
        headers=login_headers(client, "alice", "alice-secret"),
    )

    assert response.json() == {"deleted": True}
    assert store.read_records(owner_user_id=ids["alice"]) == []
```

- [ ] **Step 3: Run the integration tests and verify RED**

Run:

```powershell
python -m pytest tests/test_image_ownership_api.py -q
```

Expected: FAIL because the routes do not require identity and the management service returns/deletes global records.

- [ ] **Step 4: Filter the management service by current user**

Import `AuthenticatedUser`, then change the two service methods:

```python
def list_images(self, current_user: AuthenticatedUser) -> dict[str, Any]:
    owner_user_id = None if current_user.role == "admin" else current_user.id
    records = self.repository.read_records(owner_user_id=owner_user_id)
    protected = sum(1 for item in records if item.get("status") == "保护中")
    leaks = sum(1 for item in records if item.get("status") == "泄露预警")
    hits = sum(1 for item in records if item.get("status") == "溯源命中")
    detection_stats = self._read_detection_stats()
    attempts = detection_stats["attempts"]
    successes = detection_stats["successes"]
    success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
    return {
        "items": records,
        "stats": {
            "total": len(records),
            "protected": protected,
            "leaks": leaks,
            "hits": hits,
            "today": self._today_count(records),
            "detection_attempts": attempts,
            "detection_successes": successes,
            "detection_success_rate": success_rate,
        },
        "db_enabled": self.database_enabled,
        "db_ready": self._database_ready(),
        "db_error": self._database_error(),
        "db_url": self._masked_db_url(),
    }

def delete_image(
    self, image_id: str, current_user: AuthenticatedUser
) -> dict[str, bool]:
    owner_user_id = None if current_user.role == "admin" else current_user.id
    target = self.repository.delete_record(
        image_id,
        owner_user_id=owner_user_id,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    for key in ("original_url", "download_url", "thumbnail_url"):
        value = target.get(key)
        if value and value.startswith("/uploads/"):
            path = self.settings.upload_dir / value.replace("/uploads/", "")
            if path.exists():
                path.unlink()
    return {"deleted": True}
```

- [ ] **Step 5: Require identity in image routes**

Update `trace_app/api/images.py`:

```python
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service


@router.get("")
def list_images(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    return service.list_images(current_user)


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.delete_image(image_id, current_user)
```

- [ ] **Step 6: Run image-management tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_image_ownership_api.py tests/test_application_structure.py -q
```

Expected: all selected tests PASS; unauthorized deletion leaves both the row and file intact.

- [ ] **Step 7: Commit image-management isolation**

```powershell
git add tests/test_image_ownership_api.py trace_app/management/service.py trace_app/api/images.py
git commit -m "feat: scope image management by owner"
```

### Task 6: Assign Upload Ownership from the Login Session

**Files:**

- Modify: `trace_app/api/watermark.py:16-62`
- Modify: `trace_app/watermark/service.py:134-183`
- Modify: `trace_app/watermark/service.py:378`
- Modify: `tests/test_image_ownership_api.py`
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `tests/test_false_positive_gate.py`
- Modify: `tests/commercial_benchmark_config.py`
- Modify: `tests/commercial_attack_benchmark.py`
- Modify: `tests/commercial_quality_benchmark.py`
- Modify: `tests/commercial_negative_benchmark.py`
- Modify: `tests/commercial_trace_benchmark.py`
- Modify: `tests/video_platform_smoke_benchmark.py`

- [ ] **Step 1: Write the failing authenticated-upload ownership test**

Append to `tests/test_image_ownership_api.py`:

```python
def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (512, 384), (80, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_uses_authenticated_owner_without_replacing_payload_user_id(
    ownership_context,
):
    client, store, _settings, ids = ownership_context

    response = client.post(
        "/api/watermark/embed",
        headers=login_headers(client, "alice", "alice-secret"),
        files={"file": ("source.png", png_bytes(), "image/png")},
        data={
            "user_id": "watermark-alias",
            "robust_watermark_version": "1",
            "copyright_enabled": "false",
            "small_crop_trace_enabled": "false",
            "dot_matrix_trace_enabled": "false",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == "watermark-alias"
    owned_ids = {
        item["id"] for item in store.read_records(owner_user_id=ids["alice"])
    }
    assert response.json()["id"] in owned_ids


def test_upload_requires_login(ownership_context):
    client, _store, _settings, _ids = ownership_context
    response = client.post(
        "/api/watermark/embed",
        files={"file": ("source.png", png_bytes(), "image/png")},
        data={"user_id": "untrusted"},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run the upload tests and verify RED**

Run:

```powershell
python -m pytest tests/test_image_ownership_api.py::test_upload_uses_authenticated_owner_without_replacing_payload_user_id tests/test_image_ownership_api.py::test_upload_requires_login -q
```

Expected: FAIL because embed is unauthenticated and persists no relational owner.

- [ ] **Step 3: Pass authenticated ownership through the route and service**

Add the current-user dependency to `embed_watermark`:

```python
current_user: AuthenticatedUser = Depends(get_current_user),
service: WatermarkService = Depends(get_watermark_service),
```

Pass the numeric ID separately:

```python
return await service.embed(
    file=file,
    owner_user_id=current_user.id,
    user_id=user_id,
    mode=mode,
    copyright_enabled=copyright_enabled,
    copyright_text=copyright_text,
    copyright_opacity=copyright_opacity,
    copyright_complexity=copyright_complexity,
    copyright_irregular_enabled=copyright_irregular_enabled,
    copyright_prominent_corner_enabled=copyright_prominent_corner_enabled,
    fidelity_level=fidelity_level,
    robust_watermark_strength=robust_watermark_strength,
    robust_watermark_version=robust_watermark_version,
    small_crop_trace_enabled=small_crop_trace_enabled,
    small_crop_trace_strength=small_crop_trace_strength,
    small_crop_trace_density=small_crop_trace_density,
    dot_matrix_trace_enabled=dot_matrix_trace_enabled,
    dot_matrix_trace_strength=dot_matrix_trace_strength,
)
```

Add `owner_user_id: int` beside `user_id: str` in `WatermarkService.embed()`, then change persistence to:

```python
self.repository.add_record(record, owner_user_id=owner_user_id)
```

Do not modify payload construction; its existing `"user_id": user_id` remains the watermark alias.

- [ ] **Step 4: Authenticate existing embed API tests**

In `tests/test_watermark_v4_api.py`, clear sessions in the autouse fixture and add:

```python
main.runtime.auth_sessions.clear()


def _authenticated_client() -> TestClient:
    client = TestClient(main.app)
    response = client.post(
        "/auth/login",
        data={"username": "test-admin", "password": "admin-password"},
    )
    assert response.status_code == 200, response.text
    client.headers["Authorization"] = f"Bearer {response.json()['token']}"
    return client
```

Replace clients used for embed calls with `_authenticated_client()`. Add this ownership assertion after one successful V4 embed:

```python
record = response.json()
owner_id = main.require_store().get_user_by_username("test-admin")["id"]
assert record in main.require_store().read_records(owner_user_id=owner_id)
```

In `tests/test_false_positive_gate.py`, seed `test-admin`, clear sessions during reset, and use the same authenticated-client pattern before every embed.

- [ ] **Step 5: Authenticate benchmark clients without embedding credentials**

Add to `tests/commercial_benchmark_config.py`:

```python
def authenticate_client(client, username: str, password: str) -> None:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )
    response.raise_for_status()
    client.headers["Authorization"] = f"Bearer {response.json()['token']}"
```

Import it in each listed benchmark module and call this immediately after constructing the client:

```python
client = TestClient(main.app)
authenticate_client(client, main.ADMIN_USER, main.ADMIN_PASS)
```

Use environment-backed `main.ADMIN_USER` and `main.ADMIN_PASS`; do not add literal credentials.

- [ ] **Step 6: Run protected upload and benchmark contract tests**

Run:

```powershell
python -m pytest tests/test_image_ownership_api.py tests/test_watermark_v4_api.py tests/test_false_positive_gate.py tests/test_commercial_benchmark_config.py -q
```

Expected: all selected tests PASS. Existing watermark payload assertions remain unchanged while database owner assertions use numeric user IDs.

- [ ] **Step 7: Commit authenticated upload ownership**

```powershell
git add trace_app/api/watermark.py trace_app/watermark/service.py tests/test_image_ownership_api.py tests/test_watermark_v4_api.py tests/test_false_positive_gate.py tests/commercial_benchmark_config.py tests/commercial_attack_benchmark.py tests/commercial_quality_benchmark.py tests/commercial_negative_benchmark.py tests/commercial_trace_benchmark.py tests/video_platform_smoke_benchmark.py
git commit -m "feat: assign image owner from login session"
```

### Task 7: Send and Invalidate Frontend Login Sessions

**Files:**

- Create: `frontend/src/auth-session.js`
- Modify: `frontend/src/api/client.js:1-34`
- Modify: `frontend/src/api/trace.js:10-15`
- Modify: `frontend/src/state/app.js:1-42`
- Modify: `frontend/src/App.vue:1-22`
- Modify: `frontend/tests/api-contract.test.js`
- Modify: `frontend/tests/app-state.test.js`

- [ ] **Step 1: Write failing Bearer and invalidation tests**

Add these tests to `frontend/tests/api-contract.test.js`:

```javascript
test('authenticated requests send the cached login token', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'local-session' }));
  const fetchMock = vi.fn().mockResolvedValue(okJson({ items: [] }));
  vi.stubGlobal('fetch', fetchMock);

  await listImages();

  expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer local-session');
});

test('login never sends a stale bearer token', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'stale' }));
  const fetchMock = vi.fn().mockResolvedValue(okJson({ token: 'fresh' }));
  vi.stubGlobal('fetch', fetchMock);

  await login('admin', 'secret');

  expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined();
});

test('an authenticated 401 clears login state and emits invalidation', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'expired' }));
  const listener = vi.fn();
  window.addEventListener('trace:authentication-invalid', listener, { once: true });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
    JSON.stringify({ detail: '登录已失效，请重新登录' }),
    { status: 401 },
  )));

  await expect(listImages()).rejects.toThrow('登录已失效，请重新登录');

  expect(localStorage.getItem('currentUser')).toBeNull();
  expect(listener).toHaveBeenCalledOnce();
});
```

- [ ] **Step 2: Run the frontend tests and verify RED**

Run:

```powershell
npm --prefix frontend test -- --run frontend/tests/api-contract.test.js frontend/tests/app-state.test.js
```

Expected: FAIL because the client neither sends tokens nor invalidates stale login state.

- [ ] **Step 3: Create the shared auth-session module**

Create `frontend/src/auth-session.js`:

```javascript
export const CURRENT_USER_KEY = 'currentUser';
export const AUTH_INVALID_EVENT = 'trace:authentication-invalid';

export function readStoredUser() {
  try {
    const saved = localStorage.getItem(CURRENT_USER_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

export function writeStoredUser(user) {
  localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
}

export function clearStoredUser() {
  localStorage.removeItem(CURRENT_USER_KEY);
}

export function bearerHeaders() {
  const token = readStoredUser()?.token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function invalidateAuthentication() {
  clearStoredUser();
  window.dispatchEvent(new Event(AUTH_INVALID_EVENT));
}

export function subscribeToAuthenticationInvalidation(handler) {
  window.addEventListener(AUTH_INVALID_EVENT, handler);
  return () => window.removeEventListener(AUTH_INVALID_EVENT, handler);
}
```

- [ ] **Step 4: Attach tokens and handle HTTP 401 in the API client**

Replace `request()` with:

```javascript
import { bearerHeaders, invalidateAuthentication } from '../auth-session.js';


export async function request(path, options = {}) {
  const { authenticated = true, ...fetchOptions } = options;
  const headers = {
    ...(fetchOptions.headers || {}),
    ...(authenticated ? bearerHeaders() : {}),
  };
  let response;
  try {
    response = await fetch(path, { ...fetchOptions, headers });
  } catch (error) {
    throw new Error(error?.message || '网络请求失败');
  }

  const body = await parseJson(response);
  if (!response.ok) {
    if (authenticated && response.status === 401) invalidateAuthentication();
    throw new Error(body?.detail || body?.message || fallbackMessage(response.status));
  }
  if (body === undefined) {
    throw new Error('响应格式无效');
  }
  return body;
}
```

The `client.js` import path is exactly `../auth-session.js` for the planned file location.

Mark login unauthenticated:

```javascript
return request('/auth/login', { method: 'POST', body, authenticated: false });
```

- [ ] **Step 5: Use shared storage and reopen login on invalidation**

In `frontend/src/state/app.js`, import the storage helpers and replace direct current-user storage calls:

```javascript
import {
  clearStoredUser,
  readStoredUser,
  writeStoredUser,
} from '../auth-session.js';
```

Use `currentUser: readStoredUser()`, `writeStoredUser(user)` in `setUser()`, and `clearStoredUser()` in `clearUser()`.

In `App.vue`, add lifecycle imports and subscribe once:

```javascript
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { subscribeToAuthenticationInvalidation } from './auth-session.js';

let stopAuthenticationInvalidation;
onMounted(() => {
  stopAuthenticationInvalidation = subscribeToAuthenticationInvalidation(
    () => state.clearUser(),
  );
});
onBeforeUnmount(() => stopAuthenticationInvalidation?.());
```

- [ ] **Step 6: Run frontend unit tests and build**

Run:

```powershell
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

Expected: all Vitest tests PASS and Vite rebuilds `assets/app/app.js` plus `assets/app/app.css` without warnings or missing imports.

- [ ] **Step 7: Commit frontend session integration**

```powershell
git add frontend/src/auth-session.js frontend/src/api/client.js frontend/src/api/trace.js frontend/src/state/app.js frontend/src/App.vue frontend/tests/api-contract.test.js frontend/tests/app-state.test.js assets/app/app.js assets/app/app.css
git commit -m "feat: send authenticated frontend requests"
```

### Task 8: Document and Verify the Complete Permission Boundary

**Files:**

- Modify: `README.md`
- Verify: all files changed in Tasks 1-7

- [ ] **Step 1: Document the authenticated image APIs**

Add after the API overview table in `README.md`:

```markdown
`POST /auth/login` returns a login token. Send it to protected endpoints as
`Authorization: Bearer <token>`. Image creation and image management require a
valid login. Administrators can list and delete all image records; other roles
can list and delete only records whose `image_records.user_id` matches their
current `users.id`.

The serialized watermark payload's `user_id` remains a trace label and is not
used for management authorization.
```

- [ ] **Step 2: Run focused backend verification**

Run:

```powershell
python -m pytest tests/test_database_store.py tests/test_application_structure.py tests/test_image_ownership_api.py tests/test_watermark_v4_api.py tests/test_false_positive_gate.py -q
```

Expected: PASS with no warnings or unexpected HTTP 401 responses.

- [ ] **Step 3: Run frontend verification**

Run:

```powershell
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

Expected: all Vitest tests PASS and the production bundle completes successfully.

- [ ] **Step 4: Run the full Python suite**

Run:

```powershell
python -m pytest -q
```

Expected: the complete suite PASS. Investigate any failure before proceeding; do not weaken authentication assertions to restore unrelated tests.

- [ ] **Step 5: Check formatting and the exact diff**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors. Only planned files plus the user's pre-existing deployment/release changes are present.

- [ ] **Step 6: Commit documentation and final regression adjustments**

```powershell
git add README.md
git commit -m "docs: describe image ownership permissions"
```

- [ ] **Step 7: Apply and verify the deployed backfill**

Start the application in a separate PowerShell window with the configured
database:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

After startup succeeds, run this read-only verification from the working
PowerShell window:

```powershell
@'
from sqlalchemy import create_engine, text
from trace_app.config import settings

engine = create_engine(settings.db_url, future=True)
with engine.connect() as connection:
    counts = connection.execute(text(
        "SELECT "
        "COUNT(*) AS total, "
        "SUM(CASE WHEN i.user_id IS NULL THEN 1 ELSE 0 END) AS unowned, "
        "SUM(CASE WHEN u.id IS NULL THEN 1 ELSE 0 END) AS invalid "
        "FROM image_records i LEFT JOIN users u ON u.id = i.user_id"
    )).mappings().one()
print({key: int(value or 0) for key, value in counts.items()})
engine.dispose()
'@ | python -
```

Expected for the currently observed database: `total` is `6`, `unowned` is `0`, and `invalid` is `0`.

- [ ] **Step 8: Smoke-test both roles through the running server**

Use one administrator and one operator account:

1. Log in as the operator, upload one image, and open Image Management.
2. Confirm the operator sees the new image and no administrator-owned historical images.
3. Log in as the administrator and confirm all seven images are visible.
4. Confirm an operator deletion request for an administrator image returns HTTP 404.

Expected: UI counts match each role's visible rows, the administrator sees the full set, and no unauthorized database or filesystem mutation occurs.
