# V4 Production Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the JSON-scanning legacy path with an authenticated, PostgreSQL/pgvector-backed, source-group-first V4 pipeline using the new 64-bit codec and opaque media addresses.

**Architecture:** Keep the existing users, roles, and administrator identity, but introduce a separate relational V4 domain whose repositories never call `read_records()`. Generation creates one source group and reusable model/geometry features; detection recalls source groups with DINOv2, confirms geometry, extracts one immutable A/B observation per group, and authenticates a record through an indexed `(source_group_id, auth_tag)` lookup. All expensive image and model work runs behind shared deadlines in isolated workers, and all client media access resolves opaque IDs through scoped signatures.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy Core, PostgreSQL 16+, pgvector, psycopg 3, Pillow, OpenCV, PyTorch/DINOv2, `kornia==0.8.1` SuperPoint/LightGlue, reedsolo, pytest, Vue 3/Vitest.

**Approved design:** `docs/superpowers/specs/2026-07-29-v4-production-pipeline-hardening-design.md`

---

## File Map And Execution Rules

New code is grouped by responsibility. Do not add the new behavior to `trace_app/compat.py`, `database_store.py`, or legacy modules under `trace_app/watermark/`; those remain compatibility code outside the new V4 path.

- `trace_app/v4/domain.py`: immutable IDs, records, outcomes, evidence, and error types shared across V4.
- `trace_app/v4/schema.py`: PostgreSQL/SQLite test table definitions and required index names.
- `trace_app/v4/repository.py`: owner-scoped, indexed V4 queries and atomic counters only.
- `trace_app/v4/keys.py`: key-ring loading, canonical HMAC message, active key selection.
- `trace_app/v4/media.py`: opaque media mappings, logical storage keys, scoped signatures, safe resolution.
- `trace_app/v4/deadlines.py`: shared monotonic deadlines and typed timeout/resource outcomes.
- `trace_app/v4/workers.py`: isolated execution, quotas, cancellation, and termination mapping.
- `trace_app/v4/models.py`: pinned model manifests, checksums, health, DINO/SuperPoint/LightGlue adapters.
- `trace_app/v4/features.py`: non-pickle feature serialization and validation.
- `trace_app/v4/recall.py`: multi-view construction, DINO batching, pgvector aggregation.
- `trace_app/v4/geometry.py`: ORB/RANSAC and bounded LightGlue confirmation.
- `trace_app/v4/generation.py`: transactional source-group creation and V4 record generation.
- `trace_app/v4/detection.py`: exact match, recall, geometry, one-warp extraction, indexed authentication.
- `trace_app/v4/jobs.py`: explicit deep-forensics job lifecycle.
- `trace_app/api/v4.py`: authenticated V4 generation/detection/job routes.
- `trace_app/api/media.py`: mapped media transfer route.
- `tools/initialize_v4.py`: offline preflight, backup verification, destructive reset, smoke tests, ready marker.

Every task below ends in a focused commit. Before staging, run `git status --short` and stage only the paths listed in that task because the working tree contains unrelated user changes. SQLite tests verify repository semantics only; PostgreSQL integration tests are mandatory for pgvector, query plans, locking, and release approval.

## Checkpoint 1: Contracts And Relational Storage

### Task 1: Add V4 Domain Contracts And Production Configuration

**Files:**
- Create: `trace_app/v4/__init__.py`
- Create: `trace_app/v4/domain.py`
- Modify: `trace_app/config.py`
- Modify: `.env.example`
- Test: `tests/v4/test_domain.py`
- Test: `tests/v4/test_settings.py`

- [ ] **Step 1: Write failing domain and settings tests**

```python
from pathlib import Path

import pytest

from trace_app.config import Settings
from trace_app.v4.domain import DetectionOutcome, OwnerScope


def test_v4_outcomes_are_closed_and_owner_scope_is_explicit() -> None:
    assert {item.value for item in DetectionOutcome} == {
        "success", "not_found", "ambiguous", "timeout",
        "resource_exhausted", "service_unavailable",
    }
    assert OwnerScope(user_id=7, cross_owner=False).query_owner_id == 7
    assert OwnerScope(user_id=7, cross_owner=True).query_owner_id is None


def test_production_v4_requires_postgresql_and_fixed_deadlines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings.from_values(
            base_dir=tmp_path, upload_dir="uploads", data_dir="data",
            db_url="sqlite+pysqlite:///:memory:", admin_user="admin",
            admin_pass="secret", environment="production",
        )
    settings = Settings.from_values(
        base_dir=tmp_path, upload_dir="uploads", data_dir="data",
        db_url="postgresql+psycopg://trace:test@db/trace",
        admin_user="admin", admin_pass="secret", environment="production",
    )
    assert settings.v4_sync_p95_seconds == 120
    assert settings.v4_sync_timeout_seconds == 300
    assert settings.v4_deep_timeout_seconds == 1000
```

- [ ] **Step 2: Run the tests and verify missing contracts fail**

Run: `pytest tests/v4/test_domain.py tests/v4/test_settings.py -q`

Expected: FAIL because `trace_app.v4` and the production settings fields do not exist.

- [ ] **Step 3: Implement the immutable contracts and validated settings**

Define `DetectionOutcome(str, Enum)`, `OwnerScope`, `V4Record`, `SourceGroup`, `DetectionEvidence`, and `DetectionResult` as frozen slot dataclasses. `OwnerScope.query_owner_id` returns `None` only for an explicit administrator cross-owner request. Extend `Settings.from_values()` with `environment`, model manifest path, worker quotas, `MEDIA_PUBLIC_BASE_URL`, and the fixed 120/300/1000 deadline defaults. Reject production URLs whose backend is not PostgreSQL; reject non-positive quotas and any altered hard/deep deadline that exceeds the approved values.

```python
class DetectionOutcome(str, Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass(frozen=True, slots=True)
class OwnerScope:
    user_id: int
    cross_owner: bool = False

    @property
    def query_owner_id(self) -> int | None:
        return None if self.cross_owner else self.user_id
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/v4/test_domain.py tests/v4/test_settings.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add trace_app/v4/__init__.py trace_app/v4/domain.py trace_app/config.py .env.example tests/v4/test_domain.py tests/v4/test_settings.py
git commit -m "feat: define hardened v4 domain contracts"
```

### Task 2: Create The Relational V4 Schema And pgvector Startup Gate

**Files:**
- Create: `trace_app/v4/schema.py`
- Create: `trace_app/v4/startup.py`
- Modify: `requirements.txt`
- Modify: `trace_app/database/connection.py`
- Test: `tests/v4/test_schema.py`
- Test: `tests/v4/test_postgres_schema.py`

- [ ] **Step 1: Write failing schema tests**

Assert the exact tables `source_groups`, `source_group_embeddings`, `source_group_features`, `v4_records`, `media_objects`, `auth_sessions`, `rate_limit_buckets`, `v4_counters`, `audit_events`, and `deep_forensics_jobs`. Assert named unique constraints `uq_source_group_owner_sha256`, `uq_v4_owner_trace`, and `uq_v4_group_auth_tag`, plus paired MD5/SHA-256 indexes and owner/vector indexes. The PostgreSQL test must query `pg_extension`, `pg_indexes`, and `pg_get_indexdef()` and require `vector_cosine_ops` on an HNSW index.

```python
def test_v4_record_authentication_is_a_concrete_unique_key(v4_tables) -> None:
    table = v4_tables.v4_records
    assert table.c.auth_tag.type.length == 8
    assert table.c.metadata_json.type.__class__.__name__ in {"JSON", "JSONB"}
    names = {constraint.name for constraint in table.constraints}
    assert "uq_v4_group_auth_tag" in names
```

- [ ] **Step 2: Verify the tests fail before the schema exists**

Run: `pytest tests/v4/test_schema.py -q`

Expected: FAIL importing `trace_app.v4.schema`.

- [ ] **Step 3: Define portable tables and PostgreSQL-only vector DDL**

Use SQLAlchemy Core. Store hashes and auth tags as fixed `LargeBinary`; use timezone-aware timestamps; use `CheckConstraint` for status, feature kind, and media variant. SQLite receives a `LargeBinary` embedding adapter for unit tests only. PostgreSQL startup executes `CREATE EXTENSION IF NOT EXISTS vector`, declares `vector(384)`, creates the HNSW cosine index, and verifies every required index by name and definition. Add pinned `pgvector`, `torch`, `torchvision`, `safetensors`, and `kornia==0.8.1`; all model weights remain local and checksum-verified.

- [ ] **Step 4: Run portable and PostgreSQL integration tests**

Run: `pytest tests/v4/test_schema.py -q`

Expected: PASS.

Run: `pytest tests/v4/test_postgres_schema.py -q -m postgres`

Expected: PASS against `TEST_POSTGRES_URL`; otherwise SKIP with the single explicit reason `TEST_POSTGRES_URL is not configured` and never count that skip as release approval.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add trace_app/v4/schema.py trace_app/v4/startup.py requirements.txt trace_app/database/connection.py tests/v4/test_schema.py tests/v4/test_postgres_schema.py
git commit -m "feat: add relational v4 pgvector schema"
```

### Task 3: Implement Indexed Owner-Scoped V4 Repositories

**Files:**
- Create: `trace_app/v4/repository.py`
- Test: `tests/v4/test_repository.py`
- Test: `tests/v4/test_query_plans.py`

- [ ] **Step 1: Write failing repository behavior tests**

Cover source-group upsert by `(owner_user_id, original_image_sha256)`, exact-file lookup by paired hashes, owner-scoped list/delete, indexed auth lookup, collision propagation, atomic counters, 100,000 versions in one group, and the absence of `read_records` calls. Inject a legacy repository spy whose `read_records()` raises `AssertionError` into every detection repository test.

```python
def test_auth_lookup_is_group_local_and_owner_scoped(v4_repo, seeded_groups) -> None:
    found = v4_repo.find_record_by_auth_tag(
        source_group_id=seeded_groups.alice.id,
        auth_tag=b"12345678",
        owner_user_id=seeded_groups.alice.owner_user_id,
    )
    assert found.trace_id == "TR-ALICE"
    assert v4_repo.find_record_by_auth_tag(
        source_group_id=seeded_groups.alice.id,
        auth_tag=b"12345678",
        owner_user_id=seeded_groups.bob.owner_user_id,
    ) is None
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/v4/test_repository.py -q`

Expected: FAIL because `V4Repository` does not exist.

- [ ] **Step 3: Implement narrow repository methods**

Expose only typed methods: `find_or_create_source_group`, `insert_embeddings`, `put_feature`, `insert_record`, `find_exact_file`, `recall_groups`, `find_record_by_auth_tag`, `list_records`, `delete_record`, `increment_counter`, `append_audit`, and deep-job CRUD. Every non-admin query accepts a non-null `owner_user_id` and applies it in SQL. `recall_groups` uses a window/aggregate query capped at 40 groups; no repository method parses `metadata_json` for filtering.

- [ ] **Step 4: Prove behavior and index use**

Run: `pytest tests/v4/test_repository.py -q`

Expected: PASS.

Run: `pytest tests/v4/test_query_plans.py -q -m postgres`

Expected: `EXPLAIN (FORMAT JSON)` contains index names for exact-file and group-auth queries and contains no sequential scan of `v4_records` at the release-scale fixture.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/repository.py tests/v4/test_repository.py tests/v4/test_query_plans.py
git commit -m "feat: add indexed owner scoped v4 repository"
```

**Checkpoint 1 gate:** portable schema/repository tests pass; PostgreSQL creates pgvector HNSW; query-plan tests use indexes; no V4 module imports `database_store.DatabaseStore` or calls `read_records()`.

## Checkpoint 2: Authentication, Media, Network, And Resource Boundaries

### Task 4: Replace Process Sessions With Hashed Database Sessions And Atomic Rate Limits

**Files:**
- Create: `trace_app/v4/security.py`
- Modify: `trace_app/auth/service.py`
- Modify: `trace_app/dependencies.py`
- Modify: `trace_app/api/auth.py`
- Modify: `trace_app/api/users.py`
- Modify: `trace_app/application.py`
- Test: `tests/v4/test_auth_security.py`
- Test: `tests/v4/test_api_authorization.py`

- [ ] **Step 1: Write failing security tests**

Require SHA-256 token hashes only, idle and absolute expiry, logout/password-change/admin revocation, atomic account+IP login buckets, indistinguishable login errors, admin-only users/roles, and authentication on all generation/detection/management routes. Assert `/api/dev/reset` is absent in production and anonymous requests are 401.

- [ ] **Step 2: Verify current behavior fails**

Run: `pytest tests/v4/test_auth_security.py tests/v4/test_api_authorization.py -q`

Expected: FAIL because sessions are stored in `Runtime.auth_sessions` and some routes lack required dependencies.

- [ ] **Step 3: Implement database-backed authentication**

Generate 32 random token bytes, return URL-safe base64 once, and persist only `sha256(token)`. Resolve and touch sessions in one transaction. Add `require_admin()` and `resolve_owner_scope()` dependencies; cross-owner scope is accepted only when `current_user.role == "admin"` and an explicit request flag is true. Use a PostgreSQL upsert for rate buckets and return the same 401 body for unknown users and wrong passwords.

- [ ] **Step 4: Run security tests**

Run: `pytest tests/v4/test_auth_security.py tests/v4/test_api_authorization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/security.py trace_app/auth/service.py trace_app/dependencies.py trace_app/api/auth.py trace_app/api/users.py trace_app/application.py tests/v4/test_auth_security.py tests/v4/test_api_authorization.py
git commit -m "fix: enforce database sessions and v4 authorization"
```

### Task 5: Implement Opaque Media IDs And Scoped Mapped URLs

**Files:**
- Create: `trace_app/v4/media.py`
- Create: `trace_app/api/media.py`
- Modify: `trace_app/application.py`
- Modify: `trace_app/api/images.py`
- Modify: `trace_app/api/watermark.py`
- Test: `tests/v4/test_opaque_media.py`

- [ ] **Step 1: Write failing opaque-address tests**

Assert API JSON, errors, audit rows, and captured logs contain no `/uploads/`, absolute path, drive letter, internal host, bucket, or logical storage key. Test signature binding to media ID, variant, owner/access scope, and expiry; reject replay across variants/users; reject traversal and symlink escape. The public form must be `/api/media/{opaque_id}?expires=...&signature=...` or the same path under `MEDIA_PUBLIC_BASE_URL`.

- [ ] **Step 2: Verify failure against existing `/uploads` mapping**

Run: `pytest tests/v4/test_opaque_media.py -q`

Expected: FAIL because existing signed URLs reveal the storage layout.

- [ ] **Step 3: Implement mapping and route**

Use 128-bit random URL-safe media IDs. Persist `{id, owner_user_id, variant, storage_key, content_type, byte_size, sha256}`; generate storage keys from a random object digest under a fixed variant prefix. Sign the canonical length-prefixed tuple `(version, media_id, variant, scope, expires)`. Resolve the configured storage root, reject links in every path component, verify ownership before returning `FileResponse` or an internal-redirect header, and configure log filters to omit resolved paths.

- [ ] **Step 4: Run media tests**

Run: `pytest tests/v4/test_opaque_media.py tests/test_media_security.py -q`

Expected: new opaque tests PASS; update legacy assertions only where the approved API changed, without weakening traversal or SSRF checks.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/media.py trace_app/api/media.py trace_app/application.py trace_app/api/images.py trace_app/api/watermark.py tests/v4/test_opaque_media.py tests/test_media_security.py
git commit -m "feat: hide storage paths behind opaque media ids"
```

### Task 6: Add Safe Streaming, Pinned Remote Fetching, Deadlines, And Isolated Workers

**Files:**
- Create: `trace_app/v4/deadlines.py`
- Create: `trace_app/v4/workers.py`
- Create: `trace_app/v4/uploads.py`
- Modify: `trace_app/imaging/io.py`
- Test: `tests/v4/test_deadlines.py`
- Test: `tests/v4/test_workers.py`
- Test: `tests/v4/test_streaming_and_ssrf.py`

- [ ] **Step 1: Write failing boundary tests**

Test chunked byte quotas, temporary-disk quotas, no fixed decoded-pixel rejection, public HTTP(S) only, proxy environment ignored, redirect revalidation, DNS answer pinned to the connected IP, response limits, worker memory/CPU/concurrency cancellation, child termination at 300 seconds, and deep-job termination at 1,000 seconds. A worker limit must map to `resource_exhausted`, never `not_found`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_deadlines.py tests/v4/test_workers.py tests/v4/test_streaming_and_ssrf.py -q`

Expected: FAIL because the shared deadline and isolated worker contracts do not exist.

- [ ] **Step 3: Implement the boundaries**

`Deadline` stores an absolute monotonic end, exposes `remaining()` and `check(stage)`, and cannot be extended by children. Stream uploads in 1 MiB chunks to a private temp directory. Spawn workers with the platform-supported process isolation; on Linux apply `RLIMIT_AS`, `RLIMIT_CPU`, private temp quotas, and process-group termination. Use an explicit connection implementation that connects to the validated IP while sending the original Host/SNI, disables inherited proxies, and repeats validation for every redirect.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/v4/test_deadlines.py tests/v4/test_workers.py tests/v4/test_streaming_and_ssrf.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/deadlines.py trace_app/v4/workers.py trace_app/v4/uploads.py trace_app/imaging/io.py tests/v4/test_deadlines.py tests/v4/test_workers.py tests/v4/test_streaming_and_ssrf.py
git commit -m "feat: isolate v4 image processing resources"
```

**Checkpoint 2 gate:** all V4 routes authenticate; authorization is applied in SQL; media addresses are opaque; SSRF and resource tests pass; no synchronous child survives its request deadline.

## Checkpoint 3: Hardened Codec And Model Feature Layer

### Task 7: Replace The 32-bit Payload With Source-Bound HMAC64 And RS(16,8)

**Files:**
- Modify: `watermark_v4/payload.py`
- Modify: `watermark_v4/__init__.py`
- Create: `trace_app/v4/keys.py`
- Test: `tests/v4/test_hmac64_payload.py`
- Modify: `tests/test_watermark_v4_payload.py`

- [ ] **Step 1: Write failing payload tests**

Test canonical length-prefix encoding for `codec_version`, `key_id`, `owner_user_id`, 32-byte source pixel SHA-256, and `trace_id`; HMAC-SHA256 truncation to exactly 8 bytes; domain separation; key/source/owner changes; RS(16,8) correction and erasure limits; constant-time record verification; and group-local collision retry.

```python
def test_auth_tag_is_64_bit_and_source_bound() -> None:
    context = AuthContext("v4", "key-2026-07", 9, b"s" * 32, "TR-1")
    tag = authentication_tag(context, b"k" * 32)
    assert len(tag) == 8
    assert tag != authentication_tag(replace(context, owner_user_id=10), b"k" * 32)
    assert len(encode_codeword(tag)) == 16
```

- [ ] **Step 2: Verify old 32-bit implementation fails**

Run: `pytest tests/v4/test_hmac64_payload.py -q`

Expected: FAIL because the current tag/codeword lengths are 4/8 bytes.

- [ ] **Step 3: Implement the approved codec contract**

Set codec ID `hmac64_rs_16_8_split_repeat_sync_v4`, tag/data/parity/codeword sizes to 8/8/8/16 bytes, and define a stable binary encoder using an ASCII domain prefix plus unsigned big-endian field lengths. Remove candidate-count probability as an authentication mechanism. `KeyRing` loads non-secret key IDs plus secret bytes from the configured secret source, requires one active key, and never exposes key material through repr/logging.

- [ ] **Step 4: Run payload tests**

Run: `pytest tests/v4/test_hmac64_payload.py tests/test_watermark_v4_payload.py -q`

Expected: PASS after replacing obsolete 32-bit expectations rather than retaining a compatibility branch.

- [ ] **Step 5: Commit**

```bash
git add watermark_v4/payload.py watermark_v4/__init__.py trace_app/v4/keys.py tests/v4/test_hmac64_payload.py tests/test_watermark_v4_payload.py
git commit -m "feat: replace v4 payload with source bound hmac64"
```

### Task 8: Implement Split A/B Checkerboard Embedding And Immutable Observation

**Files:**
- Modify: `watermark_v4/config.py`
- Modify: `watermark_v4/dct.py`
- Modify: `watermark_v4/detector.py`
- Create: `watermark_v4/observation.py`
- Test: `tests/v4/test_split_carrier.py`
- Modify: `tests/test_watermark_v4_dct.py`
- Modify: `tests/test_watermark_v4_detector.py`

- [ ] **Step 1: Write failing split-carrier tests**

Assert `(tile_x + tile_y) % 2` selects A/B, A carries logical bits 0..63, B carries 64..127, neighbors differ, both DCT coefficient pairs redundantly carry the same physical bit, deterministic phases distribute each half, and decoding rejects missing class/phase/coverage gates. Assert `extract_observation()` runs once per warped group and returns frozen tuples for 16 observed bytes, byte confidences, class evidence, and timing.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_split_carrier.py -q`

Expected: FAIL because the existing carrier repeats one 64-bit codeword in every tile.

- [ ] **Step 3: Implement split carrier in bounded tile batches**

Replace full-image temporary block arrays with row/tile batches. Embed or aggregate each checkerboard class independently, inverse-permute within its half, concatenate A+B into a 128-bit observation, and RS-decode blindly to an 8-byte tag. Preserve evidence fields `tile_counts`, `phase_counts`, `coverage`, `corrected_symbols`, `erasures`, `bit_errors`, `signal_score`, `sync_confidence`, and `elapsed_seconds`; none may produce success without HMAC verification.

- [ ] **Step 4: Run codec and image tests**

Run: `pytest tests/v4/test_split_carrier.py tests/test_watermark_v4_dct.py tests/test_watermark_v4_detector.py -q`

Expected: PASS with no legacy-codec fallback.

- [ ] **Step 5: Commit**

```bash
git add watermark_v4/config.py watermark_v4/dct.py watermark_v4/detector.py watermark_v4/observation.py tests/v4/test_split_carrier.py tests/test_watermark_v4_dct.py tests/test_watermark_v4_detector.py
git commit -m "feat: split v4 rs codeword across checkerboard tiles"
```

### Task 9: Add Pinned Model Registry And Strict Feature Serialization

**Files:**
- Create: `trace_app/v4/models.py`
- Create: `trace_app/v4/features.py`
- Create: `models/v4-models.example.json`
- Test: `tests/v4/test_model_registry.py`
- Test: `tests/v4/test_feature_serialization.py`

- [ ] **Step 1: Write failing model and serialization tests**

Require manifest name/version/SHA-256/input/output metadata; fail on missing or replaced weights; batch DINO output shape `(n, 384)` normalized to unit length; SuperPoint point/descriptor and LightGlue match shape validation. Reject pickle/object dtype, decompression bombs, wrong dtype, non-finite values, count overflow, bad checksum, and schema/model version mismatch.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_model_registry.py tests/v4/test_feature_serialization.py -q`

Expected: FAIL because no backend model registry exists.

- [ ] **Step 3: Implement strict adapters**

Use a JSON manifest whose own path comes from settings and whose weight entries contain exact SHA-256 values. Load DINOv2 and Kornia SuperPoint/LightGlue weights with `safetensors`, call `eval()` and inference mode, and expose health only after deterministic smoke inference. Serialize arrays into a versioned binary envelope containing explicit dtype, rank, shape, byte length, payload SHA-256, and compressed raw bytes; validate declared limits before allocating or decompressing.

- [ ] **Step 4: Run tests**

Run: `pytest tests/v4/test_model_registry.py tests/v4/test_feature_serialization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/models.py trace_app/v4/features.py models/v4-models.example.json tests/v4/test_model_registry.py tests/v4/test_feature_serialization.py
git commit -m "feat: add verified v4 model and feature layer"
```

**Checkpoint 3 gate:** exact codec vectors pass; old 32-bit images are intentionally unsupported; A/B gates cannot be bypassed; model checksum failure prevents startup; feature payloads cannot deserialize executable objects.

## Checkpoint 4: Source-Group Generation And Detection

### Task 10: Implement Multi-View DINO Recall And Geometry Confirmation

**Files:**
- Create: `trace_app/v4/recall.py`
- Create: `trace_app/v4/geometry.py`
- Test: `tests/v4/test_recall.py`
- Test: `tests/v4/test_geometry.py`

- [ ] **Step 1: Write failing recall/geometry tests**

Define deterministic full-image and overlapping multi-scale view boxes; batch them through DINO; aggregate neighbors by best cosine distance, number of matching views, then distance consistency with a stable source-group ID tie-break. Cap output at 40. Test ORB/RANSAC first, LightGlue only for a configured bounded difficult/low-texture subset, homography sanity, and deadline checks between every candidate.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_recall.py tests/v4/test_geometry.py -q`

Expected: FAIL importing the new modules.

- [ ] **Step 3: Implement deterministic recall and confirmation**

Store the view policy version with every group. Query pgvector with owner filtering inside the vector SQL, aggregate to at most 40 distinct groups, and load only their feature rows. Geometry returns `ConfirmedGroup(source_group_id, homography, method, inliers, ratio, reprojection_error)` and never a record ID. Limit LightGlue by count and remaining deadline; model/feature errors propagate as `service_unavailable`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/v4/test_recall.py tests/v4/test_geometry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/recall.py trace_app/v4/geometry.py tests/v4/test_recall.py tests/v4/test_geometry.py
git commit -m "feat: recall and confirm v4 source groups"
```

### Task 11: Implement Transactional V4 Generation

**Files:**
- Create: `trace_app/v4/generation.py`
- Test: `tests/v4/test_generation.py`

- [ ] **Step 1: Write failing generation tests**

Cover canonical decoded RGB SHA-256 grouping, owner separation, one-time group embeddings/features, per-version tag generation, retry on `uq_v4_group_auth_tag`, atomic media staging, cleanup on any failure, and commit only after decode/model/embed/output/hash checks. Assert the stored codec is exactly `hmac64_rs_16_8_split_repeat_sync_v4`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_generation.py -q`

Expected: FAIL because `V4GenerationService` does not exist.

- [ ] **Step 3: Implement the generation unit of work**

Stream and hash input, decode in a worker, derive the pixel hash, create/reuse the owner group under a database uniqueness lock, generate models/features only for a new group, retry cryptographically random trace IDs on the group/tag unique violation, embed in the worker, stage three opaque media objects, and promote them only after the database unit commits. A failed transaction removes staged files and media rows and appends a redacted failure audit event.

- [ ] **Step 4: Run tests**

Run: `pytest tests/v4/test_generation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/generation.py tests/v4/test_generation.py
git commit -m "feat: generate source grouped hardened v4 records"
```

### Task 12: Implement Exact-First, Source-Group-First Detection

**Files:**
- Create: `trace_app/v4/detection.py`
- Test: `tests/v4/test_detection_pipeline.py`
- Test: `tests/v4/test_same_source_versions.py`

- [ ] **Step 1: Write failing pipeline tests**

Test exact MD5+SHA-256 lookup before decode; owner-scoped vector recall; max 40 groups; geometry before watermark extraction; one warp/observation per confirmed group; blind RS tag decode; indexed `(group, tag)` lookup; constant-time source-bound HMAC verification; unique success, zero-match `not_found`, multi-match `ambiguous` with no record, timeout/resource/service distinctions, and order/cache/recent-state invariance. Seed 100,000 versions in one group and verify the correct oldest/middle/newest record without per-version image processing, including the previously failing correct-version-ranked-fourth case.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_detection_pipeline.py tests/v4/test_same_source_versions.py -q`

Expected: FAIL because current detection ranks and processes record candidates.

- [ ] **Step 3: Implement the orchestration state machine**

Hash while streaming. Exact lookup returns only after ownership and record status checks. Otherwise decode and recall groups, confirm geometry, warp once, extract one immutable observation, decode its tag, query one record by the group/tag unique index, recompute HMAC with that record's key ID, and compare with `hmac.compare_digest`. Collect authenticated records across all confirmed groups; return data only for exactly one. Catch only typed deadline/resource/service exceptions and preserve their distinct outcomes.

- [ ] **Step 4: Run detection tests**

Run: `pytest tests/v4/test_detection_pipeline.py tests/v4/test_same_source_versions.py -q`

Expected: PASS; spies report zero `read_records()` and zero per-version image/model operations.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/detection.py tests/v4/test_detection_pipeline.py tests/v4/test_same_source_versions.py
git commit -m "feat: authenticate v4 records through source groups"
```

**Checkpoint 4 gate:** generation is atomic; source features are reused; the fourth/100,000th same-source version is found; candidate order and cache state do not change outcomes; no visual-only success is possible.

## Checkpoint 5: API, Jobs, Management, And Observability

### Task 13: Wire V4-Only APIs And Remove Legacy Fallbacks From The Runtime Path

**Files:**
- Create: `trace_app/api/v4.py`
- Modify: `trace_app/application.py`
- Modify: `trace_app/api/images.py`
- Modify: `trace_app/api/dashboard.py`
- Modify: `frontend/src/forms/watermark.js`
- Modify: `frontend/src/views/WatermarkView.vue`
- Modify: `frontend/src/views/TraceView.vue`
- Test: `tests/v4/test_v4_api.py`
- Test: `frontend/tests/v4-only-contract.test.js`

- [ ] **Step 1: Write failing API contracts**

Require login for upload generation, upload detection, URL detection, listing, deletion, and media issuance. Reject any requested codec/version other than the new V4 codec. Non-admins see only their owner scope; admins must explicitly request cross-owner detection. Assert API payloads expose opaque access URLs and typed outcomes only. Assert frontend labels DINOv2/LightGlue as available only from `/api/v4/capabilities` health data.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_v4_api.py -q`

Run: `npm test -- --run frontend/tests/v4-only-contract.test.js`

Expected: FAIL because the application still wires the generic legacy watermark service.

- [ ] **Step 3: Wire only new services for production V4 routes**

Register `/api/v4/generate`, `/api/v4/detect`, `/api/v4/detect-url`, `/api/v4/records`, and `/api/v4/capabilities`. Keep legacy modules importable for their old unit tests but do not call them from these routes. Remove `/api/dev/reset` registration. Return HTTP 401/403/404 without existence leaks and map typed outcomes consistently; do not map timeouts or service errors to 404.

- [ ] **Step 4: Run API and frontend tests**

Run: `pytest tests/v4/test_v4_api.py tests/v4/test_api_authorization.py -q`

Run: `npm test -- --run frontend/tests/v4-only-contract.test.js frontend/tests/media-access-contract.test.js`

Expected: PASS.

- [ ] **Step 5: Commit without staging unrelated frontend edits**

```bash
git add trace_app/api/v4.py trace_app/application.py trace_app/api/images.py trace_app/api/dashboard.py frontend/src/forms/watermark.js frontend/src/views/WatermarkView.vue frontend/src/views/TraceView.vue tests/v4/test_v4_api.py frontend/tests/v4-only-contract.test.js
git commit -m "feat: expose authenticated v4 only api"
```

### Task 14: Add Deep-Forensics Jobs, Audit, And Structured Telemetry

**Files:**
- Create: `trace_app/v4/jobs.py`
- Create: `trace_app/v4/telemetry.py`
- Modify: `trace_app/api/v4.py`
- Test: `tests/v4/test_deep_jobs.py`
- Test: `tests/v4/test_telemetry.py`

- [ ] **Step 1: Write failing job and telemetry tests**

Test explicit asynchronous creation, owner-scoped status, progress, cancellation, lease recovery, absolute 1,000-second termination, and no synchronous work continuing after 300 seconds. Capture telemetry and require timings/counts for every design stage and a final typed outcome. Assert tokens, keys, auth tags, image bytes, absolute paths, internal hosts, and buckets never occur in logs/audit.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_deep_jobs.py tests/v4/test_telemetry.py -q`

Expected: FAIL because jobs and redacted structured events do not exist.

- [ ] **Step 3: Implement durable jobs and allowlisted events**

Persist queued/running/completed/failed/cancelled states, progress 0..100, lease owner/expiry, requested owner scope, and result media/evidence IDs. Workers claim with `FOR UPDATE SKIP LOCKED`, renew leases, check cancellation and the fixed absolute deadline, and terminate descendants. Telemetry serializes only an explicit field allowlist; audit records actor/action/target/outcome/correlation/time without request secrets.

- [ ] **Step 4: Run tests**

Run: `pytest tests/v4/test_deep_jobs.py tests/v4/test_telemetry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_app/v4/jobs.py trace_app/v4/telemetry.py trace_app/api/v4.py tests/v4/test_deep_jobs.py tests/v4/test_telemetry.py
git commit -m "feat: add bounded v4 forensic jobs and telemetry"
```

**Checkpoint 5 gate:** every product operation is authenticated; owner boundaries hold for API, database, jobs, and media; synchronous and deep deadlines are distinct; logs pass the sensitive-value canary suite.

## Checkpoint 6: Destructive Initialization And Release Proof

### Task 15: Build The Offline Backup, Reset, Restore, And Startup Marker Workflow

**Files:**
- Create: `tools/initialize_v4.py`
- Create: `tools/restore_v4_backup.py`
- Create: `docs/operations/v4-initialization.md`
- Modify: `trace_app/v4/startup.py`
- Test: `tests/v4/test_initialize_v4.py`
- Test: `tests/v4/test_restore_v4.py`

- [ ] **Step 1: Write failing maintenance tests**

Use disposable database/filesystem fixtures. Require absolute resolved non-root targets, rejection of workspace/home/drive roots, PostgreSQL/pgvector/free-space/model/worker preflight, readable database and upload backups before confirmation, the exact confirmation value `RESET-V4:{environment}:{database_name}`, generated new key ID without secret output, preservation of users/roles/admin, clearing of old image/V4/features/statistics/media/uploads, schema creation, smoke tests, and ready marker last. Inject failure after every phase and prove service remains offline and restore recovers fixtures.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/v4/test_initialize_v4.py tests/v4/test_restore_v4.py -q`

Expected: FAIL because the offline workflow does not exist.

- [ ] **Step 3: Implement guarded CLI commands**

Provide `preflight`, `backup`, `apply --confirm RESET-V4:{environment}:{database_name}`, `verify`, and `restore` subcommands. Use `pg_dump` custom format plus `pg_restore --list` verification; archive upload files without following links and verify archive members/checksum. Snapshot preserved user/role/admin IDs, delete only the explicit algorithm/media tables and validated upload children, create/verify new structures, run smoke tests, then atomically write a non-secret ready marker containing schema/model/key IDs. Never print or log the generated secret.

- [ ] **Step 4: Exercise disposable reset and restore**

Run: `pytest tests/v4/test_initialize_v4.py tests/v4/test_restore_v4.py -q`

Expected: PASS, including all injected failure points.

Run: `python tools/initialize_v4.py preflight --env-file .env.v4.test`

Expected: prints only checked target identifiers and `preflight: ok`; does not mutate data.

- [ ] **Step 5: Commit**

```bash
git add tools/initialize_v4.py tools/restore_v4_backup.py docs/operations/v4-initialization.md trace_app/v4/startup.py tests/v4/test_initialize_v4.py tests/v4/test_restore_v4.py
git commit -m "feat: add guarded v4 initialization and restore"
```

### Task 16: Add Correctness, Security, Scale, Quality, And Performance Release Gates

**Files:**
- Create: `tests/v4/benchmark_manifest.py`
- Create: `tests/v4/run_release_gates.py`
- Create: `tests/v4/test_negative_gate.py`
- Create: `tests/v4/test_scale_gate.py`
- Create: `tests/v4/test_attack_gate.py`
- Create: `tests/v4/test_quality_gate.py`
- Create: `tests/v4/test_performance_gate.py`
- Create: `docs/operations/v4-release-gates.md`
- Modify: `tools/build_centos_release.py`

- [ ] **Step 1: Write failing release-gate tests**

Freeze dataset hashes and keep development/release/blind manifests disjoint. Encode these exact thresholds: DINO recall >=99% for full/resize/JPEG and >=95% qualifying crops; final attribution >=95%; wrong traces exactly zero; at least 30,000 independent negatives with zero attribution; PSNR >=38 and SSIM >=0.95 by low/high texture, photo, text, UI, and synthetic strata; standard P95 <=120 seconds and every request <=300 seconds; deep jobs <=1,000 seconds. Include JPEG/recompression, crop, rotation, screenshot, screen-photo, denoise, sharpen, noise, pilot notch, DCT attenuation, same-source collusion, and overwrite attacks.

- [ ] **Step 2: Verify gates fail closed without datasets/hardware metadata**

Run: `pytest tests/v4/test_negative_gate.py tests/v4/test_scale_gate.py tests/v4/test_attack_gate.py tests/v4/test_quality_gate.py tests/v4/test_performance_gate.py -q`

Expected: FAIL with explicit missing manifest/reference hardware/model evidence, never SKIP or PASS.

- [ ] **Step 3: Implement deterministic runners and signed reports**

Record git commit, schema/codec/model versions, manifest hashes, random seed, reference CPU/GPU/RAM, per-stage timing distributions, outcome counts, quality strata, and raw evidence artifact hashes. `run_release_gates.py` exits nonzero for a missing suite, threshold miss, wrong trace, timeout overrun, dataset overlap, model-health failure, query-plan failure, or sensitive log leak. Release building reads the successful report for the current commit and refuses stale or absent evidence.

- [ ] **Step 4: Run fast regression and full release commands**

Run: `pytest tests/v4 -q -m "not postgres and not benchmark"`

Expected: PASS.

Run: `python -m tests.v4.run_release_gates --manifest tests/fixtures/v4/release.json --output test_output/v4-release-report.json`

Expected: exits 0 only after all PostgreSQL, 30,000-negative, scale, attack, quality, security, and timing gates pass on documented hardware.

- [ ] **Step 5: Commit**

```bash
git add tests/v4/benchmark_manifest.py tests/v4/run_release_gates.py tests/v4/test_negative_gate.py tests/v4/test_scale_gate.py tests/v4/test_attack_gate.py tests/v4/test_quality_gate.py tests/v4/test_performance_gate.py docs/operations/v4-release-gates.md tools/build_centos_release.py
git commit -m "test: enforce hardened v4 release gates"
```

**Checkpoint 6 gate:** disposable reset and restore pass; preserved users/roles/admin IDs match exactly; old algorithm data and files are gone; new key is active; startup marker is issued only after smoke tests; all release thresholds pass on the documented system.

## Final Integration Verification

- [ ] **Step 1: Prove the V4 runtime has no legacy scan or fallback dependency**

Run: `rg -n "read_records\(|detect_by_visual_match|detect_by_residual_match|detect_robust_watermark|legacy" trace_app/v4 trace_app/api/v4.py`

Expected: no matches except an explicit negative assertion in comments/tests; remove any runtime match before proceeding.

- [ ] **Step 2: Run all backend and frontend regression tests**

Run: `pytest -q`

Expected: PASS.

Run: `npm test -- --run`

Expected: PASS.

- [ ] **Step 3: Run PostgreSQL and full release gates**

Run: `pytest tests/v4 -q -m postgres`

Expected: PASS with no skipped PostgreSQL requirement.

Run: `python -m tests.v4.run_release_gates --manifest tests/fixtures/v4/release.json --output test_output/v4-release-report.json`

Expected: PASS with zero wrong traces, zero of at least 30,000 negative attributions, P95 <=120 seconds, hard max <=300 seconds, and deep max <=1,000 seconds.

- [ ] **Step 4: Review the complete diff and secret/path hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: user-owned pre-existing changes remain untouched; only intentional V4 files are staged/committed by this plan.

Run the repository secret-hygiene tests and scan API fixtures/reports for absolute paths, `/uploads/`, internal hosts, auth tokens, auth tags, and key material. Any occurrence blocks release.

- [ ] **Step 5: Perform the production maintenance only after explicit operator confirmation**

Follow `docs/operations/v4-initialization.md`: keep the site offline, run preflight, create and verify backups, record the exact preservation snapshot, supply the exact destructive confirmation, run apply/verify, and retain the backup until post-deployment acceptance is signed. This is the only step that clears old data; plan execution and normal application startup must never delete it automatically.

## Self-Review Record

- Spec coverage: all fixed decisions, generation/detection stages, relational entities, authorization, remote URL controls, worker limits, 120/300/1000 deadlines, opaque media mapping, destructive reset/restore, observability, and acceptance gates map to Tasks 1-16.
- Placeholder scan: every implementation action, test command, expected result, and commit boundary is specified; no deferred work or shorthand cross-task references remain.
- Type consistency: `OwnerScope`, `DetectionOutcome`, `Deadline`, source-group IDs, eight-byte `auth_tag`, 384-dimensional embeddings, opaque media IDs, and fixed codec ID retain the same meaning across schema, services, API, jobs, and tests.
- Scope control: V1/V2/V3, legacy image algorithms, MySQL migration, and old-data backfill are excluded. Existing users, roles, and administrator identity are preserved.
