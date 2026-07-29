# V4 Production Pipeline Hardening Design

**Date:** 2026-07-29

## Goal

Replace the current record-wide JSON scan and per-record V4 candidate flow with
a secure, PostgreSQL-backed, source-group-first V4 production pipeline. The new
pipeline must support at least 100,000 source groups, 1,000,000 V4 records, and
100,000 versions in one source group without missing an older version.

The deployed product will generate and detect only the new hardened V4 codec.
Existing image, watermark, feature, and statistics data will be cleared. User,
role, and administrator records remain unchanged.

## Scope

This work covers the complete V4 production path:

- authenticated V4 generation, upload detection, URL detection, management,
  media access, and administrator operations;
- PostgreSQL relational storage and pgvector source-group recall;
- DINOv2 recall, ORB/RANSAC geometry, and SuperPoint/LightGlue fallback;
- a new 64-bit authenticated V4 codec;
- source-group-level geometry and full-version authentication;
- bounded worker execution for images of unrestricted business pixel count;
- opaque mapped media addresses;
- data reset, key replacement, observability, and release gates.

V1, V2, V3, legacy LSB, DWT, dot-matrix, and legacy attribution algorithms are
outside scope and will not be used as fallbacks by the new V4 path.

## Fixed Decisions

- Production requires PostgreSQL with pgvector. SQLite remains a test adapter.
- MySQL is not a supported runtime or migration source for the new V4 data.
- The site is offline during the change; no dual-write migration is required.
- All old image and algorithm data is cleared instead of backfilled.
- Users, roles, and the administrator identity remain unchanged.
- A new key set and key ID are created. Old V4 keys are not required.
- All V4 generation and detection operations require authentication.
- Standard detection has a P95 target of 120 seconds and a hard timeout of 300
  seconds. Explicit deep-forensics jobs have an absolute limit of 1,000 seconds.
- There is no product pixel-count limit. Resource isolation, not a fixed
  megapixel rejection threshold, protects the service.

## Architecture

### Generation

1. Authenticate the caller and establish the owner boundary.
2. Stream the upload into controlled temporary storage while enforcing the
   configured byte and temporary-disk quotas.
3. Decode RGB pixels in an isolated worker and calculate the canonical decoded
   pixel SHA-256.
4. Find or create the source group identified by
   `(owner_user_id, original_image_sha256)`.
5. For a new group, create the fixed full-image and overlapping multi-scale
   views, run DINOv2 as one batch, and store normalized 384-dimensional vectors.
6. Extract and store one canonical ORB feature set and one canonical SuperPoint
   feature set for the source group.
7. Generate a trace ID and calculate the source-bound 64-bit authentication tag.
   Retry the trace ID if the group-local authentication uniqueness constraint
   reports a collision.
8. Embed the new V4 pilot and split repeated RS codeword.
9. Stage the original, watermarked output, and thumbnail under opaque logical
   storage keys.
10. Commit the source group, V4 record, media mappings, hashes, and audit event
    only after all model, image, and file checks succeed.

### Detection

1. Authenticate the caller and restrict the search to the caller's owner scope.
   Administrators may explicitly select the cross-owner scope.
2. Calculate MD5 and SHA-256 while streaming the input. Use indexed exact-file
   queries before image decoding.
3. If exact matching misses, decode the image in an isolated worker and produce
   the DINOv2 query vector.
4. Query the pgvector HNSW index and aggregate view results by source group.
   Rank by best distance, matching-view count, and distance consistency, then
   retain at most 40 groups.
5. Confirm recalled groups with ORB/RANSAC. Apply SuperPoint/LightGlue only to a
   bounded number of difficult or low-texture candidates.
6. For each geometrically confirmed source group, warp the query once and
   extract one immutable A/B V4 codeword observation.
7. RS-decode the observation to a 64-bit authentication tag and query the
   `(source_group_id, auth_tag)` unique index.
8. Recalculate the HMAC using the record's key ID and compare it in constant
   time. Visual or geometry scores never replace this step.
9. Return a result only when exactly one record authenticates across every
   confirmed group. Zero matches mean not found; multiple matches produce an
   explicit authentication ambiguity response.

DINOv2, ORB, LightGlue, caches, and recent-record hints may affect execution
order and performance only. Removing caches or changing candidate order must
not change success, miss, ambiguity, or timeout outcomes.

## Hardened V4 Codec

The new codec identifier is:

```text
hmac64_rs_16_8_split_repeat_sync_v4
```

The authentication message is the canonical encoding of:

```text
codec_version || key_id || owner_user_id || source_image_sha256 || trace_id
```

HMAC-SHA256 is truncated to eight bytes. The canonical encoder uses explicit
field lengths and UTF-8/byte representations so concatenation is unambiguous.

The eight-byte tag is encoded with RS(16,8), producing a 16-byte, 128-bit
codeword. A tile continues to encode 64 physical bits, with both DCT
coefficient pairs redundantly representing the same bit:

- checkerboard class A repeats the first 64 codeword bits;
- checkerboard class B repeats the second 64 codeword bits;
- horizontal and vertical neighbors belong to different classes;
- each class retains deterministic phase permutations to distribute burst
  errors;
- decoding requires valid evidence from both classes and the configured phase,
  coverage, and tile gates.

The detector aggregates class A and B independently before assembling the
128-bit observation. It records tile counts, phases, coverage, corrected
symbols, erasures, bit errors, signal score, synchronization confidence, and
elapsed time. These evidence values describe signal quality but do not replace
the HMAC decision.

The database enforces `UNIQUE(source_group_id, auth_tag)`. At 100,000 versions,
the birthday-collision probability for a random 64-bit tag is approximately
2.7e-10; a detected collision is handled by generating a new trace ID.

## Relational Data Model

### `source_groups`

- `id` primary key;
- `owner_user_id` foreign key;
- `original_image_sha256` fixed-length binary value;
- `image_width` and `image_height`;
- opaque `original_media_id`;
- model and feature schema versions;
- status and timestamps;
- unique constraint on `(owner_user_id, original_image_sha256)`.

### `source_group_embeddings`

- `source_group_id` with cascading deletion;
- `owner_user_id` for filtered vector queries;
- `view_index` and `view_kind`;
- `embedding vector(384)`;
- model version;
- primary key on `(source_group_id, view_index)`;
- HNSW index using cosine distance;
- B-tree index on `owner_user_id`.

### `source_group_features`

- `source_group_id` with cascading deletion;
- `feature_kind`, restricted to `orb` or `superpoint`;
- feature schema and model version;
- compressed feature bytes and their SHA-256;
- primary key on `(source_group_id, feature_kind)`.

Feature payloads use versioned, non-pickle serialization with strict shape,
dtype, count, and checksum validation.

### `v4_records`

- record ID, source group ID, and owner user ID;
- trace ID, codec, eight-byte authentication tag, and key ID;
- original and watermarked MD5/SHA-256 file hashes;
- original and watermarked decoded-pixel SHA-256 values;
- opaque output and thumbnail media IDs;
- evidence UUID, creation time, and status;
- optional JSONB metadata for non-query display extensions only.

Required constraints and indexes include:

- unique `(owner_user_id, trace_id)`;
- unique `(source_group_id, auth_tag)`;
- B-tree indexes for source group, owner, creation time, codec, and key ID;
- indexed paired MD5/SHA-256 exact-file lookups.

No authorization, exact match, source recall, V4 candidate selection, or
authentication query may scan or parse the metadata JSONB column. The V4
detection path must not call the legacy full-record `read_records()` API.

### Sessions, rate limits, statistics, and audit

- Server-side sessions store only token hashes, user ID, creation time, idle
  expiry, absolute expiry, revocation state, and last-use time.
- Password changes and administrator actions can revoke all user sessions.
- Atomic rate-limit buckets are updated with PostgreSQL upserts by user, IP,
  endpoint class, and time window.
- Generation and detection counters use atomic database increments.
- Audit rows record actor, action, target ID, outcome, request correlation ID,
  and timestamp without recording passwords, tokens, keys, or image content.

## Authorization and Security

- Generation, upload detection, URL detection, image management, and media-URL
  issuance require a valid non-expired session. The mapped media transfer route
  accepts its short-lived scoped signature so browser image elements do not
  need to attach an authorization header.
- Non-administrators are restricted at query time to their own source groups and
  records. Administrators may perform explicit cross-owner operations.
- User and role administration requires the administrator role.
- The development reset endpoint is not registered in production.
- Destructive initialization is an offline CLI with resolved target checks,
  verified backup, an explicit confirmation value, and audit output.
- Login errors do not reveal whether a username exists. Login attempts are
  limited by account and IP.
- Uploads are streamed and byte-limited. Image decoding and every model or
  watermark operation occur in isolated workers with memory, CPU-time,
  temporary-disk, concurrency, and deadline limits.
- There is no fixed business pixel limit. Images that exceed available worker
  resources fail with a resource-exhausted outcome, never with a false
  not-found attribution.
- DCT and pilot operations process original-resolution data in bounded row or
  tile batches. DINO and geometry use controlled analysis representations.
- Remote URL retrieval allows only public HTTP(S), disables inherited proxies,
  validates every redirect, pins the validated destination IP to the actual
  connection, and enforces byte and time limits.
- Missing models, checksum mismatches, unavailable pgvector support, or invalid
  indexes fail startup. The system never falls back to visual-only attribution.

## Deadline Model

Every stage receives one shared monotonic deadline and checks it between bounded
operations. Stage telemetry records elapsed and remaining time.

- Standard synchronous detection: P95 target at or below 120 seconds.
- Standard synchronous hard timeout: 300 seconds.
- Explicit deep-forensics asynchronous job: absolute limit of 1,000 seconds.

A synchronous request stops all associated worker work at 300 seconds. It does
not continue in the background. Only an explicitly created deep-forensics job
may use the 1,000-second limit. Timeout is a distinct result and must not be
reported as no watermark.

## Opaque Media Mapping

Database records and API responses never contain an absolute server path,
drive letter, internal hostname, upload-root name, object-storage bucket, or
direct storage URL.

The storage layer maps opaque media IDs to logical storage keys such as:

```text
originals/7f/7f3a...bin
watermarked/a2/a268...jpg
thumbnails/19/198c...webp
```

Clients receive only a mapped, expiring address:

```text
/api/media/{opaque_media_id}?expires=...&signature=...
```

`MEDIA_PUBLIC_BASE_URL` may place the same route behind a dedicated media
domain, CDN, reverse proxy, or object-store gateway without changing business
records. Media signatures bind the media ID, variant, owner/access scope, and
expiry so a thumbnail authorization cannot be reused for an original.

The resolver enforces allowed storage roots, rejects traversal and symlink
escapes, and verifies media ownership before transfer. Nginx internal redirects
may transfer bytes efficiently, but internal paths remain absent from client
responses and application logs. Models, feature indexes, temporary files, and
backups are never registered as media.

## Offline Initialization and Rollback

1. Resolve and validate every database and filesystem target.
2. Verify PostgreSQL and pgvector versions, free space, model checksums, and
   worker-runtime availability.
3. Create database and upload-directory backups and verify that they can be
   read before deleting data.
4. Generate the new active key and key ID without printing secret material.
5. Preserve users, roles, and the administrator identity.
6. Clear image records, old V4 data, feature indexes, statistics, media
   mappings, and uploaded image files.
7. Create the new relational, vector, session, rate-limit, audit, and media
   structures.
8. Run database constraints, model inference, V4 encode/decode, authorization,
   and media-mapping smoke tests.
9. Mark the new application version startable only after every check succeeds.

Any failure before the final marker leaves the service offline and triggers the
documented restore procedure. The initialization CLI is idempotent before the
destructive confirmation point and refuses ambiguous or broad filesystem
targets.

## Observability

Structured telemetry records:

- exact-hash lookup duration and outcome;
- DINO preprocessing and inference duration;
- pgvector query duration, neighbor count, and recalled group count;
- ORB and LightGlue durations and geometry evidence;
- A/B tile counts, phase counts, RS corrections, and signal scores;
- group authentication lookup count and outcome;
- worker resource usage and termination reason;
- final success, not-found, ambiguity, timeout, resource-exhausted, or service
  error outcome.

Sensitive values and real storage locations are excluded from telemetry.

## Acceptance Gates

### Data and correctness

- The V4 detection path performs no full-record JSON scan.
- Core searchable fields are concrete constrained columns.
- Exact file and group authentication queries have query-plan regression tests
  demonstrating index use.
- 100,000 source groups, 1,000,000 versions, and 100,000 versions in one group
  preserve unique attribution without per-version image processing.
- Cache state, recent-record state, and candidate ordering do not change the
  final result.

### Security

- Anonymous generation, detection, management, media, user, and role operations
  are rejected.
- Cross-owner reads, detection, downloads, and existence probing are rejected.
- Login brute force, unbounded upload reads, DNS rebinding, model replacement,
  media traversal, symlink escape, and destructive-command target errors have
  regression coverage.
- No response, error, audit row, or normal log contains a token, key, absolute
  media path, internal host, or storage bucket.

### Model and algorithm

- DINO source-group recall is at least 99 percent for full images, resizing,
  and JPEG queries, and at least 95 percent for qualifying crops.
- Final correct attribution is at least 95 percent, with zero wrong traces.
- At least 30,000 independent negative cases produce zero false attributions.
- Attack coverage includes JPEG and repeated compression, cropping, rotation,
  screenshot and screen-photo routes, denoising, sharpening, additive noise,
  pilot notch filtering, targeted DCT attenuation, same-source multi-version
  collusion, and valid-watermark overwrite.
- Minimum image-quality gates remain PSNR 38 and SSIM 0.95, reported separately
  for low-texture, high-texture, photographic, text, UI, and synthetic images.
- Real platform and device routes are collected as independent evidence after
  parameters are frozen; development fixtures are not reused as the final blind
  test set.

### Performance and operations

- Standard detection P95 is at most 120 seconds on the documented reference
  hardware, with a 300-second hard stop.
- Deep-forensics jobs stop by 1,000 seconds and support progress and cancellation.
- Worker resource exhaustion cannot terminate or exhaust the web process.
- Data reset and restore are exercised against disposable fixtures before the
  production maintenance run.
- The frontend advertises DINOv2 or LightGlue only when the backend capability
  and model health checks pass.

## Failure Semantics

- `not_found`: recall and authentication completed, with no valid V4 record.
- `ambiguous`: more than one record authenticated; no record data is returned.
- `timeout`: the applicable shared deadline expired.
- `resource_exhausted`: an isolated worker exceeded its resource allocation.
- `service_unavailable`: database, pgvector, model, key, or required index is
  unavailable or invalid.

Partial processing, visual similarity, stale cache data, or a model error must
never be converted into a successful attribution or an ordinary not-found result.
