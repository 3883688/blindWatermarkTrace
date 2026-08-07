# Model Visual Recall and Full-Version Authentication Design

**Date:** 2026-07-23

## Goal

Replace the current in-memory visual coarse ranking and top-two authentication
behavior with a scalable PostgreSQL pipeline that:

1. recalls source groups through DINOv2 image embeddings stored in pgvector;
2. confirms the source group through ORB/RANSAC geometry;
3. authenticates every new watermark record in each confirmed source group;
4. returns a record only when exactly one authentication code succeeds; and
5. uses the recently generated list only as a cache optimization.

This phase covers records generated after the change. Existing records without
source-group metadata are preserved but are not backfilled into the new visual
retrieval path.

## Current Constraints

The current V4 detector reads all records, loads file-based feature indexes,
ranks them in Python, and selects at most two visual candidates plus one recent
candidate. Authentication therefore depends on a record reaching this small
coarse-ranking set. The database stores most searchable fields inside a text
JSON payload, so exact fingerprint detection also transfers all records to the
application.

The active runtime uses PostgreSQL, and pgvector 0.8.5 is installed. The
application must keep its existing exact-file fast path and public response
shape.

## Chosen Approach

Use a pinned DINOv2 ViT-S/14 ONNX model to produce normalized 384-dimensional
visual embeddings. The model file is a local deployment asset with a recorded
SHA-256 checksum; detection never downloads model data at request time. CPU
inference uses ONNX Runtime and does not require a GPU.

Each source group stores embeddings for the full source image and a fixed set
of overlapping multi-scale crops. This produces approximately 9 to 12 vectors
per source group. At 10,000 source groups, the HNSW index therefore contains
approximately 100,000 vectors instead of more than one million per-descriptor
ORB vectors.

DINOv2 is only a recall mechanism. A model distance never establishes
ownership and is never returned as a successful trace result. ORB/RANSAC and
the embedded authentication code remain mandatory correctness gates.

## Source Groups

Images belong to the same source group when their decoded RGB pixel content
has the same `original_image_sha256`. New generation requests calculate this
hash before watermark embedding.

Add these relational fields and tables:

- `source_groups`
  - `id` string primary key
  - `original_image_sha256` unique and non-null
  - `image_width` and `image_height`
  - `created_at`
- `source_group_embeddings`
  - `source_group_id` foreign key with cascade deletion
  - `view_index` and `view_kind`
  - `embedding vector(384)`
  - primary key on `(source_group_id, view_index)`
  - HNSW index using cosine distance
- `source_group_features`
  - `source_group_id` foreign key with cascade deletion
  - ORB keypoint position, scale, angle, response, octave, and class ID
  - the corresponding 32-byte ORB descriptor
  - primary key on `(source_group_id, feature_index)`
- concrete columns on `image_records`
  - `source_group_id`
  - `robust_auth_code`
  - `robust_watermark_version`
  - original and watermarked MD5/SHA-256 file fingerprints

New records must have a source group and authentication code. Existing rows may
leave the new columns null. A unique constraint on
`(source_group_id, robust_auth_code)` guarantees group-local authentication-code
uniqueness under concurrent requests. Fingerprint columns receive database
indexes so the exact-file fast path no longer loads all record JSON.

## Generation Flow

1. Decode the uploaded image and calculate its RGB content hash.
2. Find or create the source group in a transaction. A unique constraint on the
   content hash resolves concurrent creation races.
3. For a new source group, extract the fixed full-image and crop views, run them
   as a model batch, and store their normalized embeddings. Extract and store a
   canonical ORB feature index from the unwatermarked source.
4. Generate the trace ID and authentication code. If that code already exists
   in the group, generate a new trace ID and retry before embedding.
5. Create the watermarked output and insert the record with relational lookup
   columns and the existing JSON payload in the same logical operation.
6. If a concurrent insert still hits the authentication uniqueness constraint,
   discard that generated identity and retry with a new trace ID.

Generating another watermark from the same source reuses group embeddings and
ORB features. It performs no model inference for source indexing.

## Detection Flow

1. Calculate the uploaded file MD5 and SHA-256 and use indexed repository
   queries for exact original or watermarked matches.
2. If no exact match exists, decode the query image, generate one normalized
   DINOv2 embedding, and query the pgvector HNSW index.
3. Aggregate view-level matches by source group. Rank groups by best distance,
   number of matching views, and distance consistency, then retain at most 40
   groups.
4. Load only those groups' canonical ORB features. Perform descriptor ratio
   matching and RANSAC against every recalled group. Keep only groups that pass
   the configured inlier count, inlier ratio, and reprojection-error gates.
5. For each geometrically confirmed group, warp the query once into canonical
   source coordinates and extract the tile scores, aggregate codeword, and byte
   confidences once.
6. Load every new watermark record in the group and run the inexpensive
   Reed-Solomon/authentication comparison against each record's authentication
   code. Do not cap or truncate the version list.
7. Return the complete record JSON only when exactly one record authenticates
   across all confirmed groups.

The current candidate-specific V4 decoder will be separated into two units:
one prepares an aligned observation from image pixels, and the other validates
that immutable observation against a candidate authentication code. This keeps
full-version authentication cost almost constant as group size grows.

## Recent-Generation Optimization

The recently generated trace list may prewarm source-group geometry and record
caches. It must not:

- reserve a recall position;
- add a group omitted by pgvector;
- change model or RANSAC scores;
- limit which records in a confirmed group are authenticated; or
- change a success, miss, ambiguity, or timeout result.

Tests will compare results with the list empty, populated with incorrect IDs,
and populated with the target ID.

## Performance Targets

Initial non-binding engineering targets on a modern CPU are:

- query embedding: 80 to 350 milliseconds;
- pgvector HNSW recall: 5 to 30 milliseconds at approximately 100,000 vectors;
- ORB/RANSAC for up to 40 groups: 200 milliseconds to 1.5 seconds;
- aligned authentication observation: 100 to 800 milliseconds per confirmed
  geometry; and
- candidate-code comparison: below 1 millisecond per record.

The initial end-to-end target is commonly 0.5 to 3 seconds, with slower CPUs
potentially taking 3 to 8 seconds. These are benchmark targets, not correctness
requirements. A request that reaches its deadline before every version is
checked returns a timeout and never uses a partial result.

## Failure Behavior

- An image with too few useful ORB features returns the existing not-found
  response after model recall cannot be geometrically confirmed.
- No pgvector result or no RANSAC-confirmed group returns not found.
- A confirmed group with no authenticated record returns not found; visual
  similarity must not fall through to attribution.
- More than one authenticated record returns an explicit authentication
  ambiguity error and no record data.
- A missing model, checksum mismatch, missing pgvector extension, or unusable
  production HNSW index fails startup with a clear configuration error.
- Database or inference failures return a service error, not a false not-found
  response.
- A deadline reached during geometry or complete group authentication returns a
  timeout, not a partial attribution.

SQLite remains available for existing isolated unit tests through an injected
linear embedding repository. PostgreSQL production execution always uses
pgvector and has no recent-list or in-memory full-scan fallback.

## Deletion and Consistency

Deleting one watermark record keeps its source group while other versions
remain. Deleting the final record removes the group, model embeddings, and ORB
features in one transaction. File cleanup continues after the database result
is known, following the existing management-service behavior.

Generation must not leave a record without complete group indexes. If model or
ORB extraction fails for a new group, generation fails before inserting the
watermark record. Database constraints prevent duplicate groups and duplicate
authentication codes.

## Observability

Record structured measurements for:

- model inference duration;
- pgvector query duration and recalled group count;
- ORB/RANSAC duration and confirmed group count;
- number of versions authenticated;
- aligned-observation and candidate-comparison duration;
- recent-cache hit or miss; and
- final outcome: success, not found, ambiguous, timeout, or service error.

These measurements identify whether later optimization should target model
recall, geometry, or authentication without weakening correctness gates.

## Test Strategy

Use a fixed random seed to generate approximately 50 simple source images. Each
image contains text, lines, geometric shapes, color regions, and local texture,
so the fixture has meaningful visual features without depending on external
business images.

Create one V4 record for every source group and three additional records for a
selected target group, for approximately 53 records total. Tests cover:

- exact original and exact watermarked fingerprint matches;
- full-image model recall;
- queries retaining at least 25 percent useful content;
- resizing, JPEG compression, and mild rotation;
- ORB/RANSAC rejection of visually similar but geometrically unrelated groups;
- authentication of every record in the target group;
- unique attribution to the actual target version;
- correct recognition when the target is absent from the recent list;
- identical outcomes for empty, incorrect, and target-containing recent lists;
- zero attribution for approximately 50 unrelated negative images;
- authentication-code collision retry and database uniqueness enforcement;
- concurrent source-group creation; and
- removal of group indexes only after the final version is deleted.

Because recalling 40 groups from a 40-group fixture would be meaningless, the
model quality gate uses approximately 50 groups and requires the target to rank
within the top 10. Production still retrieves up to 40 groups. Initial quality
targets are:

- 100 percent for exact original and exact watermarked files;
- at least 99 percent model recall for full images, resizing, and JPEG queries;
- at least 95 percent model recall and final attribution for qualifying crops;
- at least 90 percent final attribution for JPEG, resize, and mild-rotation
  transformations; and
- zero false attribution in the generated negative set.

PostgreSQL integration tests create a uniquely named temporary schema, use the
installed pgvector extension and a real HNSW index, and remove only that schema
afterward. Unit tests cover model preprocessing, normalized embedding shape,
group aggregation, geometry gates, aligned-observation reuse, candidate-code
validation, repository constraints, and recent-list invariance.

This generated dataset is the first regression gate, not a claim of production
accuracy. If results are insufficient, later work will use the recorded stage
metrics to adjust source views, recall limits, model choice, or geometry
thresholds without allowing model similarity to become an attribution result.
