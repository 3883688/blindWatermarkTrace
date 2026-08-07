# Model Visual Recall and Full-Version Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PostgreSQL/pgvector DINOv2 recall pipeline that geometrically confirms source groups and authenticates every V4 record in the confirmed group before returning one unique trace result.

**Architecture:** A local ONNX DINOv2-S embedder produces one query vector and ten source-view vectors. PostgreSQL HNSW retrieves source groups, stored ORB geometry confirms them, and a refactored V4 decoder extracts one aligned observation per group before comparing every group authentication code. Existing JSON records and exact-file API responses remain compatible, while new concrete columns remove full-record scans from detection.

**Tech Stack:** Python 3.10+, FastAPI, Pillow, OpenCV, NumPy, ONNX Runtime, SQLAlchemy, PostgreSQL 15+, pgvector 0.8+, pytest

---

## File Structure

- Create `trace_app/imaging/embeddings.py`: model artifact verification, image preprocessing, source views, ONNX inference, and normalized embeddings.
- Create `trace_app/database/visual_index.py`: source-group value objects plus PostgreSQL/SQLite visual-index queries.
- Create `trace_app/watermark/visual_authentication.py`: model recall, cache-only recent optimization, ORB/RANSAC confirmation, complete group authentication, and result formatting.
- Create `tools/prepare_visual_model.py`: obtain or install the pinned ONNX file and write its checksum sidecar atomically.
- Modify `database_store.py`: relational source-group tables, searchable record columns, schema upgrade, and transaction-safe writes/deletes.
- Modify `trace_app/database/repositories.py`: narrow public methods for exact fingerprints, source groups, recall, and grouped records.
- Modify `trace_app/database/connection.py`: production pgvector/model readiness checks.
- Modify `trace_app/config.py`: validated model path and recall settings.
- Modify `trace_app/watermark/service.py`: use indexed generation/detection for new V4 records without loading every record.
- Modify `trace_app/watermark/default_operations.py`: construct and inject the visual authenticator while preserving legacy callbacks.
- Modify `watermark_v4/detector.py`: separate aligned observation extraction from candidate authentication and add grouped detection.
- Modify `requirements.txt`, `.env.example`, `.gitignore`, `tools/build_centos_release.py`, `deploy.sh`, `README.md`, and `README_DEPLOY.md`: dependencies, model preparation, packaging, and PostgreSQL deployment.
- Create focused unit and PostgreSQL integration tests under `tests/`.

### Task 1: Pin Configuration and Model Artifact Handling

**Files:**
- Create: `tools/prepare_visual_model.py`
- Modify: `trace_app/config.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `.gitignore`
- Test: `tests/test_visual_model_setup.py`

- [ ] **Step 1: Write failing configuration and artifact tests**

```python
from hashlib import sha256
from pathlib import Path

import pytest

from tools.prepare_visual_model import install_model
from trace_app.config import Settings


def test_settings_resolve_visual_model_inside_base_dir(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="postgresql+psycopg://trace@localhost/trace",
        admin_user="admin",
        admin_pass="secret",
        visual_model_path="models/dinov2-small.onnx",
    )
    assert settings.visual_model_path == tmp_path / "models/dinov2-small.onnx"
    assert settings.visual_model_checksum_path == tmp_path / "models/dinov2-small.onnx.sha256"


def test_install_model_copies_and_records_sha256(tmp_path: Path) -> None:
    source = tmp_path / "download.onnx"
    source.write_bytes(b"fixed-model-payload")
    destination = tmp_path / "models/dinov2-small.onnx"

    digest = install_model(source.as_uri(), destination)

    assert destination.read_bytes() == b"fixed-model-payload"
    assert digest == sha256(b"fixed-model-payload").hexdigest()
    assert destination.with_suffix(".onnx.sha256").read_text(encoding="ascii") == digest + "\n"


def test_install_model_does_not_replace_valid_existing_asset(tmp_path: Path) -> None:
    destination = tmp_path / "dinov2-small.onnx"
    destination.write_bytes(b"existing")
    digest = sha256(b"existing").hexdigest()
    destination.with_suffix(".onnx.sha256").write_text(digest + "\n", encoding="ascii")

    assert install_model("https://invalid.example/model.onnx", destination) == digest
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_visual_model_setup.py -q`

Expected: collection fails because `install_model` and the visual model settings do not exist.

- [ ] **Step 3: Add model dependencies and settings**

Append `onnxruntime` and `pgvector` to `requirements.txt`. Add these environment defaults to `.env.example`:

```dotenv
VISUAL_MODEL_PATH=./models/dinov2-small.onnx
VISUAL_RECALL_LIMIT=40
VISUAL_RECALL_NEIGHBORS=200
VISUAL_QUERY_TIMEOUT_SECONDS=8
```

Extend `Settings` with resolved model and recall fields:

```python
@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    upload_dir: Path
    data_dir: Path
    db_url: str
    admin_user: str
    admin_pass: str
    app_name: str = "WatermarkSystem"
    visual_model_path: Path = Path("models/dinov2-small.onnx")
    visual_recall_limit: int = 40
    visual_recall_neighbors: int = 200
    visual_query_timeout_seconds: float = 8.0

    @property
    def visual_model_checksum_path(self) -> Path:
        return self.visual_model_path.with_suffix(".onnx.sha256")
```

In `from_values`, resolve a relative `visual_model_path` against `base_dir` and validate `1 <= visual_recall_limit <= 100`, `visual_recall_neighbors >= visual_recall_limit`, and a positive timeout. Add `models/*.onnx` and `models/*.onnx.sha256` to `.gitignore`; release creation will include the local files explicitly after preparation.

- [ ] **Step 4: Implement atomic model installation**

```python
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


DEFAULT_MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/"
    "main/onnx/model.onnx?download=true"
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _valid_existing(destination: Path) -> str | None:
    sidecar = destination.with_suffix(".onnx.sha256")
    if not destination.is_file() or not sidecar.is_file():
        return None
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    return expected if len(expected) == 64 and _digest(destination) == expected else None


def install_model(url: str, destination: Path) -> str:
    existing = _valid_existing(destination)
    if existing:
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        digest = _digest(temporary)
        temporary.replace(destination)
        destination.with_suffix(".onnx.sha256").write_text(
            digest + "\n", encoding="ascii"
        )
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--output", type=Path, default=Path("models/dinov2-small.onnx"))
    arguments = parser.parse_args()
    print(install_model(arguments.url, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_visual_model_setup.py -q`

Expected: `3 passed`.

```powershell
git add requirements.txt .env.example .gitignore trace_app/config.py tools/prepare_visual_model.py tests/test_visual_model_setup.py
git commit -m "feat: configure local visual embedding model"
```

### Task 2: Implement Deterministic DINOv2 Embeddings

**Files:**
- Create: `trace_app/imaging/embeddings.py`
- Test: `tests/test_visual_embeddings.py`

- [ ] **Step 1: Write failing preprocessing and inference tests**

```python
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from trace_app.imaging.embeddings import DinoV2Embedder, source_views


class FakeInput:
    name = "pixel_values"


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feed):
        batch = feed["pixel_values"]
        tokens = np.zeros((len(batch), 197, 384), dtype=np.float32)
        tokens[:, 0, 0] = 3.0
        tokens[:, 0, 1] = 4.0
        return [tokens]


def test_source_views_are_full_plus_nine_overlapping_crops() -> None:
    image = Image.new("RGB", (400, 300), "white")
    views = source_views(image)
    assert [kind for kind, _ in views] == ["full"] + [f"grid-{index}" for index in range(9)]
    assert views[0][1].size == (400, 300)
    assert all(view.size == (200, 150) for _, view in views[1:])


def test_embedder_normalizes_cls_vectors(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    model.with_suffix(".onnx.sha256").write_text(
        sha256(b"model").hexdigest() + "\n", encoding="ascii"
    )
    embedder = DinoV2Embedder(model, session_factory=lambda _: FakeSession())

    vectors = embedder.embed([Image.new("RGB", (320, 240), "red")])

    assert vectors.shape == (1, 384)
    assert vectors.dtype == np.float32
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert np.allclose(vectors[0, :2], [0.6, 0.8])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_visual_embeddings.py -q`

Expected: collection fails because `trace_app.imaging.embeddings` does not exist.

- [ ] **Step 3: Implement fixed views, preprocessing, checksum verification, and batched inference**

```python
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


EMBEDDING_DIMENSIONS = 384
MODEL_SIZE = 224
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def source_views(image: Image.Image) -> tuple[tuple[str, Image.Image], ...]:
    rgb = image.convert("RGB")
    crop_width = max(1, rgb.width // 2)
    crop_height = max(1, rgb.height // 2)
    lefts = (0, max(0, (rgb.width - crop_width) // 2), rgb.width - crop_width)
    tops = (0, max(0, (rgb.height - crop_height) // 2), rgb.height - crop_height)
    crops = []
    for top in tops:
        for left in lefts:
            index = len(crops)
            crops.append(
                (
                    f"grid-{index}",
                    rgb.crop((left, top, left + crop_width, top + crop_height)),
                )
            )
    return (("full", rgb), *crops)


def _preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    scale = MODEL_SIZE / min(rgb.size)
    resized = rgb.resize(
        (max(MODEL_SIZE, round(rgb.width * scale)), max(MODEL_SIZE, round(rgb.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - MODEL_SIZE) // 2
    top = (resized.height - MODEL_SIZE) // 2
    array = np.asarray(
        resized.crop((left, top, left + MODEL_SIZE, top + MODEL_SIZE)),
        dtype=np.float32,
    ) / 255.0
    normalized = (array - IMAGE_MEAN) / IMAGE_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


class DinoV2Embedder:
    def __init__(
        self,
        model_path: Path,
        *,
        session_factory: Callable[[Path], object] | None = None,
    ) -> None:
        expected = model_path.with_suffix(".onnx.sha256").read_text(encoding="ascii").strip()
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("visual model checksum mismatch")
        factory = session_factory or (
            lambda path: ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        )
        self._session = factory(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        batch = np.stack([_preprocess(image) for image in images])
        output = np.asarray(
            self._session.run(None, {self._input_name: batch})[0],
            dtype=np.float32,
        )
        cls = output[:, 0, :] if output.ndim == 3 else output
        if cls.shape != (len(images), EMBEDDING_DIMENSIONS):
            raise RuntimeError(f"unexpected visual model output shape: {cls.shape}")
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("visual model produced an empty embedding")
        return np.ascontiguousarray(cls / norms, dtype=np.float32)
```

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_visual_embeddings.py -q`

Expected: `2 passed`.

```powershell
git add trace_app/imaging/embeddings.py tests/test_visual_embeddings.py
git commit -m "feat: extract deterministic DINOv2 embeddings"
```

### Task 3: Add Source-Group and Searchable Record Schema

**Files:**
- Modify: `database_store.py`
- Modify: `trace_app/database/connection.py`
- Test: `tests/test_database_store.py`
- Test: `tests/test_visual_schema_postgresql.py`

- [ ] **Step 1: Write failing SQLite schema tests**

```python
def test_schema_contains_visual_source_group_tables(store: DatabaseStore) -> None:
    names = set(inspect(store.engine).get_table_names())
    assert {"source_groups", "source_group_embeddings", "source_group_features"} <= names
    image_columns = {column["name"] for column in inspect(store.engine).get_columns("image_records")}
    assert {
        "source_group_id",
        "robust_auth_code",
        "robust_watermark_version",
        "original_file_md5",
        "original_file_sha256",
        "watermarked_file_md5",
        "watermarked_file_sha256",
    } <= image_columns
```

- [ ] **Step 2: Run the SQLite schema test and verify RED**

Run: `python -m pytest tests/test_database_store.py::test_schema_contains_visual_source_group_tables -q`

Expected: FAIL because the new tables and columns are absent.

- [ ] **Step 3: Define the relational schema**

Add `LargeBinary`, `Float`, `UniqueConstraint`, `Index`, and `TypeDecorator` imports plus `Vector` from `pgvector.sqlalchemy`. Use a dialect-aware type so SQLite unit tests store JSON while PostgreSQL gets a real vector:

```python
class Embedding384(TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(384))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        normalized = [float(item) for item in value]
        return normalized if dialect.name == "postgresql" else json.dumps(normalized)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return list(value) if dialect.name == "postgresql" else json.loads(value)
```

Define:

```python
self.source_groups = Table(
    "source_groups",
    self.metadata,
    Column("id", String(64), primary_key=True),
    Column("original_image_sha256", String(64), nullable=False, unique=True),
    Column("image_width", Integer, nullable=False),
    Column("image_height", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.current_timestamp()),
)
self.source_group_embeddings = Table(
    "source_group_embeddings",
    self.metadata,
    Column("source_group_id", String(64), ForeignKey("source_groups.id", ondelete="CASCADE"), primary_key=True),
    Column("view_index", Integer, primary_key=True),
    Column("view_kind", String(32), nullable=False),
    Column("embedding", Embedding384(), nullable=False),
)
self.source_group_features = Table(
    "source_group_features",
    self.metadata,
    Column("source_group_id", String(64), ForeignKey("source_groups.id", ondelete="CASCADE"), primary_key=True),
    Column("feature_index", Integer, primary_key=True),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("size", Float, nullable=False),
    Column("angle", Float, nullable=False),
    Column("response", Float, nullable=False),
    Column("octave", Integer, nullable=False),
    Column("class_id", Integer, nullable=False),
    Column("descriptor", LargeBinary(32), nullable=False),
)
```

Add the seven searchable columns to `image_records`, with a foreign key from `source_group_id` and a unique constraint named `uq_image_records_source_auth` on `(source_group_id, robust_auth_code)`. Existing rows remain nullable.

- [ ] **Step 4: Upgrade existing PostgreSQL tables before creating the HNSW index**

In `DatabaseStore.create_schema`, open a transaction and run `create extension if not exists vector` before `metadata.create_all` when `engine.dialect.name == "postgresql"`. Then execute these exact existing-table upgrades:

```sql
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS source_group_id varchar(64) REFERENCES source_groups(id);
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS robust_auth_code varchar(8);
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS robust_watermark_version integer;
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS original_file_md5 varchar(32);
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS original_file_sha256 varchar(64);
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS watermarked_file_md5 varchar(32);
ALTER TABLE image_records ADD COLUMN IF NOT EXISTS watermarked_file_sha256 varchar(64);
```

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_image_records_source_auth
ON image_records (source_group_id, robust_auth_code)
WHERE source_group_id IS NOT NULL AND robust_auth_code IS NOT NULL
```

```sql
CREATE INDEX IF NOT EXISTS ix_source_group_embeddings_hnsw
ON source_group_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

Also create B-tree indexes for all four fingerprint columns and `image_records.source_group_id`. Keep SQLite on `metadata.create_all` only; SQLite is a unit-test compatibility path and does not create HNSW.

- [ ] **Step 5: Write and run a real PostgreSQL schema test**

Create `tests/test_visual_schema_postgresql.py` with a fixture that reads `TEST_POSTGRES_URL` or `DB_URL`, creates a unique schema named `trace_visual_test_<uuid>`, sets `search_path`, calls `DatabaseStore.create_schema`, asserts `vector` column type plus the HNSW index, and drops only that generated schema in `finally`.

Run: `python -m pytest tests/test_visual_schema_postgresql.py -q`

Expected: PASS against the local PostgreSQL/pgvector 0.8.5 database. If neither URL is configured, the module skips with `PostgreSQL test URL is unavailable`.

- [ ] **Step 6: Run schema regressions and commit**

Run: `python -m pytest tests/test_database_store.py tests/test_visual_schema_postgresql.py -q`

Expected: all tests pass, including the updated table-name assertion containing the three new tables.

```powershell
git add database_store.py trace_app/database/connection.py tests/test_database_store.py tests/test_visual_schema_postgresql.py
git commit -m "feat: add pgvector source group schema"
```

### Task 4: Implement Visual Index Persistence and pgvector Recall

**Files:**
- Create: `trace_app/database/visual_index.py`
- Modify: `database_store.py`
- Modify: `trace_app/database/repositories.py`
- Test: `tests/test_visual_index_repository.py`
- Test: `tests/test_visual_index_postgresql.py`

- [ ] **Step 1: Write failing repository contract tests**

```python
import numpy as np
from PIL import Image
from sqlalchemy import create_engine

from database_store import DatabaseStore
from trace_app.database.repositories import Repository
from trace_app.database.visual_index import PreparedSourceGroup
from watermark_v4.features import extract_feature_index


def prepared_source(seed: int, digest: str) -> PreparedSourceGroup:
    image = Image.new("RGB", (320, 240), (seed, seed * 2 % 255, seed * 3 % 255))
    vectors = np.zeros((10, 384), dtype=np.float32)
    vectors[:, seed % 384] = 1.0
    return PreparedSourceGroup(
        source_group_id=f"group-{seed}",
        original_image_sha256=digest,
        image_width=320,
        image_height=240,
        view_kinds=("full", *(f"grid-{index}" for index in range(9))),
        embeddings=vectors,
        feature_index=extract_feature_index(image),
    )


def test_insert_visual_record_reuses_source_group_and_lists_all_versions() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    repository = Repository(store)
    source = prepared_source(7, "a" * 64)

    repository.add_visual_record(
        {"id": "record-a", "trace_id": "TR-A", "robust_auth_code": "01020304", "robust_watermark_version": 4},
        source=source,
    )
    repository.add_visual_record(
        {"id": "record-b", "trace_id": "TR-B", "robust_auth_code": "05060708", "robust_watermark_version": 4},
        source=source,
    )

    assert repository.source_group_count() == 1
    assert [item.record_id for item in repository.auth_records_for_groups((source.source_group_id,))] == [
        "record-a",
        "record-b",
    ]


def test_indexed_fingerprint_match_does_not_call_read_records(monkeypatch) -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    repository = Repository(store)
    source = prepared_source(8, "b" * 64)
    repository.add_visual_record(
        {
            "id": "record-c",
            "trace_id": "TR-C",
            "robust_auth_code": "11121314",
            "robust_watermark_version": 4,
            "original_file_md5": "m" * 32,
            "original_file_sha256": "s" * 64,
        },
        source=source,
    )
    monkeypatch.setattr(store, "read_records", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full scan")))

    match = repository.find_file_fingerprint("m" * 32, "s" * 64)

    assert match is not None
    assert match["id"] == "record-c"
```

- [ ] **Step 2: Run repository tests and verify RED**

Run: `python -m pytest tests/test_visual_index_repository.py -q`

Expected: collection fails because `PreparedSourceGroup` and repository methods do not exist.

- [ ] **Step 3: Add immutable visual-index value objects**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from watermark_v4.features import FeatureIndex


@dataclass(frozen=True, slots=True)
class PreparedSourceGroup:
    source_group_id: str
    original_image_sha256: str
    image_width: int
    image_height: int
    view_kinds: tuple[str, ...]
    embeddings: np.ndarray
    feature_index: FeatureIndex


@dataclass(frozen=True, slots=True)
class ExistingSourceGroup:
    source_group_id: str
    original_image_sha256: str


SourceGroupReference = PreparedSourceGroup | ExistingSourceGroup


@dataclass(frozen=True, slots=True)
class RecallHit:
    source_group_id: str
    best_distance: float
    matching_views: int


@dataclass(frozen=True, slots=True)
class AuthRecord:
    record_id: str
    source_group_id: str
    trace_id: str
    auth_tag: bytes
    data: dict[str, object]


class DuplicateSourceAuthenticationError(RuntimeError):
    pass
```

Validate exact dimensions, normalized finite vectors, 64-character source hash, nonempty IDs, and matching `view_kinds`/embedding row counts in `PreparedSourceGroup.__post_init__`. Validate both fields of `ExistingSourceGroup`. `find_source_group_by_hash` returns `ExistingSourceGroup | None`, and `insert_visual_record` accepts `SourceGroupReference`; an existing reference must resolve to the same stored content hash and never inserts embeddings again.

- [ ] **Step 4: Implement source-group serialization and indexed record writes**

Add helpers in `database_store.py` to serialize every `FeatureIndex` keypoint/descriptor row and insert the group, its ten embeddings, and its ORB rows. `_insert_source_group_if_absent` must use PostgreSQL or SQLite dialect `insert(...).on_conflict_do_nothing(index_elements=["original_image_sha256"]).returning(id)`, then select the canonical ID when another transaction won the race. Only the transaction receiving a returned ID inserts embeddings/features.

Implement `insert_visual_record` in one transaction:

```python
def insert_visual_record(
    self,
    record: dict[str, Any],
    source: SourceGroupReference,
    *,
    owner_user_id: int | None = None,
) -> None:
    with self.engine.begin() as connection:
        group_id, created = _insert_source_group_if_absent(
            connection,
            self.source_groups,
            source,
        )
        if created:
            connection.execute(insert(self.source_group_embeddings), [
                {
                    "source_group_id": group_id,
                    "view_index": index,
                    "view_kind": kind,
                    "embedding": vector.tolist(),
                }
                for index, (kind, vector) in enumerate(zip(source.view_kinds, source.embeddings, strict=True))
            ])
            connection.execute(insert(self.source_group_features), _feature_rows(group_id, source.feature_index))
        _insert_record_row(
            connection,
            self.image_records,
            record,
            owner_user_id=owner_user_id,
            source_group_id=group_id,
        )
```

Inspect `IntegrityError.orig` for the exact `uq_image_records_source_auth` constraint name and raise `DuplicateSourceAuthenticationError` only for that violation. Do not convert unrelated integrity errors into authentication collisions.

- [ ] **Step 5: Implement exact fingerprints and group reads**

Add database and repository methods:

```python
def find_file_fingerprint(self, md5: str, sha256: str) -> dict[str, Any] | None:
    original = and_(
        self.image_records.c.original_file_md5 == md5,
        self.image_records.c.original_file_sha256 == sha256,
    )
    marked = and_(
        self.image_records.c.watermarked_file_md5 == md5,
        self.image_records.c.watermarked_file_sha256 == sha256,
    )
    with self.engine.connect() as connection:
        row = connection.execute(
            select(self.image_records.c.data, original.label("is_original")).where(or_(original, marked)).limit(1)
        ).mappings().first()
    if row is None:
        return None
    payload = json.loads(row["data"])
    payload["matched_file_type"] = "original" if row["is_original"] else "watermarked"
    payload["matched_hash_type"] = "file_md5_sha256"
    return payload
```

Implement `load_source_group_features(group_ids)` by rebuilding exact `FeatureIndex` instances ordered by `(source_group_id, feature_index)`. Implement `auth_records_for_groups(group_ids)` ordered by `source_group_id, created_at, id`; validate an eight-character hexadecimal auth code and ignore nullable historical rows.

- [ ] **Step 6: Implement PostgreSQL HNSW recall and SQLite linear test recall**

For PostgreSQL, use a nearest-view subquery and group it:

```python
distance = cast(
    self.source_group_embeddings.c.embedding,
    Vector(384),
).cosine_distance(query_embedding.tolist())
nearest = (
    select(
        self.source_group_embeddings.c.source_group_id,
        distance.label("distance"),
    )
    .order_by(distance)
    .limit(neighbor_limit)
    .subquery()
)
query = (
    select(
        nearest.c.source_group_id,
        func.min(nearest.c.distance).label("best_distance"),
        func.count().label("matching_views"),
    )
    .group_by(nearest.c.source_group_id)
    .order_by(func.min(nearest.c.distance), func.count().desc(), nearest.c.source_group_id)
    .limit(group_limit)
)
```

The SQLite branch reads only embedding rows and computes cosine distance in NumPy for unit tests. It must not inspect recent trace IDs in either branch.

- [ ] **Step 7: Add a real pgvector ranking test**

In `tests/test_visual_index_postgresql.py`, create 50 groups with deterministic normalized 384-dimensional vectors, place the target close to the query, call `recall_source_groups(query, group_limit=10, neighbor_limit=50)`, and assert the target is first. Assert the query plan contains `ix_source_group_embeddings_hnsw` after `SET LOCAL hnsw.ef_search = 100` and `enable_seqscan = off` inside the test transaction. Add two concurrent transactions inserting the same source hash and assert one source group, ten embeddings, and both records remain.

Run: `python -m pytest tests/test_visual_index_repository.py tests/test_visual_index_postgresql.py -q`

Expected: all repository and real pgvector ranking tests pass.

- [ ] **Step 8: Commit the visual repository**

```powershell
git add database_store.py trace_app/database/repositories.py trace_app/database/visual_index.py tests/test_visual_index_repository.py tests/test_visual_index_postgresql.py
git commit -m "feat: persist and recall visual source groups"
```

### Task 5: Extract One Aligned Observation Per Source Group

**Files:**
- Modify: `watermark_v4/detector.py`
- Test: `tests/test_watermark_v4_detector.py`
- Test: `tests/test_grouped_v4_detection.py`

- [ ] **Step 1: Write a failing observation-reuse test**

```python
from unittest.mock import patch

from watermark_v4.detector import (
    AuthenticationCandidate,
    SourceGroupCandidate,
    detect_v4_source_groups,
)


def test_group_versions_share_one_aligned_observation() -> None:
    image, marked_candidate = _marked_candidate()
    group = SourceGroupCandidate(
        source_group_id="group-a",
        feature_index=marked_candidate.feature_index,
        records=(
            AuthenticationCandidate("wrong", "TR-WRONG", bytes.fromhex("00000000")),
            AuthenticationCandidate(marked_candidate.record_id, marked_candidate.trace_id, marked_candidate.auth_tag),
            AuthenticationCandidate("wrong-2", "TR-WRONG-2", bytes.fromhex("ffffffff")),
        ),
    )
    with patch("watermark_v4.detector.prepare_aligned_observation", wraps=prepare_aligned_observation) as prepare:
        result = detect_v4_source_groups(image, (group,), V4Config())

    assert result is not None
    assert result.record_id == marked_candidate.record_id
    assert prepare.call_count == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_grouped_v4_detection.py::test_group_versions_share_one_aligned_observation -q`

Expected: collection fails because grouped candidate types and detection do not exist.

- [ ] **Step 3: Introduce immutable observation and candidate types**

```python
@dataclass(frozen=True, slots=True)
class AlignedObservation:
    observed: bytes
    byte_confidences: tuple[float, ...]
    tile_count: int
    phase_count: int
    minimum_coverage: float
    mean_abs_score: float


@dataclass(frozen=True, slots=True)
class AuthenticationCandidate:
    record_id: str
    trace_id: str
    auth_tag: bytes


@dataclass(frozen=True, slots=True)
class SourceGroupCandidate:
    source_group_id: str
    feature_index: FeatureIndex
    records: tuple[AuthenticationCandidate, ...]


class AuthenticationAmbiguityError(RuntimeError):
    pass
```

Validate exact tag length, unique record IDs, nonempty group IDs, and exact `FeatureIndex` instances.

- [ ] **Step 4: Split pixel extraction from code comparison**

Move lines currently responsible for `warpPerspective`, coverage, DCT tile scores, phase aggregation, observed bytes, and byte confidences into:

```python
def prepare_aligned_observation(
    image: Image.Image,
    query_to_target: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    config: V4Config,
    deadline: float | None = None,
) -> AlignedObservation | None:
```

Add the inexpensive comparison:

```python
def authenticate_observation(
    observation: AlignedObservation,
    candidate: AuthenticationCandidate,
) -> CandidateEvidence | None:
    decoded = decode_candidate_codeword(
        observation.observed,
        candidate.auth_tag,
        observation.byte_confidences,
    )
    if decoded is None:
        return None
    return CandidateEvidence(
        record_id=candidate.record_id,
        trace_id=candidate.trace_id,
        tile_count=observation.tile_count,
        phase_count=observation.phase_count,
        minimum_coverage=observation.minimum_coverage,
        corrected_symbols=decoded.corrected_symbols,
        erasure_count=decoded.erasure_count,
        bit_errors=decoded.bit_errors,
        mean_abs_score=observation.mean_abs_score,
    )
```

Keep `decode_aligned_candidate` as a compatibility wrapper that prepares once and authenticates one `V4Candidate`, preserving existing tests and public imports.

- [ ] **Step 5: Implement complete grouped detection**

`detect_v4_source_groups` must extract the query ORB index once, try ORB/RANSAC for every recalled group, prepare one observation for each accepted geometry, authenticate every record in that group, and collect successes globally. Return `None` for zero successes and raise `AuthenticationAmbiguityError` for more than one. Do not slice `group.records` and do not accept `recent_record_ids`.

```python
authenticated: list[tuple[CandidateEvidence, FeatureMatch]] = []
for group in groups:
    match = match_feature_indexes(query_index, group.feature_index)
    if match is None:
        continue
    observation = prepare_aligned_observation(
        image,
        match.query_to_target,
        target_width=group.feature_index.image_width,
        target_height=group.feature_index.image_height,
        config=config,
        deadline=effective_deadline,
    )
    if observation is None:
        continue
    for record in group.records:
        evidence = authenticate_observation(observation, record)
        if evidence is not None:
            authenticated.append((evidence, match))
if len(authenticated) > 1:
    raise AuthenticationAmbiguityError("multiple records authenticated")
```

Retain the current constrained FFT/ORB geometry fallback and translation refinements, but each distinct geometry may prepare at most one observation before trying every record.

- [ ] **Step 6: Add no-cap, ambiguity, and timeout tests**

Add tests with 12 group records to prove all 12 authentication tags are evaluated, two matching tags to prove ambiguity raises, and a deadline that expires midway to prove no partial success is returned.

Run: `python -m pytest tests/test_grouped_v4_detection.py tests/test_watermark_v4_detector.py -q`

Expected: all grouped and existing V4 detector tests pass, including the user's current uncommitted screenshot regression.

- [ ] **Step 7: Commit the decoder refactor**

```powershell
git add watermark_v4/detector.py tests/test_watermark_v4_detector.py tests/test_grouped_v4_detection.py
git commit -m "feat: authenticate complete V4 source groups"
```

### Task 6: Orchestrate Model Recall, Geometry, and Authentication

**Files:**
- Create: `trace_app/watermark/visual_authentication.py`
- Modify: `trace_app/runtime.py`
- Test: `tests/test_visual_authentication_service.py`

- [ ] **Step 1: Write failing orchestration and recent-list invariance tests**

```python
def test_detection_uses_recall_groups_not_recent_ids() -> None:
    repository = FakeVisualRepository(recalled=("group-target",))
    detector = FakeGroupedDetector(record_id="record-target")
    service = VisualAuthenticationService(repository, FakeEmbedder(), detector=detector)
    query = Image.new("RGB", (320, 240), "white")

    empty = service.detect(query, recent_trace_ids=())
    wrong = service.detect(query, recent_trace_ids=("TR-WRONG",))
    target = service.detect(query, recent_trace_ids=("TR-TARGET",))

    assert empty == wrong == target
    assert repository.recall_calls == 3
    assert detector.group_ids == ["group-target", "group-target", "group-target"]


def test_detection_loads_every_auth_record_for_confirmed_groups() -> None:
    repository = FakeVisualRepository(recalled=("group-a",), version_count=17)
    detector = FakeGroupedDetector(record_id="record-16")
    service = VisualAuthenticationService(repository, FakeEmbedder(), detector=detector)

    service.detect(Image.new("RGB", (320, 240), "white"), recent_trace_ids=())

    assert detector.record_count == 17
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_visual_authentication_service.py -q`

Expected: collection fails because `VisualAuthenticationService` does not exist.

- [ ] **Step 3: Add a bounded source-group geometry cache**

Extend `Runtime` with an `OrderedDict[str, FeatureIndex]` named `visual_group_cache` and a lock. Implement cache `get` and `put` helpers with a maximum of 64 groups. Recent trace IDs may call `repository.group_ids_for_trace_ids` and populate this cache, but cache contents cannot create recall hits.

- [ ] **Step 4: Implement the orchestration service**

```python
class VisualAuthenticationService:
    def __init__(self, repository, embedder, *, config, runtime, detector=detect_v4_source_groups):
        self.repository = repository
        self.embedder = embedder
        self.config = config
        self.runtime = runtime
        self.detector = detector

    def detect(self, image: Image.Image, *, recent_trace_ids: tuple[str, ...]):
        started = monotonic()
        query_embedding = self.embedder.embed((image,))[0]
        hits = self.repository.recall_source_groups(
            query_embedding,
            group_limit=self.config.recall_limit,
            neighbor_limit=self.config.recall_neighbors,
        )
        group_ids = tuple(hit.source_group_id for hit in hits)
        if not group_ids:
            return None
        self._prewarm_recent_cache(recent_trace_ids, allowed_group_ids=frozenset(group_ids))
        features = self.repository.load_source_group_features(group_ids)
        records = self.repository.auth_records_for_groups(group_ids)
        groups = _join_groups_in_recall_order(group_ids, features, records)
        result = self.detector(
            image,
            groups,
            self.config.v4,
            deadline=started + self.config.timeout_seconds,
        )
        if result is None:
            return None
        return self.repository.get_record(result.record_id), result
```

Use this dedicated configuration and record stage durations through the module logger with stable fields: `model_ms`, `vector_ms`, `geometry_auth_ms`, `recalled_groups`, `loaded_versions`, `recent_cache_hits`, and `outcome`.

```python
@dataclass(frozen=True, slots=True)
class VisualAuthenticationConfig:
    recall_limit: int
    recall_neighbors: int
    timeout_seconds: float
    v4: V4Config

    def __post_init__(self) -> None:
        if not 1 <= self.recall_limit <= 100:
            raise ValueError("recall_limit must be between 1 and 100")
        if self.recall_neighbors < self.recall_limit:
            raise ValueError("recall_neighbors must cover recall_limit")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
```

- [ ] **Step 5: Map ambiguity and timeout without partial attribution**

Expose service exceptions `VisualAuthenticationAmbiguous` and `VisualAuthenticationTimeout`. Translate `AuthenticationAmbiguityError` and detector `TimeoutError` without returning record JSON. Unit tests assert no `get_record` call occurs for either error.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_visual_authentication_service.py -q`

Expected: all orchestration, cache, complete-version, ambiguity, and timeout tests pass.

```powershell
git add trace_app/runtime.py trace_app/watermark/visual_authentication.py tests/test_visual_authentication_service.py
git commit -m "feat: orchestrate visual source authentication"
```

### Task 7: Integrate Indexed V4 Generation and Detection

**Files:**
- Modify: `trace_app/watermark/service.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/application.py`
- Modify: `trace_app/compat.py`
- Test: `tests/test_application_structure.py`
- Test: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write failing service tests for source reuse and authentication collisions**

```python
def test_second_v4_generation_reuses_existing_source_index(service_fixture) -> None:
    operations, calls = service_fixture.operations_with_visual_index()
    service = service_fixture.service(operations)

    first = service_fixture.embed_v4(service, source_seed=17)
    second = service_fixture.embed_v4(service, source_seed=17)

    assert first["source_group_id"] == second["source_group_id"]
    assert calls["model_batches"] == 1
    assert calls["inserted_records"] == 2


def test_v4_generation_retries_group_authentication_collision(service_fixture) -> None:
    operations, calls = service_fixture.operations_with_visual_index(
        insert_collision_tags={bytes.fromhex("01020304")}
    )
    operations = replace(
        operations,
        v4_authentication_tag=Mock(
            side_effect=(bytes.fromhex("01020304"), bytes.fromhex("05060708"))
        ),
    )
    service = service_fixture.service(operations)

    result = service_fixture.embed_v4(service, source_seed=18)

    assert result["robust_auth_code"] == "05060708"
    assert operations.v4_authentication_tag.call_count == 2
    assert calls["v4_embeds"] == 2
```

- [ ] **Step 2: Run focused service tests and verify RED**

Run: `python -m pytest tests/test_application_structure.py -k "source_index or authentication_collision" -q`

Expected: tests fail because visual source preparation and group-aware identity generation are not integrated.

- [ ] **Step 3: Extend service dependencies without breaking legacy test factories**

Add these optional fields at the end of `WatermarkOperations`:

```python
prepare_visual_source: Callable[[Image.Image, str], object] | None = None
visual_auth_code_exists: Callable[[str, bytes], bool] | None = None
add_visual_record: Callable[[dict[str, Any], object, int | None], None] | None = None
find_indexed_file_fingerprint: Callable[[bytes], dict[str, Any] | None] | None = None
detect_indexed_v4: Callable[[Image.Image, tuple[str, ...]], dict[str, Any] | None] | None = None
has_indexed_v4_records: Callable[[], bool] | None = None
```

Keep them optional so isolated legacy unit tests still construct the dataclass. Production PostgreSQL startup must provide all six callbacks as a set.

- [ ] **Step 4: Prepare or reuse a source group before V4 embedding**

Implement `VisualAuthenticationService.prepare_source(image, original_image_sha256)`. It first calls `repository.find_source_group_by_hash`; if found, return a lightweight `ExistingSourceGroup`. Otherwise run `source_views`, one batched embed call, and `extract_feature_index(image)` to create `PreparedSourceGroup` with a UUID group ID.

In `WatermarkService.embed`, calculate the decoded RGB hash before choosing a V4 identity. For V4 with visual callbacks, loop up to 16 generated trace IDs until `visual_auth_code_exists(original_image_sha256, tag)` is false. Raise HTTP 503 if no unique tag is found; do not embed a known collision.

- [ ] **Step 5: Persist group and record through one callback**

Refactor the identity-dependent V4 pilot/codeword embedding, saved output, fingerprints, and record construction into `_build_v4_attempt`. Run up to four attempts. After each attempt call:

```python
if robust_version == op.robust_version_v4 and visual_source is not None:
    if op.add_visual_record is None:
        raise HTTPException(status_code=503, detail="视觉索引服务不可用")
    op.add_visual_record(record, visual_source, owner_user_id)
else:
    self.repository.add_record(record, owner_user_id=owner_user_id)
```

If the callback raises the named `DuplicateSourceAuthenticationError`, remove only that attempt's watermarked output and thumbnail, generate a new trace ID/tag, and repeat `_build_v4_attempt`. After four database collisions, return HTTP 503. Do not catch other integrity errors. Set `record["source_group_id"]` before persistence and stop writing a per-record V4 `.npz` feature file for indexed V4 records. Preserve legacy per-record feature files for V1-V3.

- [ ] **Step 6: Replace full-scan extraction for indexed V4 data**

In `extract_upload`, call `find_indexed_file_fingerprint(content)` before any `read_records`. It computes file hashes once, performs the indexed query, adds existing evidence fields, records success, and returns.

In `extract_image`, call `detect_indexed_v4(image, tuple(runtime.generated_trace_ids))` before loading legacy records. If indexed V4 records exist and indexed detection returns no result, record a failed attempt and raise the existing 404. This preserves the existing rule that a V4 negative cannot fall through to legacy visual attribution.

Map `VisualAuthenticationAmbiguous` to HTTP 409 with `认证码匹配不唯一`; map `VisualAuthenticationTimeout` to HTTP 504 with `水印认证超时`; keep service/inference failures as HTTP 503.

- [ ] **Step 7: Build real callbacks only for PostgreSQL production**

In `application.create_app`, construct `DinoV2Embedder` and `VisualAuthenticationService` when the runtime store dialect is PostgreSQL. Missing model/checksum or pgvector readiness must fail application startup. When database initialization is disabled under pytest, or isolated tests explicitly use SQLite, leave callbacks unset and retain the existing in-memory V4 compatibility path.

Update `build_default_operations` and `compat.py` to accept an optional visual authenticator and bind the six callbacks. Do not instantiate ONNX Runtime at module import time.

- [ ] **Step 8: Add API tests proving no full record read**

In the PostgreSQL-capable V4 API fixture, monkeypatch `repository.read_records` to raise, upload a transformed indexed V4 image, and assert successful attribution. Add tests for a target trace removed from `generated_trace_ids`, ambiguity 409, timeout 504, and indexed negative 404 without legacy fallback.

Run: `python -m pytest tests/test_application_structure.py tests/test_watermark_v4_api.py -q`

Expected: all existing and new service/API tests pass.

- [ ] **Step 9: Commit service integration**

```powershell
git add trace_app/watermark/service.py trace_app/watermark/default_operations.py trace_app/application.py trace_app/compat.py tests/test_application_structure.py tests/test_watermark_v4_api.py
git commit -m "feat: route V4 tracing through visual source groups"
```

### Task 8: Complete Deletion, Cache, and Observability Semantics

**Files:**
- Modify: `database_store.py`
- Modify: `trace_app/database/repositories.py`
- Modify: `trace_app/management/service.py`
- Modify: `trace_app/watermark/visual_authentication.py`
- Test: `tests/test_database_store.py`
- Test: `tests/test_image_ownership.py`
- Test: `tests/test_visual_authentication_service.py`

- [ ] **Step 1: Write failing last-version deletion tests**

```python
def test_deleting_last_version_removes_source_indexes(visual_repository) -> None:
    source = prepared_source(21, "c" * 64)
    visual_repository.add_visual_record(record("record-a", "01020304"), source=source)
    visual_repository.add_visual_record(record("record-b", "05060708"), source=source)

    visual_repository.delete_record("record-a")
    assert visual_repository.source_group_count() == 1
    assert visual_repository.source_embedding_count(source.source_group_id) == 10

    visual_repository.delete_record("record-b")
    assert visual_repository.source_group_count() == 0
    assert visual_repository.source_embedding_count(source.source_group_id) == 0
    assert visual_repository.source_feature_count(source.source_group_id) == 0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_database_store.py -k "last_version" -q`

Expected: FAIL because deletion leaves the source group and indexes.

- [ ] **Step 3: Delete orphaned source groups in the record transaction**

Select `source_group_id` with the record before deletion. After deleting and compacting position indexes, count remaining group records. Delete `source_groups` only when the count is zero; foreign-key cascades remove embeddings/features. Return the same record JSON shape as before.

- [ ] **Step 4: Evict deleted source groups from the cache**

Have repository deletion return `(record, removed_source_group_id)` internally while preserving the public management result. Call `runtime.visual_group_cache.pop(group_id, None)` only when the group was removed. Add a management test proving deleting one of multiple versions does not evict the shared group.

- [ ] **Step 5: Emit stable stage metrics without sensitive data**

Use one event per detection with concrete stable keys:

```python
logger.info(
    "visual_authentication",
    extra={
        "model_ms": round(model_ms, 3),
        "vector_ms": round(vector_ms, 3),
        "geometry_auth_ms": round(geometry_auth_ms, 3),
        "recalled_groups": recalled_groups,
        "loaded_versions": loaded_versions,
        "recent_cache_hits": recent_cache_hits,
        "outcome": outcome,
    },
)
```

Never log embeddings, authentication codes, database URLs, file bytes, or user credentials. Add a `caplog` test asserting the required keys and absence of `auth_tag`/`embedding`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_database_store.py tests/test_image_ownership.py tests/test_visual_authentication_service.py -q`

Expected: all delete, ownership, cache, and logging tests pass.

```powershell
git add database_store.py trace_app/database/repositories.py trace_app/management/service.py trace_app/watermark/visual_authentication.py tests/test_database_store.py tests/test_image_ownership.py tests/test_visual_authentication_service.py
git commit -m "feat: clean visual indexes with final source version"
```

### Task 9: Add the 50-Source Real Model and pgvector Regression

**Files:**
- Create: `tests/model_visual_recall_benchmark.py`
- Create: `tests/test_model_visual_recall_postgresql.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write the deterministic synthetic image generator**

```python
def synthetic_source(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    randomizer = random.Random(seed)
    image = Image.new("RGB", size, (245, 245, 242))
    draw = ImageDraw.Draw(image)
    for index in range(24):
        x = randomizer.randrange(10, size[0] - 90)
        y = randomizer.randrange(10, size[1] - 70)
        width = randomizer.randrange(24, 110)
        height = randomizer.randrange(18, 90)
        color = tuple(randomizer.randrange(25, 230) for _ in range(3))
        if index % 2:
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        else:
            draw.ellipse((x, y, x + width, y + height), outline=color, width=3)
    draw.text((24, 22), f"SOURCE {seed:03d}", fill=(15, 15, 15), stroke_width=1)
    for row in range(0, size[1], 12):
        shade = 210 + ((row // 12 + seed) % 24)
        draw.line((0, row, size[0], row), fill=(shade, shade, shade), width=1)
    return image
```

Add transformation helpers for a qualifying 25%-content crop, 0.5x resize, JPEG quality 70, and rotations of plus/minus 5 degrees.

- [ ] **Step 2: Write a failing real-model Recall@10 test**

Prepare embeddings for 50 fixed seeds, insert ten views per source into a temporary PostgreSQL schema, transform each query, and assert the correct group is within the first ten results. Record results by attack type.

Run: `python -m pytest tests/test_model_visual_recall_postgresql.py::test_real_model_recall_at_ten -q -m model_integration`

Expected before model preparation: FAIL at startup with a clear missing-model error. After running `python tools/prepare_visual_model.py`, the test executes against the real ONNX model.

- [ ] **Step 3: Add end-to-end complete-version attribution**

Create one indexed V4 record for every source group and three additional records for a selected source. To control runtime, source groups not used for authentication may insert model/ORB indexes plus valid record metadata without embedding redundant watermarks. Generate real V4 pixels for the four selected-source versions, query each version through crop/JPEG/resize transforms, and assert:

```python
assert result["source_group_id"] == target_group_id
assert result["id"] == expected_record_id
assert metrics["loaded_versions"] == 4
assert expected_record_id not in recently_generated_ids
```

Add 50 unrelated negative images and require zero successful attributions.

- [ ] **Step 4: Enforce agreed quality gates**

The benchmark returns nonzero if any gate fails:

```python
assert exact_success_rate == 1.0
assert full_resize_jpeg_recall_at_10 >= 0.99
assert qualifying_crop_recall_at_10 >= 0.95
assert transformed_final_attribution_rate >= 0.90
assert negative_false_attributions == 0
```

Write a JSON report to `test_output/model-visual-recall/report.json` containing seed, case counts, rates, and p50/p95 stage durations. Do not commit generated reports.

- [ ] **Step 5: Run the focused real integration suite**

Run: `python -m pytest tests/test_model_visual_recall_postgresql.py tests/test_watermark_v4_api.py -q -m "model_integration or not model_integration"`

Expected: all tests pass; the model test reports approximately 50 source groups, top-10 recall gates, four complete-version candidates, and zero false attributions.

- [ ] **Step 6: Commit the benchmark**

```powershell
git add tests/model_visual_recall_benchmark.py tests/test_model_visual_recall_postgresql.py tests/test_watermark_v4_api.py
git commit -m "test: benchmark model visual recall across 50 sources"
```

### Task 10: Package, Deploy, Document, and Verify

**Files:**
- Modify: `tools/build_centos_release.py`
- Modify: `tests/test_release_builder.py`
- Modify: `deploy.sh`
- Modify: `README.md`
- Modify: `README_DEPLOY.md`

- [ ] **Step 1: Write failing release model-asset tests**

```python
def test_release_requires_model_and_checksum_assets(tmp_path: Path) -> None:
    _write_required_root_files(tmp_path)
    for tree in builder.RECURSIVE_TREES:
        (tmp_path / tree).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="dinov2-small.onnx"):
        release_files(tmp_path)


def test_release_contains_model_and_checksum(tmp_path: Path) -> None:
    _write_required_root_files(tmp_path)
    for tree in builder.RECURSIVE_TREES:
        (tmp_path / tree).mkdir(parents=True, exist_ok=True)
    model = tmp_path / "models/dinov2-small.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    model.with_suffix(".onnx.sha256").write_text(digest + "\n", encoding="ascii")
    paths = {path.as_posix() for path in release_files(tmp_path)}
    assert "models/dinov2-small.onnx" in paths
    assert "models/dinov2-small.onnx.sha256" in paths
```

- [ ] **Step 2: Run release tests and verify RED**

Run: `python -m pytest tests/test_release_builder.py -k model -q`

Expected: tests fail because model assets are not required or collected.

- [ ] **Step 3: Include verified model assets in releases**

Add both model paths to `ROOT_FILES`. Before copying, verify the sidecar digest against the ONNX bytes and fail the build on mismatch. Keep `.env`, business data, uploads, and secrets excluded.

- [ ] **Step 4: Update deployment for PostgreSQL and pgvector**

Change `deploy.sh check-db` and service installation checks to require a PostgreSQL `DB_URL`, query `pg_extension` for `vector`, and execute the application schema readiness check. Install runtime dependencies from the updated requirements, verify the packaged model checksum, and fail before restarting systemd if either dependency is unavailable. Do not auto-download the model on the production host.

- [ ] **Step 5: Document the new operational contract**

Update README documents with PostgreSQL/pgvector requirements, `python tools/prepare_visual_model.py` before local start or release build, checksum/package-size behavior, source-group semantics, no history backfill, detection order, recent IDs as cache-only state, benchmark commands, and the 404/409/504/503 outcomes.

- [ ] **Step 6: Run complete fresh verification**

Run these commands in order and inspect every exit code:

```powershell
python -m pytest tests/test_visual_model_setup.py tests/test_visual_embeddings.py tests/test_database_store.py tests/test_visual_schema_postgresql.py tests/test_visual_index_repository.py tests/test_visual_index_postgresql.py tests/test_grouped_v4_detection.py tests/test_visual_authentication_service.py tests/test_application_structure.py tests/test_watermark_v4_api.py tests/test_image_ownership.py tests/test_release_builder.py -q
python -m pytest tests/test_model_visual_recall_postgresql.py -q -m model_integration
python -m pytest -q
python tools/build_centos_release.py
```

Expected: focused tests pass, the real model/pgvector quality gates pass, the full suite has zero failures, and release creation exits zero after verifying the ONNX checksum.

- [ ] **Step 7: Inspect requirement coverage and commit**

Confirm from the benchmark report and tests that vector recall returns tens of groups from a large index, RANSAC is mandatory, every group version is checked, exactly one auth result is returned, and recent-list contents do not affect outcomes.

```powershell
git add tools/build_centos_release.py tests/test_release_builder.py deploy.sh README.md README_DEPLOY.md
git commit -m "docs: deploy model visual recall pipeline"
```
