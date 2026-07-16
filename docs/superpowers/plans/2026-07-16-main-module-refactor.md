# Main Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3,946-line `main.py` with a thin FastAPI entry point while separating configuration, persistence, authentication, image processing, watermark algorithms, services, and routers without changing observable behavior.

**Architecture:** Build a `trace_app` package around a FastAPI application factory. Routers depend on services, services depend on repositories and focused watermark/image modules, and algorithm modules remain unaware of FastAPI and persistence. A compatibility module re-exports the legacy public Python API while tests patch the module that owns each dependency.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy, Pillow, NumPy, OpenCV, PyWavelets, pytest, httpx

---

## File Map

- `main.py`: deployment entry point; create `app` and re-export compatibility names only.
- `trace_app/config.py`: immutable settings, environment parsing, path resolution, and algorithm constants.
- `trace_app/runtime.py`: application-scoped database and generated-trace state.
- `trace_app/application.py`: `create_app()`, startup initialization, static mounts, and router registration.
- `trace_app/database/connection.py`: SQLAlchemy engine creation, schema creation, and default data seeding.
- `trace_app/database/repositories.py`: records, statistics, roles, and users persistence adapter.
- `trace_app/auth/service.py`: login, roles, users, and menu rules.
- `trace_app/watermark/lsb.py`: payload packets and LSB carriers.
- `trace_app/watermark/frequency.py`: legacy DCT, DWT, and FFT layers.
- `trace_app/watermark/dot_matrix.py`: dot-matrix embed and detection.
- `trace_app/watermark/small_crop.py`: small-crop and code-tile embed/detection.
- `trace_app/watermark/robust.py`: robust v1-v3 embed and aligned decoders.
- `trace_app/watermark/detection.py`: ordered extraction and fallback orchestration helpers.
- `trace_app/watermark/service.py`: API-level embed/extract use cases.
- `trace_app/imaging/io.py`: upload, byte, and URL image loading.
- `trace_app/imaging/fingerprints.py`: content/path hashes and stored-file fingerprint matching.
- `trace_app/imaging/feature_matching.py`: candidate indexes, alignment, residuals, and visual matching.
- `trace_app/imaging/visible_mark.py`: visible copyright drawing and detection.
- `trace_app/api/auth.py`: login router.
- `trace_app/api/users.py`: roles and users router.
- `trace_app/api/watermark.py`: watermark embed/extract routers.
- `trace_app/api/images.py`: image listing and deletion router.
- `trace_app/api/dashboard.py`: dashboard statistics and development reset router.
- `trace_app/compat.py`: legacy public symbol exports, with no business implementation.
- `tests/test_application_structure.py`: entry-point, factory, route, import-side-effect, and compatibility contracts.
- Existing `tests/test_*.py`: update monkeypatch targets as symbols acquire a new owner.

### Task 1: Freeze the Existing Application Contract

**Files:**
- Create: `tests/test_application_structure.py`
- Inspect: `main.py`

- [ ] **Step 1: Write the entry-point and route characterization tests**

```python
from pathlib import Path

import main


EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/site-logo.png"),
    ("GET", "/favicon.ico"),
    ("GET", "/favico.ico"),
    ("POST", "/auth/login"),
    ("GET", "/api/roles"),
    ("PUT", "/api/roles/{role_key}"),
    ("GET", "/api/users"),
    ("POST", "/api/users"),
    ("PUT", "/api/users/{username}"),
    ("DELETE", "/api/users/{username}"),
    ("POST", "/api/watermark/embed"),
    ("POST", "/api/watermark/extract"),
    ("POST", "/api/watermark/extract-url"),
    ("GET", "/api/dashboard-stats"),
    ("GET", "/api/images"),
    ("DELETE", "/api/images/{image_id}"),
    ("POST", "/api/dev/reset"),
}


def test_existing_route_contract_is_frozen() -> None:
    actual = {
        (method, route.path)
        for route in main.app.routes
        for method in getattr(route, "methods", set())
    }
    assert EXPECTED_ROUTES <= actual


def test_main_currently_exposes_required_python_api() -> None:
    required = {
        "app",
        "ensure_dirs",
        "embed_robust_watermark",
        "embed_robust_watermark_v2",
        "embed_robust_watermark_v3",
        "detect_aligned_authenticated_watermark",
        "extract_watermark_from_image",
        "align_query_to_record",
        "file_sha256",
    }
    assert required <= set(dir(main))


def test_target_entry_point_has_no_business_definitions() -> None:
    source = Path("main.py").read_text(encoding="utf-8")
    # This assertion is activated in Task 10 after the final entry point is written.
    assert "def embed_watermark(" in source
```

- [ ] **Step 2: Run the characterization tests**

Run: `pytest tests/test_application_structure.py -v`

Expected: all three tests pass against the current monolith.

- [ ] **Step 3: Record the pre-refactor focused baseline**

Run: `pytest tests/test_watermark_v4_api.py tests/test_aligned_authenticated_detection.py tests/test_prominent_corner_copyright.py -q`

Expected: exit code 0. Save the passed/failed count in the implementation notes before moving code; existing failures must be reported and not hidden by the refactor.

- [ ] **Step 4: Commit the characterization tests**

```powershell
git add tests/test_application_structure.py
git commit -m "test: freeze main application contract"
```

### Task 2: Introduce Configuration and Runtime Boundaries

**Files:**
- Create: `trace_app/__init__.py`
- Create: `trace_app/config.py`
- Create: `trace_app/runtime.py`
- Modify: `main.py`
- Modify: `tests/test_application_structure.py`

- [ ] **Step 1: Write failing settings and runtime tests**

Append to `tests/test_application_structure.py`:

```python
from trace_app.config import Settings
from trace_app.runtime import Runtime


def test_settings_resolve_relative_directories_from_base(tmp_path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )
    assert settings.upload_dir == tmp_path / "uploads"
    assert settings.data_dir == tmp_path / "data"
    assert settings.original_dir == tmp_path / "uploads" / "originals"


def test_runtime_keeps_mutable_resources_out_of_settings() -> None:
    runtime = Runtime()
    assert runtime.engine is None
    assert runtime.store is None
    assert runtime.generated_trace_ids == []
```

- [ ] **Step 2: Verify the tests fail because the package does not exist**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'trace_app'`.

- [ ] **Step 3: Implement immutable settings and application runtime**

Create `trace_app/__init__.py` as an empty package marker. Create `trace_app/config.py` with the current constants moved unchanged after this settings shell:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    upload_dir: Path
    data_dir: Path
    db_url: str
    admin_user: str
    admin_pass: str
    app_name: str = "WatermarkSystem"

    @property
    def original_dir(self) -> Path:
        return self.upload_dir / "originals"

    @property
    def watermarked_dir(self) -> Path:
        return self.upload_dir / "watermarked"

    @property
    def thumbnail_dir(self) -> Path:
        return self.upload_dir / "thumbnails"

    @classmethod
    def from_values(
        cls,
        *,
        base_dir: Path,
        upload_dir: str,
        data_dir: str,
        db_url: str,
        admin_user: str,
        admin_pass: str,
    ) -> "Settings":
        upload_path = Path(upload_dir)
        data_path = Path(data_dir)
        return cls(
            base_dir=base_dir,
            upload_dir=upload_path if upload_path.is_absolute() else base_dir / upload_path,
            data_dir=data_path if data_path.is_absolute() else base_dir / data_path,
            db_url=db_url.strip(),
            admin_user=admin_user.strip(),
            admin_pass=admin_pass,
            app_name=os.getenv("APP_NAME", "WatermarkSystem"),
        )


BASE_DIR = Path(__file__).resolve().parent.parent
settings = Settings.from_values(
    base_dir=BASE_DIR,
    upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
    data_dir=os.getenv("DATA_DIR", "./data"),
    db_url=os.getenv("DB_URL", ""),
    admin_user=os.getenv("ADMIN_USER", ""),
    admin_pass=os.getenv("ADMIN_PASS", ""),
)
```

Move the constant declarations from `main.py:69-146` into `trace_app/config.py` without changing values. Create `trace_app/runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from database_store import DatabaseStore


@dataclass(slots=True)
class Runtime:
    engine: Engine | None = None
    store: DatabaseStore | None = None
    db_error: str = ""
    generated_trace_ids: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Replace path and constant definitions in `main.py` with imports**

```python
from trace_app.config import *
from trace_app.config import settings

UPLOAD_DIR = settings.upload_dir
DATA_DIR = settings.data_dir
ORIGINAL_DIR = settings.original_dir
WATERMARKED_DIR = settings.watermarked_dir
THUMBNAIL_DIR = settings.thumbnail_dir
DB_URL = settings.db_url
ADMIN_USER = settings.admin_user
ADMIN_PASS = settings.admin_pass
```

Keep the aliases temporarily because later tasks move their consumers and compatibility exports.

- [ ] **Step 5: Run the settings and focused regression tests**

Run: `pytest tests/test_application_structure.py tests/test_watermark_v4_api.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit the configuration boundary**

```powershell
git add trace_app main.py tests/test_application_structure.py
git commit -m "refactor: extract application configuration"
```

### Task 3: Extract Database Connection and Repositories

**Files:**
- Create: `trace_app/database/__init__.py`
- Create: `trace_app/database/connection.py`
- Create: `trace_app/database/repositories.py`
- Modify: `main.py`
- Modify: `tests/test_application_structure.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write failing repository tests**

Append to `tests/test_application_structure.py`:

```python
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine

from database_store import DatabaseStore
from trace_app.database.repositories import Repository


def test_repository_rejects_access_without_store() -> None:
    repository = Repository(None)
    with pytest.raises(HTTPException) as exc:
        repository.read_records()
    assert exc.value.status_code == 503


def test_repository_round_trips_records() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    repository = Repository(store)
    repository.replace_records([{"id": "one"}])
    assert repository.read_records() == [{"id": "one"}]
```

- [ ] **Step 2: Verify the repository import fails**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails because `trace_app.database.repositories` is absent.

- [ ] **Step 3: Implement connection creation and the repository adapter**

Create `trace_app/database/connection.py` with `create_runtime(settings, enabled=True) -> Runtime`, using the current `initialize_database()` engine arguments, `DatabaseStore.create_schema()`, and default role/admin seeding behavior. The concrete entry point is:

```python
def create_runtime(settings: Settings, *, enabled: bool = True) -> Runtime:
    runtime = Runtime()
    if not enabled:
        return runtime
    missing = [
        name
        for name, value in (
            ("DB_URL", settings.db_url),
            ("ADMIN_USER", settings.admin_user),
            ("ADMIN_PASS", settings.admin_pass),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable: {missing[0]}")
    try:
        runtime.engine = create_engine(
            settings.db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )
        runtime.store = DatabaseStore(runtime.engine)
        runtime.store.create_schema()
        seed_database_defaults(runtime.store, settings)
        return runtime
    except SQLAlchemyError as exc:
        runtime.db_error = type(exc).__name__
        runtime.store = None
        raise RuntimeError("Database initialization failed") from exc
```

Create `Repository` in `trace_app/database/repositories.py`. Move the bodies of `read_records`, `write_records`, `add_record`, detection/watermark statistics, `read_roles`, `read_users`, and `db_clear_all` into instance methods. Use this store guard:

```python
class Repository:
    def __init__(self, store: DatabaseStore | None) -> None:
        self._store = store

    @property
    def store(self) -> DatabaseStore:
        if self._store is None:
            raise HTTPException(status_code=503, detail="数据库不可用")
        return self._store

    def read_records(self) -> list[dict[str, Any]]:
        return self.store.read_records()

    def replace_records(self, records: list[dict[str, Any]]) -> None:
        self.store.replace_records(records)
```

- [ ] **Step 4: Delegate the temporary `main.py` database functions to one repository**

```python
runtime = create_runtime(settings, enabled=DB_ENABLED)
repository = Repository(runtime.store)
db_engine = runtime.engine
db_store = runtime.store
db_error = runtime.db_error


def read_records() -> list[dict[str, Any]]:
    return repository.read_records()
```

Apply the same one-line delegation to every persistence helper moved in Step 3. In `tests/test_watermark_v4_api.py`, replace `monkeypatch.setattr(main, "db_store", store)` with setting `main.runtime.store = store`, `main.repository = Repository(store)`, and the owning service dependency once Task 8 introduces it.

- [ ] **Step 5: Run persistence and API tests**

Run: `pytest tests/test_database_store.py tests/test_application_structure.py tests/test_watermark_v4_api.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit the persistence boundary**

```powershell
git add trace_app/database main.py tests/test_application_structure.py tests/test_watermark_v4_api.py
git commit -m "refactor: extract database repositories"
```

### Task 4: Extract Image Processing Modules

**Files:**
- Create: `trace_app/imaging/__init__.py`
- Create: `trace_app/imaging/io.py`
- Create: `trace_app/imaging/fingerprints.py`
- Create: `trace_app/imaging/feature_matching.py`
- Create: `trace_app/imaging/visible_mark.py`
- Modify: `main.py`
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `tests/test_prominent_corner_copyright.py`
- Modify: `tests/test_residual_attribution_gate.py`

- [ ] **Step 1: Add ownership tests for extracted image helpers**

Append to `tests/test_application_structure.py`:

```python
from io import BytesIO

from trace_app.imaging.fingerprints import file_sha256
from trace_app.imaging.io import load_image_from_bytes


def _one_pixel_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_helpers_are_available_without_importing_main() -> None:
    image = load_image_from_bytes(_one_pixel_png())
    assert image.size == (1, 1)
    assert file_sha256(b"abc") == (
        "BA7816BF8F01CFEA414140DE5DAE2223"
        "B00361A396177A9CB410FF61F20015AD"
    ).lower()
```

- [ ] **Step 2: Run the test and verify missing module imports fail**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails for `trace_app.imaging`.

- [ ] **Step 3: Move image helpers by responsibility without changing their bodies**

Move these exact functions from `main.py`:

```text
imaging/io.py:
  load_upload_image, load_image_from_bytes, load_image_from_url, save_thumbnail
imaging/fingerprints.py:
  file_sha256, path_sha256, image_content_sha256, matched_file_fingerprint
imaging/feature_matching.py:
  save_record_feature_index, save_record_feature_index_v4,
  record_feature_index_path, image_to_cv_gray, record_visual_consistency,
  residual_candidate_evidence, detect_by_residual_match, feature_match_score,
  feature_match_homography, align_query_to_record, resize_for_residual,
  robust_residual_score, detect_by_visual_match, is_registered_original_image,
  rank_aligned_candidates
imaging/visible_mark.py:
  detect_visible_copyright, load_font, load_random_font, draw_text_pattern,
  draw_irregular_text_pattern, draw_prominent_corner_label,
  apply_visible_copyright
```

Replace implicit `UPLOAD_DIR` and `DATA_DIR` access with keyword-only `settings: Settings = default_settings` parameters where a path is needed. Replace record reads with a `records` argument or `Repository` argument. Preserve existing defaults at compatibility call sites.

- [ ] **Step 4: Re-export the moved functions temporarily from `main.py`**

```python
from trace_app.imaging.feature_matching import *
from trace_app.imaging.fingerprints import *
from trace_app.imaging.io import *
from trace_app.imaging.visible_mark import *
```

Update monkeypatches in the three affected test files to patch `trace_app.imaging.<owner>.<name>`. Patch `default_settings` or pass a test `Settings` instance instead of assigning `main.UPLOAD_DIR`.

- [ ] **Step 5: Run image and aligned detection regressions**

Run: `pytest tests/test_application_structure.py tests/test_prominent_corner_copyright.py tests/test_residual_attribution_gate.py tests/test_aligned_authenticated_detection.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit image extraction**

```powershell
git add trace_app/imaging main.py tests
git commit -m "refactor: extract image processing modules"
```

### Task 5: Extract Stateless Watermark Families

**Files:**
- Create: `trace_app/watermark/__init__.py`
- Create: `trace_app/watermark/lsb.py`
- Create: `trace_app/watermark/frequency.py`
- Create: `trace_app/watermark/dot_matrix.py`
- Create: `trace_app/watermark/small_crop.py`
- Modify: `main.py`
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `tests/test_false_positive_gate.py`

- [ ] **Step 1: Write import and deterministic carrier tests**

Append to `tests/test_application_structure.py`:

```python
from trace_app.watermark.lsb import bits_from_bytes, bytes_from_bits
from trace_app.watermark.small_crop import small_trace_short_code


def test_extracted_watermark_helpers_keep_deterministic_contracts() -> None:
    assert bytes_from_bits(bits_from_bytes(b"trace")) == b"trace"
    assert small_trace_short_code("TR-EXAMPLE") == small_trace_short_code("TR-EXAMPLE")
```

- [ ] **Step 2: Verify the watermark package test fails**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails because the extracted modules do not exist.

- [ ] **Step 3: Move functions into the four focused modules**

Use these exact ownership groups:

```text
watermark/lsb.py:
  bits_from_bytes through extract_block_lsb
watermark/frequency.py:
  robust_pattern, layer_seed, pseudo_random_signs, apply_dct_layer,
  dct_layer_score, apply_dwt_layer, dwt_layer_score, fft_pattern,
  apply_fft_layer, fft_layer_score, apply_frequency_layers,
  layer_scores_for_image
watermark/dot_matrix.py:
  dot_matrix_bits_from_trace through detect_dot_matrix_trace
watermark/small_crop.py:
  small_trace_short_code through detect_small_crop_trace,
  apply_code_layer through detect_watermark_code
```

Import constants from `trace_app.config`, image-only helpers from `trace_app.imaging`, and pass candidate records explicitly to detection functions that currently call `read_records()`.

- [ ] **Step 4: Re-export the moved public functions from `main.py` and update patch targets**

```python
from trace_app.watermark.dot_matrix import *
from trace_app.watermark.frequency import *
from trace_app.watermark.lsb import *
from trace_app.watermark.small_crop import *
```

Update tests to patch the owning watermark module. Do not keep duplicate function definitions in `main.py`.

- [ ] **Step 5: Run focused legacy watermark tests**

Run: `pytest tests/test_application_structure.py tests/test_false_positive_gate.py tests/test_aligned_authenticated_detection.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit stateless watermark extraction**

```powershell
git add trace_app/watermark main.py tests
git commit -m "refactor: extract legacy watermark algorithms"
```

### Task 6: Extract Robust and Detection Pipelines

**Files:**
- Create: `trace_app/watermark/robust.py`
- Create: `trace_app/watermark/detection.py`
- Modify: `main.py`
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Add a pure detection pipeline order test**

Add to `tests/test_application_structure.py`:

```python
from PIL import Image

from trace_app.watermark.detection import DetectionPipeline


def test_detection_pipeline_returns_first_success() -> None:
    calls: list[str] = []
    pipeline = DetectionPipeline(
        detectors=(
            lambda image: calls.append("first") or None,
            lambda image: calls.append("second") or {"trace_id": "TR-1"},
            lambda image: calls.append("third") or {"trace_id": "TR-2"},
        )
    )
    assert pipeline.detect(Image.new("RGB", (8, 8))) == {"trace_id": "TR-1"}
    assert calls == ["first", "second"]
```

- [ ] **Step 2: Verify the detection pipeline import fails**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails for `trace_app.watermark.detection`.

- [ ] **Step 3: Move robust implementations and make candidates explicit**

Move `robust_code_from_trace` through `detect_robust_watermark`, excluding functions already assigned in Tasks 4 and 5, into `trace_app/watermark/robust.py`. Keep v1, v2, and v3 codec tags and dispatch unchanged. Change `robust_candidate_records` and `legacy_robust_candidate_records` to require `records: list[dict[str, Any]]`. Change `detect_aligned_authenticated_watermark` to require keyword-only `records: list[dict[str, Any]]` and accept optional `generated_trace_ids: list[str] | None = None`. Replace each former `read_records()` call with the supplied `records` value and each former `app.state.generated_trace_ids` read with the supplied list.

- [ ] **Step 4: Implement the ordered detector container and move extraction orchestration**

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

Detector = Callable[[Image.Image], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class DetectionPipeline:
    detectors: tuple[Detector, ...]

    def detect(self, image: Image.Image) -> dict[str, Any] | None:
        for detector in self.detectors:
            result = detector(image)
            if result is not None:
                return result
        return None
```

Move `should_run_frequency_fallbacks`, `should_run_visual_match_fallback`, `v4_candidate_records`, `detect_v4_watermark`, and `extract_watermark_from_image` orchestration into `detection.py`. Preserve the exact current branch order and feature flags; inject repository records and statistics callbacks.

- [ ] **Step 5: Re-export functions and migrate patch targets**

Import the new functions from `robust.py` and `detection.py` in `main.py`. Update pipeline monkeypatches to patch `trace_app.watermark.detection`, which is now the effective lookup site.

- [ ] **Step 6: Run robust, v4, and false-positive regressions**

Run: `pytest tests/test_aligned_authenticated_detection.py tests/test_watermark_v4_api.py tests/test_false_positive_gate.py -q`

Expected: exit code 0 and no change in expected attribution decisions.

- [ ] **Step 7: Commit robust pipeline extraction**

```powershell
git add trace_app/watermark main.py tests
git commit -m "refactor: extract watermark detection pipeline"
```

### Task 7: Extract Authentication Service

**Files:**
- Create: `trace_app/auth/__init__.py`
- Create: `trace_app/auth/schemas.py`
- Create: `trace_app/auth/service.py`
- Modify: `main.py`
- Modify: `tests/test_application_structure.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write failing authentication service tests**

Add to `tests/test_application_structure.py`:

```python
from trace_app.auth.service import AuthService


def test_auth_service_filters_unknown_menu_keys() -> None:
    service = AuthService(repository=None)
    assert service.allowed_menu_keys(["watermark", "unknown", "trace"]) == [
        "watermark",
        "trace",
    ]
```

- [ ] **Step 2: Verify the service import fails**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails for `trace_app.auth.service`.

- [ ] **Step 3: Implement authentication schemas and service**

Move `public_users`, `allowed_menu_keys`, `role_for_username`, login validation, role updates, and user CRUD rules into `AuthService`. Its constructor and pure menu method are:

```python
class AuthService:
    def __init__(self, repository: Repository | None) -> None:
        self.repository = repository

    @staticmethod
    def allowed_menu_keys(menus: Any) -> list[str]:
        if not isinstance(menus, list):
            return []
        return [key for key in menus if key in MENU_LABELS]
```

Define Pydantic request models only for JSON bodies currently typed as `dict[str, Any]`; configure them to accept exactly the existing fields and keep response dictionaries unchanged.

- [ ] **Step 4: Delegate temporary authentication functions in `main.py`**

Create `auth_service = AuthService(repository)` and delegate route business operations to it. Keep route decorators in `main.py` until Task 9.

- [ ] **Step 5: Run user, password, and API tests**

Run: `pytest tests/test_password_security.py tests/test_database_store.py tests/test_watermark_v4_api.py tests/test_application_structure.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit authentication extraction**

```powershell
git add trace_app/auth main.py tests
git commit -m "refactor: extract authentication service"
```

### Task 8: Introduce the Watermark Application Service

**Files:**
- Create: `trace_app/watermark/service.py`
- Modify: `main.py`
- Modify: `tests/test_application_structure.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write a failing service dependency test**

Add to `tests/test_application_structure.py`:

```python
from trace_app.watermark.service import WatermarkService


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )


def test_watermark_service_owns_repository_and_settings(test_settings) -> None:
    repository = object()
    runtime = Runtime()
    service = WatermarkService(
        settings=test_settings,
        repository=repository,
        runtime=runtime,
    )
    assert service.settings is test_settings
    assert service.repository is repository
    assert service.runtime is runtime
```

- [ ] **Step 2: Verify the service import fails**

Run: `pytest tests/test_application_structure.py -v`

Expected: collection fails for `trace_app.watermark.service`.

- [ ] **Step 3: Move API-level embed and extract use cases into `WatermarkService`**

Create `WatermarkService` with a keyword-only constructor accepting `settings: Settings`, `repository: Repository`, and `runtime: Runtime`, storing all three as instance attributes. Move the complete body of `embed_watermark` from `main.py:3573-3765` into an asynchronous `embed` method with the same form inputs as explicit keyword-only parameters. Make these substitutions throughout that body:

```text
UPLOAD_DIR / ORIGINAL_DIR / WATERMARKED_DIR / THUMBNAIL_DIR
  -> self.settings paths
read_records / add_record / record_watermark_generation
  -> self.repository methods
remember_generated_trace(trace_id)
  -> insert into self.runtime.generated_trace_ids and retain only 24 values
```

Move `extract_watermark_from_image` into synchronous `extract_image(image)`, the current upload route body into asynchronous `extract_upload(file)`, and the URL route body into synchronous `extract_url(url)`. Each method calls the extracted algorithm/image modules and repository methods directly and contains no route decorators. Preserve every current branch, response field, and `HTTPException` exactly.

- [ ] **Step 4: Make existing routes thin service calls**

The temporary route body becomes:

```python
@app.post("/api/watermark/extract")
async def extract_watermark(file: UploadFile = File(...)) -> dict[str, Any]:
    return await watermark_service.extract_upload(file)
```

Apply the same delegation to embed and URL extraction.

- [ ] **Step 5: Run API and pipeline tests**

Run: `pytest tests/test_watermark_v4_api.py tests/test_false_positive_gate.py tests/test_aligned_authenticated_detection.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit the watermark service**

```powershell
git add trace_app/watermark/service.py main.py tests
git commit -m "refactor: add watermark application service"
```

### Task 9: Move HTTP Endpoints to FastAPI Routers and Add the Application Factory

**Files:**
- Create: `trace_app/api/__init__.py`
- Create: `trace_app/api/auth.py`
- Create: `trace_app/api/users.py`
- Create: `trace_app/api/watermark.py`
- Create: `trace_app/api/images.py`
- Create: `trace_app/api/dashboard.py`
- Create: `trace_app/dependencies.py`
- Create: `trace_app/application.py`
- Modify: `main.py`
- Modify: `tests/test_application_structure.py`

- [ ] **Step 1: Replace the temporary structure assertion with the final factory contract**

Replace `test_target_entry_point_has_no_business_definitions` and add a factory isolation test:

```python
from trace_app.application import create_app
from trace_app.config import Settings


def test_main_is_a_thin_entry_point() -> None:
    source = Path("main.py").read_text(encoding="utf-8")
    assert "def " not in source
    assert "@app." not in source
    assert len(source.splitlines()) <= 10


def test_application_factory_registers_routes_without_database(tmp_path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )
    app = create_app(settings=settings, initialize_database=False)
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }
    assert EXPECTED_ROUTES <= actual
    assert app.state.runtime.store is None
```

- [ ] **Step 2: Verify the final factory tests fail**

Run: `pytest tests/test_application_structure.py -v`

Expected: import failure for `trace_app.application` or failure because `main.py` still defines routes.

- [ ] **Step 3: Implement dependency accessors**

Create `trace_app/dependencies.py`:

```python
from fastapi import Request


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_watermark_service(request: Request) -> WatermarkService:
    return request.app.state.watermark_service
```

- [ ] **Step 4: Create thin routers with unchanged paths and signatures**

Each router uses `APIRouter` and `Depends`. The watermark extraction pattern is:

```python
router = APIRouter(prefix="/api/watermark", tags=["watermark"])


@router.post("/extract")
async def extract_watermark(
    file: UploadFile = File(...),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    return await service.extract_upload(file)
```

Move every route named in `EXPECTED_ROUTES` to its mapped router. Route functions may parse FastAPI inputs and call one service method; database, filesystem, image, and watermark operations must remain outside router modules.

- [ ] **Step 5: Implement the application factory**

Create `trace_app/application.py`:

```python
import os
import sys

from trace_app.config import Settings, settings as default_settings


def running_pytest() -> bool:
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None


def create_app(
    *,
    settings: Settings = default_settings,
    initialize_database: bool | None = None,
) -> FastAPI:
    ensure_directories(settings)
    enabled = not running_pytest() if initialize_database is None else initialize_database
    runtime = create_runtime(settings, enabled=enabled)
    repository = Repository(runtime.store)

    app = FastAPI(title=settings.app_name)
    app.state.runtime = runtime
    app.state.repository = repository
    app.state.auth_service = AuthService(repository)
    app.state.watermark_service = WatermarkService(
        settings=settings,
        repository=repository,
        runtime=runtime,
    )
    register_static_routes(app, settings)
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(watermark.router)
    app.include_router(images.router)
    app.include_router(dashboard.router)
    return app
```

Move `ensure_dirs`, MIME registration, static mounts, homepage, logo, and favicon handlers into this module as `ensure_directories()` and `register_static_routes()`.

- [ ] **Step 6: Run route and API tests**

Run: `pytest tests/test_application_structure.py tests/test_homepage_data_loading.py tests/test_watermark_v4_api.py -q`

Expected: exit code 0 and exactly one registration for each business route.

- [ ] **Step 7: Commit routers and factory**

```powershell
git add trace_app/api trace_app/dependencies.py trace_app/application.py main.py tests
git commit -m "refactor: add FastAPI application factory and routers"
```

### Task 10: Finalize Compatibility Exports and Thin `main.py`

**Files:**
- Create: `trace_app/compat.py`
- Replace: `main.py`
- Modify: `tests/test_application_structure.py`
- Modify: existing tests that still patch `main`

- [ ] **Step 1: Add a compatibility export identity test**

Append to `tests/test_application_structure.py`:

```python
import trace_app.compat as compat
from trace_app.watermark.robust import embed_robust_watermark


def test_compatibility_exports_delegate_to_owner_modules() -> None:
    assert compat.embed_robust_watermark is embed_robust_watermark
    assert main.embed_robust_watermark is embed_robust_watermark
    assert main.app is not None
```

- [ ] **Step 2: Create compatibility exports without implementations**

`trace_app/compat.py` imports and re-exports the public constants and functions used by `rg -o "main\.[A-Za-z_][A-Za-z0-9_]*" tests tools *.py`. Its structure is:

```python
from trace_app.config import *
from trace_app.database.repositories import Repository
from trace_app.imaging.feature_matching import *
from trace_app.imaging.fingerprints import *
from trace_app.imaging.io import *
from trace_app.imaging.visible_mark import *
from trace_app.watermark.detection import *
from trace_app.watermark.dot_matrix import *
from trace_app.watermark.frequency import *
from trace_app.watermark.lsb import *
from trace_app.watermark.robust import *
from trace_app.watermark.small_crop import *
```

Do not define wrappers containing business logic. Update remaining test monkeypatches from `main.<name>` to the module where `<name>` is looked up at runtime.

- [ ] **Step 3: Replace `main.py` with the thin entry point**

```python
from trace_app.application import create_app
from trace_app.compat import *  # noqa: F401,F403

app = create_app()
```

- [ ] **Step 4: Verify no business code remains in `main.py`**

Run: `pytest tests/test_application_structure.py -v`

Expected: all tests pass, including the line-count and no-definition assertions.

Run: `rg -n "^(def |async def |class |@app\.)" main.py`

Expected: no output and exit code 1.

- [ ] **Step 5: Run all focused refactor regressions**

Run: `pytest tests/test_database_store.py tests/test_password_security.py tests/test_prominent_corner_copyright.py tests/test_residual_attribution_gate.py tests/test_aligned_authenticated_detection.py tests/test_false_positive_gate.py tests/test_watermark_v4_api.py -q`

Expected: exit code 0.

- [ ] **Step 6: Commit the thin entry point**

```powershell
git add main.py trace_app/compat.py tests
git commit -m "refactor: reduce main to FastAPI entry point"
```

### Task 11: Full Verification and Deployment Contract

**Files:**
- Modify only files required to correct verified regressions.

- [ ] **Step 1: Run static import and syntax checks**

Run: `python -m compileall -q main.py trace_app watermark_v4`

Expected: exit code 0.

Run: `python -c "import main; print(main.app.title); print(len(main.app.routes))"`

Expected: prints the configured application title and a nonzero route count without a traceback. Run with test-safe environment variables if production database credentials are not present.

- [ ] **Step 2: Run the complete automated test suite**

Run: `pytest -q`

Expected: exit code 0. Any pre-existing baseline failure recorded in Task 1 must remain unchanged and be explicitly reported; all new failures must be fixed before completion.

- [ ] **Step 3: Run fast watermark quality gates**

Run: `pytest tests/test_watermark_v4_quick_matrix.py tests/test_false_positive_gate.py -q`

Expected: exit code 0 with the same positive and negative decisions as baseline.

- [ ] **Step 4: Verify deployment references still use the stable entry point**

Run: `rg -n "uvicorn|main:app" deploy.sh deploy.ps1 README_DEPLOY.md release tests/test_centos_deploy_contract.py`

Expected: deployment continues to reference `main:app`; no deployment file needs a new module path.

- [ ] **Step 5: Check the final diff for accidental duplication and generated files**

Run: `git diff --check`

Expected: no output.

Run: `rg -n "^(def |async def )" main.py`

Expected: no output.

Run: `git status --short`

Expected: only intentional source and test changes are listed.

- [ ] **Step 6: Commit verification fixes if any were required**

```powershell
git add main.py trace_app tests
git commit -m "test: verify modular FastAPI application"
```

Skip this commit when Step 1 through Step 5 require no file changes.
