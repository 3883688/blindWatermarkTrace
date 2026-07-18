# V4 JPEG Watermark Output Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save JPEG-source V4 watermarks as quality 90-95 JPEG files with the source JPEG's chroma sampling, targeting at most 1.25 times the uploaded size when possible, while preserving existing lossless behavior for every other path.

**Architecture:** A focused imaging output module owns source-sampling normalization, JPEG encoding, adaptive quality selection, final file naming, and persisted-image reload. `WatermarkService` decides whether the current request is the eligible JPEG+V4 case, captures the source sampling before transforms, then uses the persisted result for thumbnails, feature indexes, hashes, and record URLs. Compatibility operation builders inject the helpers in the same way as existing image I/O functions.

**Tech Stack:** Python 3, Pillow, FastAPI `UploadFile`, pytest, existing V4 watermark detector and repository contracts.

---

## File Map

- Create `trace_app/imaging/output.py`: normalize source JPEG sampling, encode quality 90-95 candidates with that sampling, select the highest fitting quality, save `.jpg` or `.png`, and return the persisted image.
- Create `tests/test_watermarked_output.py`: unit coverage for sampling normalization, quality selection, quality floor, real JPEG output, and PNG fallback.
- Modify `trace_app/watermark/service.py`: select JPEG only for true JPEG input with V4, capture source sampling and uploaded byte size, and derive downstream artifacts from the persisted output.
- Modify `trace_app/watermark/default_operations.py`: inject the output and source-sampling helpers into the default service operations.
- Modify `trace_app/compat.py`: inject private helpers into the compatibility service factory without extending the strict legacy `__all__` contract.
- Modify `tests/test_watermark_v4_api.py`: verify source-sampling preservation, JPEG+V4 routing, size behavior, fingerprints, lossy-resave V4 detection, PNG V4 behavior, and legacy lossless behavior.

### Task 1: Adaptive JPEG Output Unit

**Files:**
- Create: `trace_app/imaging/output.py`
- Create: `tests/test_watermarked_output.py`

- [ ] **Step 1: Write failing adaptive-quality tests**

Create `tests/test_watermarked_output.py` with deterministic selection tests and real Pillow encoding coverage:

```python
from pathlib import Path

from PIL import Image

from trace_app.imaging.output import (
    JPEG_MIN_QUALITY,
    WatermarkedOutput,
    encode_adaptive_jpeg,
    encode_jpeg,
    save_watermarked_output,
)


def _image() -> Image.Image:
    return Image.effect_noise((320, 240), 32).convert("RGB")


def test_adaptive_jpeg_selects_highest_quality_within_target() -> None:
    def sized_encoder(image: Image.Image, quality: int) -> bytes:
        return bytes(quality * 10)

    content, quality = encode_adaptive_jpeg(
        _image(),
        source_size=744,
        encoder=sized_encoder,
    )

    assert quality == 93
    assert len(content) == 930


def test_adaptive_jpeg_never_drops_below_quality_90() -> None:
    def sized_encoder(image: Image.Image, quality: int) -> bytes:
        return bytes(quality * 10)

    content, quality = encode_adaptive_jpeg(
        _image(),
        source_size=1,
        encoder=sized_encoder,
    )

    assert quality == JPEG_MIN_QUALITY == 90
    assert len(content) == 900


def test_encode_jpeg_produces_real_jpeg() -> None:
    content = encode_jpeg(_image(), 92)

    assert content.startswith(b"\xff\xd8")


def test_save_watermarked_output_returns_persisted_jpeg(tmp_path: Path) -> None:
    result = save_watermarked_output(
        _image(),
        tmp_path / "watermarked",
        jpeg_output=True,
        source_size=1,
    )

    assert isinstance(result, WatermarkedOutput)
    assert result.path.suffix == ".jpg"
    assert result.quality == 90
    assert result.path.exists()
    with Image.open(result.path) as loaded:
        assert loaded.format == "JPEG"
    assert result.image.mode == "RGB"


def test_save_watermarked_output_keeps_png_path_lossless(tmp_path: Path) -> None:
    source = _image()

    result = save_watermarked_output(
        source,
        tmp_path / "watermarked",
        jpeg_output=False,
        source_size=1,
    )

    assert result.path.suffix == ".png"
    assert result.quality is None
    with Image.open(result.path) as loaded:
        assert loaded.format == "PNG"
        assert loaded.convert("RGB").tobytes() == source.tobytes()
```

- [ ] **Step 2: Run the unit tests and verify RED**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trace_app.imaging.output'`.

- [ ] **Step 3: Implement the focused output module**

Create `trace_app/imaging/output.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image


JPEG_MIN_QUALITY = 90
JPEG_MAX_QUALITY = 95
JPEG_TARGET_RATIO = 1.25
JpegEncoder = Callable[[Image.Image, int], bytes]


@dataclass(frozen=True, slots=True)
class WatermarkedOutput:
    path: Path
    image: Image.Image
    quality: int | None


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=0,
    )
    return buffer.getvalue()


def encode_adaptive_jpeg(
    image: Image.Image,
    source_size: int,
    *,
    encoder: JpegEncoder = encode_jpeg,
) -> tuple[bytes, int]:
    target_size = max(1, int(source_size * JPEG_TARGET_RATIO))
    minimum_content: bytes | None = None
    for quality in range(JPEG_MAX_QUALITY, JPEG_MIN_QUALITY - 1, -1):
        content = encoder(image, quality)
        if quality == JPEG_MIN_QUALITY:
            minimum_content = content
        if len(content) <= target_size:
            return content, quality
    if minimum_content is None:
        raise RuntimeError("JPEG quality range is empty")
    return minimum_content, JPEG_MIN_QUALITY


def save_watermarked_output(
    image: Image.Image,
    output_base: Path,
    *,
    jpeg_output: bool,
    source_size: int,
) -> WatermarkedOutput:
    quality = None
    if jpeg_output:
        path = output_base.with_suffix(".jpg")
        content, quality = encode_adaptive_jpeg(image, source_size)
        path.write_bytes(content)
    else:
        path = output_base.with_suffix(".png")
        image.save(path, format="PNG")
    with Image.open(path) as loaded:
        loaded.load()
        persisted = loaded.copy()
    return WatermarkedOutput(path=path, image=persisted, quality=quality)
```

- [ ] **Step 4: Run the unit tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit the output unit**

```powershell
git add -- trace_app/imaging/output.py tests/test_watermarked_output.py
git commit -m "feat: add adaptive JPEG watermark output"
```

### Task 2: Service Routing and Persisted-Image Consistency

**Files:**
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `trace_app/watermark/service.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/compat.py`

- [ ] **Step 1: Add JPEG+V4 and unchanged-path API tests before service code**

Add these helpers and test to `tests/test_watermark_v4_api.py`:

```python
def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    _feature_image((512, 384), seed=808).save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
    )
    return buffer.getvalue()


def _embed_v4_jpeg(client: TestClient):
    return client.post(
        "/api/watermark/embed",
        files={"file": ("v4-source.jpg", _jpeg_bytes(), "image/jpeg")},
        data={
            "user_id": "pytest-v4-jpeg",
            "robust_watermark_version": "4",
            "copyright_enabled": "false",
            "small_crop_trace_enabled": "true",
            "dot_matrix_trace_enabled": "true",
        },
    )


def test_v4_jpeg_generation_preserves_jpeg_output_contract() -> None:
    response = _embed_v4_jpeg(TestClient(main.app))

    assert response.status_code == 200, response.text
    record = response.json()
    assert record["download_url"].endswith("-watermarked.jpg")
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    with Image.open(output_path) as loaded:
        assert loaded.format == "JPEG"
        assert record["watermarked_image_sha256"] == main.image_content_sha256(
            loaded
        )
    assert output_path.stat().st_size <= int(len(_jpeg_bytes()) * 1.25)


def test_v4_jpeg_generation_uses_decoded_format_not_upload_metadata() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/api/watermark/embed",
        files={"file": ("misleading.png", _jpeg_bytes(), "image/png")},
        data={
            "user_id": "pytest-v4-jpeg-metadata",
            "robust_watermark_version": "4",
            "copyright_enabled": "false",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["download_url"].endswith("-watermarked.jpg")


def test_v4_jpeg_output_survives_lossy_resave_detection() -> None:
    client = TestClient(main.app)
    record = _embed_v4_jpeg(client).json()
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    transformed = BytesIO()
    with Image.open(output_path) as loaded:
        loaded.convert("RGB").save(
            transformed,
            format="JPEG",
            quality=90,
            subsampling=0,
        )

    response = _extract_bytes(
        client,
        transformed.getvalue(),
        "resaved-v4.jpg",
    )

    assert response.status_code == 200, response.text
    assert response.json()["trace_id"] == record["trace_id"]
    assert response.json()["code_recovery"]["codec"] == V4Config().codec


def test_v4_png_generation_remains_png() -> None:
    record = _embed_v4(TestClient(main.app)).json()
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )

    assert record["download_url"].endswith("-watermarked.png")
    with Image.open(output_path) as loaded:
        assert loaded.format == "PNG"


def test_legacy_jpeg_generation_remains_lossless_png_with_lsb() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/api/watermark/embed",
        files={"file": ("legacy-source.jpg", _jpeg_bytes(), "image/jpeg")},
        data={
            "user_id": "pytest-legacy-jpeg",
            "robust_watermark_version": "1",
            "copyright_enabled": "false",
            "small_crop_trace_enabled": "false",
            "dot_matrix_trace_enabled": "false",
        },
    )

    assert response.status_code == 200, response.text
    record = response.json()
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    assert output_path.suffix == ".png"
    with Image.open(output_path) as loaded:
        assert loaded.format == "PNG"
        payload = main.extract_full_lsb(loaded)
    assert payload is not None
    assert payload["trace_id"] == record["trace_id"]


def test_v4_jpeg_exact_file_fingerprint_still_matches() -> None:
    client = TestClient(main.app)
    record = _embed_v4_jpeg(client).json()
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )

    response = _extract_bytes(client, output_path.read_bytes(), "exact.jpg")

    assert response.status_code == 200, response.text
    assert response.json()["trace_id"] == record["trace_id"]
    assert response.json()["matched_file_type"] == "watermarked"
    assert response.json()["matched_hash_type"] == "file_md5_sha256"
```

- [ ] **Step 2: Run the API test and verify RED**

Run:

```powershell
python -m pytest tests/test_watermark_v4_api.py -k "jpeg or remains_png" -q
```

Expected: the JPEG routing tests FAIL because `download_url` ends in `-watermarked.png`; unchanged PNG and legacy assertions pass.

- [ ] **Step 3: Add the output operation contract and inject it**

In `trace_app/watermark/service.py`, import `WatermarkedOutput` and add the operation beside `save_thumbnail`:

```python
from trace_app.imaging.output import WatermarkedOutput


save_watermarked_output: Callable[..., WatermarkedOutput]
save_thumbnail: Callable[[Image.Image, Path], None]
```

In `trace_app/watermark/default_operations.py`, import the module and inject the function:

```python
from trace_app.imaging import feature_matching, fingerprints, io, output, visible_mark

save_watermarked_output=output.save_watermarked_output,
save_thumbnail=io.save_thumbnail,
```

In `trace_app/compat.py`, add the compatibility import and wrapper:

```python
from trace_app.imaging import output as imaging_output


def _save_watermarked_output(
    image: Image.Image,
    output_base: Path,
    *,
    jpeg_output: bool,
    source_size: int,
) -> imaging_output.WatermarkedOutput:
    return imaging_output.save_watermarked_output(
        image,
        output_base,
        jpeg_output=jpeg_output,
        source_size=source_size,
    )
```

Inject it in `get_watermark_service()`:

```python
save_watermarked_output=_save_watermarked_output,
save_thumbnail=save_thumbnail,
```

- [ ] **Step 4: Route eligible requests through adaptive output**

In `WatermarkService.embed`, replace the fixed output path setup and capture the real source properties before image transforms:

```python
output_base = self.settings.watermarked_dir / f"{image_id}-watermarked"
thumbnail_path = self.settings.thumbnail_dir / f"{image_id}-thumb.png"

uploaded_size = getattr(file, "size", None)
image = await op.load_upload_image(file)
source_format = str(image.format or "").upper()
image.save(original_path)
source_size = (
    int(uploaded_size)
    if isinstance(uploaded_size, int) and uploaded_size > 0
    else original_path.stat().st_size
)
```

Replace the fixed PNG save and switch downstream work to the persisted result:

```python
saved_output = op.save_watermarked_output(
    watermarked,
    output_base,
    jpeg_output=(
        source_format == "JPEG"
        and robust_version == op.robust_version_v4
    ),
    source_size=source_size,
)
output_path = saved_output.path
persisted_watermarked = saved_output.image
op.save_thumbnail(persisted_watermarked, thumbnail_path)
feature_index_path = (
    op.save_record_feature_index_v4(persisted_watermarked, image_id)
    if robust_version == op.robust_version_v4
    else op.save_record_feature_index(persisted_watermarked, image_id)
)
```

Use the persisted image for the remaining pixel-dependent fields:

```python
watermarked_image_sha256 = op.image_content_sha256(persisted_watermarked)

# In the record's legacy layer score branch:
else op.layer_scores_for_image(persisted_watermarked, trace_id)
```

- [ ] **Step 5: Run focused service and API tests**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py tests/test_watermark_v4_api.py -k "jpeg or remains_png" -q
python -m pytest tests/test_application_structure.py::test_watermark_service_factory_synchronizes_current_generated_trace_list tests/test_application_structure.py::test_compat_all_matches_legacy_public_api_and_import_star -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit service integration**

```powershell
git add -- trace_app/watermark/service.py trace_app/watermark/default_operations.py trace_app/compat.py tests/test_watermark_v4_api.py
git commit -m "feat: save V4 JPEG watermarks near source size"
```

### Task 3: Detection and Legacy Regression Verification

**Files:**
- Verify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Run the complete V4 API module**

Run:

```powershell
python -m pytest tests/test_watermark_v4_api.py -q
```

Expected: the complete module passes with no failures or errors. The already-known Starlette and Pillow deprecation warnings are allowed.

- [ ] **Step 2: Confirm persisted JPEG detection evidence**

Run:

```powershell
python -m pytest tests/test_watermark_v4_api.py::test_v4_jpeg_reencode_is_detected_by_codec tests/test_watermark_v4_api.py::test_v4_jpeg_exact_reupload_matches_watermarked_fingerprint -q
```

Expected: both the decoded V4 path and the exact file-fingerprint path pass.

### Task 4: Full Verification

**Files:**
- Verify only; modify an implementation or test file only if a failing command exposes a defect in this feature.

- [ ] **Step 1: Run imaging, service, fingerprint, and V4 tests**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py tests/test_watermark_v4_api.py tests/test_application_structure.py tests/test_aligned_authenticated_detection.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the backend test suite**

Run:

```powershell
python -m pytest tests -q
```

Expected: the command currently stops on the repository's known missing untracked `img/` fixtures. Record that baseline failure, then use Task 6's explicit module selection for the broadest runnable suite; do not fabricate fixtures or call the collection failure green.

- [ ] **Step 3: Verify source formatting and scope**

Run:

```powershell
git diff --check
git status --short
git diff --stat HEAD~3..HEAD
```

Expected: no whitespace errors; only the planned source and test files plus the user's pre-existing unrelated changes appear; the feature commits do not include release archives or deployment work.

- [ ] **Step 4: Run a real-file size probe without modifying tracked files**

Run:

```powershell
@'
from io import BytesIO
from pathlib import Path
from PIL import Image
from trace_app.imaging.output import encode_adaptive_jpeg

source = Path("3.jpg")
with Image.open(source) as image:
    content, quality = encode_adaptive_jpeg(image.convert("RGB"), source.stat().st_size)
print({
    "source_bytes": source.stat().st_size,
    "output_bytes": len(content),
    "ratio": round(len(content) / source.stat().st_size, 3),
    "quality": quality,
})
'@ | python -
```

Expected: `quality` is between 90 and 95; `ratio` is at most 1.25 when a quality in that range fits, otherwise `quality` is exactly 90.

- [ ] **Step 5: Record verification state**

Run:

```powershell
git log -4 --oneline
```

Expected: the design commit and the focused implementation/test commits are present; no additional commit is needed when verification makes no changes.

### Task 5: Preserve Source JPEG Chroma Sampling

**Files:**
- Modify: `trace_app/imaging/output.py`
- Modify: `tests/test_watermarked_output.py`
- Modify: `trace_app/watermark/service.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/compat.py`
- Modify: `tests/test_watermark_v4_api.py`

- [ ] **Step 1: Write failing source-sampling unit tests**

Extend `tests/test_watermarked_output.py` so the real encoder accepts a sampling value, the source sampling reader normalizes supported values, and adaptive persistence keeps that value:

```python
import pytest

from trace_app.imaging.output import jpeg_subsampling


def _jpeg_source(subsampling: int) -> Image.Image:
    buffer = BytesIO()
    _image().save(
        buffer,
        format="JPEG",
        quality=90,
        subsampling=subsampling,
    )
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    return image


@pytest.mark.parametrize("subsampling", (0, 1, 2))
def test_jpeg_subsampling_preserves_supported_source_value(
    subsampling: int,
) -> None:
    source = _jpeg_source(subsampling)

    assert jpeg_subsampling(source) == subsampling


def test_jpeg_subsampling_uses_444_for_unknown_source() -> None:
    assert jpeg_subsampling(Image.new("RGB", (32, 32))) == 0
```

Update the real JPEG test to call `encode_jpeg(source, 92, subsampling=2)` and expect `"subsampling": 2` plus `JpegImagePlugin.get_sampling(loaded) == 2`.

Add a persistence test:

```python
def test_save_watermarked_output_keeps_requested_jpeg_subsampling(
    tmp_path: Path,
) -> None:
    result = save_watermarked_output(
        _image(),
        tmp_path / "watermarked",
        jpeg_output=True,
        source_size=1,
        jpeg_subsampling=2,
    )

    with Image.open(result.path) as loaded:
        assert JpegImagePlugin.get_sampling(loaded) == 2
```

- [ ] **Step 2: Write the failing API routing test**

Make `_jpeg_bytes` in `tests/test_watermark_v4_api.py` accept a `subsampling: int = 0` parameter and add:

```python
@pytest.mark.parametrize("subsampling", (0, 1, 2))
def test_v4_jpeg_output_preserves_source_subsampling(
    subsampling: int,
) -> None:
    client = TestClient(main.app)
    content = _jpeg_bytes(subsampling=subsampling)
    response = client.post(
        "/api/watermark/embed",
        files={"file": ("sampling.jpg", content, "image/jpeg")},
        data={
            "user_id": f"pytest-sampling-{subsampling}",
            "robust_watermark_version": "4",
            "copyright_enabled": "false",
        },
    )

    assert response.status_code == 200, response.text
    record = response.json()
    output_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    with Image.open(output_path) as loaded:
        assert JpegImagePlugin.get_sampling(loaded) == subsampling
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py tests/test_watermark_v4_api.py -k "subsampling or encode_jpeg" -q
```

Expected: tests fail because `jpeg_subsampling` and the sampling keyword do not exist, while the API still forces sampling `0`.

- [ ] **Step 4: Implement sampling-aware encoding**

Update `trace_app/imaging/output.py`:

```python
from PIL import Image, JpegImagePlugin


JPEG_SUBSAMPLING_VALUES = (0, 1, 2)


def jpeg_subsampling(image: Image.Image) -> int:
    sampling = JpegImagePlugin.get_sampling(image)
    return sampling if sampling in JPEG_SUBSAMPLING_VALUES else 0


def encode_jpeg(
    image: Image.Image,
    quality: int,
    *,
    subsampling: int = 0,
) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=subsampling,
    )
    return buffer.getvalue()
```

Keep deterministic two-argument test encoders supported while routing real encoding through the requested sampling:

```python
def encode_adaptive_jpeg(
    image: Image.Image,
    source_size: int,
    *,
    subsampling: int = 0,
    encoder: JpegEncoder | None = None,
) -> tuple[bytes, int]:
    encode = encoder or (
        lambda current, quality: encode_jpeg(
            current,
            quality,
            subsampling=subsampling,
        )
    )
    target_size = max(1, int(source_size * JPEG_TARGET_RATIO))
    minimum_content: bytes | None = None
    for quality in range(JPEG_MAX_QUALITY, JPEG_MIN_QUALITY - 1, -1):
        content = encode(image, quality)
        if quality == JPEG_MIN_QUALITY:
            minimum_content = content
        if len(content) <= target_size:
            return content, quality
    if minimum_content is None:
        raise RuntimeError("JPEG quality range is empty")
    return minimum_content, JPEG_MIN_QUALITY
```

Add `jpeg_subsampling: int = 0` to `save_watermarked_output` and pass it as `subsampling=jpeg_subsampling` to `encode_adaptive_jpeg`.

- [ ] **Step 5: Pass source sampling through service operation boundaries**

In `trace_app/watermark/service.py`, add this injected operation before the saver:

```python
jpeg_subsampling: Callable[[Image.Image], int]
save_watermarked_output: Callable[..., WatermarkedOutput]
```

Capture the value immediately after the source format and before `image.save`:

```python
source_format = str(image.format or "").upper()
source_subsampling = (
    op.jpeg_subsampling(image) if source_format == "JPEG" else 0
)
```

Pass it to persistence:

```python
saved_output = op.save_watermarked_output(
    watermarked,
    output_base,
    jpeg_output=(
        source_format == "JPEG"
        and robust_version == op.robust_version_v4
    ),
    source_size=source_size,
    jpeg_subsampling=source_subsampling,
)
```

In `trace_app/watermark/default_operations.py`, inject:

```python
jpeg_subsampling=output.jpeg_subsampling,
save_watermarked_output=output.save_watermarked_output,
```

In `trace_app/compat.py`, add a private delegator and inject it without changing `__all__`:

```python
def _jpeg_subsampling(image: Image.Image) -> int:
    return imaging_output.jpeg_subsampling(image)


jpeg_subsampling=_jpeg_subsampling,
save_watermarked_output=_save_watermarked_output,
```

Extend `_save_watermarked_output` with `jpeg_subsampling: int = 0` and pass the keyword to the focused output helper.

- [ ] **Step 6: Run focused and compatibility tests**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py tests/test_watermark_v4_api.py -q
python -m pytest tests/test_application_structure.py::test_watermark_service_factory_synchronizes_current_generated_trace_list tests/test_application_structure.py::test_compat_all_matches_legacy_public_api_and_import_star -q
```

Expected: all tests pass; the output helper tests include all three supported sampling values and the API tests prove persisted JPEG sampling matches the decoded source.

- [ ] **Step 7: Commit the sampling change**

```powershell
git add -- trace_app/imaging/output.py tests/test_watermarked_output.py trace_app/watermark/service.py trace_app/watermark/default_operations.py trace_app/compat.py tests/test_watermark_v4_api.py
git commit -m "feat: preserve source JPEG chroma sampling"
```

### Task 6: Sampling-Aware Final Verification

**Files:**
- Verify only; do not modify tracked files.

- [ ] **Step 1: Run related backend and frontend tests**

Run:

```powershell
python -m pytest tests/test_watermarked_output.py tests/test_watermark_v4_api.py tests/test_application_structure.py tests/test_aligned_authenticated_detection.py -q
Set-Location frontend
npm test -- --run
Set-Location ..
```

Expected: all related backend and all frontend tests pass; existing deprecation warnings are recorded separately from failures.

- [ ] **Step 2: Run the broadest backend suite without missing-image modules**

Use PowerShell to pass every tracked top-level `tests/test_*.py` file except `test_watermark_v4_sync.py`, `test_watermark_v4_dct.py`, `test_watermark_v4_detector.py`, and the newly confirmed `test_false_positive_gate.py` missing-image dependency.

Expected: report every failure. Known unchanged baseline failures from stale legacy inline-frontend contracts and stale release parity are not called green. Any failure in the new output, V4 API, application structure, or aligned detection modules blocks completion.

- [ ] **Step 3: Run the real V4 `3.jpg` sampling probe**

Embed the V4 pilot and authenticated codeword into `3.jpg`, read its source sampling with `jpeg_subsampling`, and call:

```python
content, quality = encode_adaptive_jpeg(
    marked,
    source.stat().st_size,
    subsampling=jpeg_subsampling(source_image),
)
```

Decode the result and verify JPEG/RGB, source sampling `2`, output sampling `2`, quality 90-95, and either ratio no more than 1.25 or quality exactly 90. Force authenticated V4 decoding from the persisted pixels and repeat after one more quality-90/sampling-2 JPEG encoding.

Expected for the tracked sample: approximately 2.15 times the source size at quality 90, at least 12% smaller than the prior 4:4:4 output, with authenticated V4 detection succeeding in both cases.

- [ ] **Step 4: Verify final repository scope**

Run:

```powershell
git diff --check 88cc3cb..HEAD
git status --short
git diff --stat 88cc3cb..HEAD
git log -10 --oneline
```

Expected: clean diff/status; only design, plan, focused imaging/service/compat code, and tests are included. No release archives, deployment changes, frontend build output, secrets, or user dirty files are present.
