# MD5 Fingerprint Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tracing compare MD5 first, confirm candidates with SHA-256, and immediately return success for exact original or watermarked files without decoding the image.

**Architecture:** Extend the focused fingerprint module with MD5 helpers and a two-stage exact-file matcher while preserving the existing image-pixel fallback. Inject path MD5 through `WatermarkOperations`, persist both MD5 values during generation, and let `WatermarkService.extract_upload` treat either exact file type as a successful detection. Keep SHA-256-only records compatible and regenerate the deterministic CentOS release from the updated source tree.

**Tech Stack:** Python 3.11+, FastAPI, Pillow, pytest, hashlib, SQLAlchemy-backed repository payloads

---

## File Structure

- Modify: `trace_app/imaging/fingerprints.py` - calculate MD5 and implement MD5 candidate plus SHA-256 confirmation.
- Modify: `trace_app/watermark/service.py` - inject path MD5, persist generated hashes, and accept original-file exact matches.
- Modify: `trace_app/watermark/default_operations.py` - wire the production path MD5 implementation.
- Modify: `trace_app/compat.py` - preserve patchable `main` compatibility wrappers and public exports.
- Modify: `tests/test_application_structure.py` - cover helper vectors, service statistics, and public API exports.
- Modify: `tests/test_aligned_authenticated_detection.py` - cover confirmed MD5 matches, collisions, and legacy SHA-256 records.
- Modify: `tests/test_watermark_v4_api.py` - prove both generated file types return HTTP 200 before decoding.
- Regenerate: `release/trace-v4-centos-20260715/trace_app/` - synchronize deployable Python sources.
- Regenerate: `release/trace-v4-centos-20260715.zip` - rebuild deterministic CentOS archive.
- Regenerate: `release/trace-v4-centos-20260715.zip.sha256` - record the rebuilt archive checksum.

### Task 1: MD5 Helpers and Confirmed Exact-File Matching

**Files:**
- Modify: `tests/test_application_structure.py`
- Modify: `tests/test_aligned_authenticated_detection.py`
- Modify: `trace_app/imaging/fingerprints.py`

- [ ] **Step 1: Write failing MD5 helper tests**

Add `file_md5` and `path_md5` to the fingerprint imports in `tests/test_application_structure.py`, then add:

```python
def test_imaging_fingerprints_hashes_file_bytes_with_md5(tmp_path: Path) -> None:
    path = tmp_path / "vector.bin"
    path.write_bytes(b"abc")

    assert file_md5(b"abc") == "900150983CD24FB0D6963F7D28E17F72"
    assert path_md5(path) == "900150983CD24FB0D6963F7D28E17F72"
```

- [ ] **Step 2: Run the helper test and verify it fails**

Run: `pytest tests/test_application_structure.py::test_imaging_fingerprints_hashes_file_bytes_with_md5 -v`

Expected: FAIL during collection because `file_md5` and `path_md5` are not defined.

- [ ] **Step 3: Implement the MD5 helpers**

Add alongside the SHA-256 helpers in `trace_app/imaging/fingerprints.py`:

```python
def file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest().upper()


def path_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()
```

- [ ] **Step 4: Run the helper test and verify it passes**

Run: `pytest tests/test_application_structure.py::test_imaging_fingerprints_hashes_file_bytes_with_md5 -v`

Expected: PASS.

- [ ] **Step 5: Write failing matcher tests for confirmation, collisions, and legacy data**

In `tests/test_aligned_authenticated_detection.py`, replace the old stored-hash expectation with focused tests using the public compatibility wrapper:

```python
@pytest.mark.parametrize("file_type", ("original", "watermarked"))
def test_fingerprint_check_confirms_md5_with_sha256_without_image_decode(
    monkeypatch: pytest.MonkeyPatch,
    file_type: str,
) -> None:
    content = f"stored-{file_type}-file".encode("ascii")
    record = {
        "id": f"{file_type}-record",
        "trace_id": f"TR-{file_type.upper()}",
        f"{file_type}_file_md5": main.file_md5(content),
        f"{file_type}_file_sha256": main.file_sha256(content),
        "original_url": "/uploads/originals/source.png",
        "download_url": "/uploads/watermarked/marked.png",
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(
        main,
        "load_image_from_bytes",
        lambda value: (_ for _ in ()).throw(AssertionError("unexpected decode")),
    )

    result = main.matched_file_fingerprint(content)

    assert result is not None
    assert result["trace_id"] == record["trace_id"]
    assert result["matched_file_type"] == file_type
    assert result["matched_hash_type"] == "file_md5_sha256"
    assert result["file_md5"] == main.file_md5(content)
    assert result["file_hash"] == main.file_sha256(content)


def test_fingerprint_check_rejects_md5_candidate_when_sha256_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"collision-candidate"
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [{
            "trace_id": "TR-COLLISION",
            "watermarked_file_md5": main.file_md5(content),
            "watermarked_file_sha256": "0" * 64,
        }],
    )
    monkeypatch.setattr(
        main,
        "load_image_from_bytes",
        lambda value: (_ for _ in ()).throw(ValueError("not an image")),
    )

    assert main.matched_file_fingerprint(content) is None


def test_fingerprint_check_supports_legacy_sha256_only_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"legacy-watermarked-file"
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [{
            "trace_id": "TR-LEGACY-SHA256",
            "watermarked_file_sha256": main.file_sha256(content),
        }],
    )

    result = main.matched_file_fingerprint(content)

    assert result is not None
    assert result["matched_hash_type"] == "file_sha256"
    assert result["file_md5"] == main.file_md5(content)
```

- [ ] **Step 6: Run the matcher tests and verify they fail**

Run: `pytest tests/test_aligned_authenticated_detection.py -k "fingerprint_check" -v`

Expected: FAIL because MD5 injection/output is absent and the legacy type is still `file_bytes`.

- [ ] **Step 7: Implement MD5-first matching with SHA-256 confirmation**

Extend the `matched_file_fingerprint` keyword-only parameters in `trace_app/imaging/fingerprints.py` with:

```python
    file_md5_fn: Callable[[bytes], str] | None = None,
```

Calculate both digests once before record iteration, and use this exact-file branch before returning the existing result:

```python
    hash_md5 = file_md5_fn or file_md5
    hash_file = file_sha256_fn or file_sha256
    md5_digest = hash_md5(content)
    sha256_digest = hash_file(content)
    query_image_digest = None
    for record in read_records():
        for file_type in ("original", "watermarked"):
            stored_md5 = str(record.get(f"{file_type}_file_md5") or "").upper()
            stored_sha256 = str(
                record.get(f"{file_type}_file_sha256") or ""
            ).upper()
            matched_hash_type = None
            matched_hash = None
            if (
                stored_md5 == md5_digest
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                matched_hash_type = "file_md5_sha256"
                matched_hash = sha256_digest
            elif (
                not stored_md5
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                matched_hash_type = "file_sha256"
                matched_hash = sha256_digest
            if matched_hash_type is None:
                stored_image_digest = str(
                    record.get(f"{file_type}_image_sha256") or ""
                ).upper()
                if not stored_image_digest:
                    continue
                try:
                    if query_image_digest is None:
                        query_image_digest = hash_image(load_image(content))
                except Exception:
                    return None
                if stored_image_digest != query_image_digest:
                    continue
                matched_hash_type = "image_pixels"
                matched_hash = query_image_digest
```

Use `sha256_digest` for the existing `file_hash` response field and add `"file_md5": md5_digest` to the result. Preserve evidence fields, URLs, pixel matching, and all other response fields unchanged.

- [ ] **Step 8: Run focused fingerprint tests**

Run: `pytest tests/test_application_structure.py::test_imaging_fingerprints_hashes_file_bytes_with_md5 tests/test_aligned_authenticated_detection.py -k "fingerprint" -v`

Expected: all selected tests PASS.

- [ ] **Step 9: Commit the focused fingerprint implementation**

```powershell
git add -- trace_app/imaging/fingerprints.py tests/test_application_structure.py tests/test_aligned_authenticated_detection.py
git commit -m "feat: add confirmed MD5 fingerprint matching"
```

### Task 2: Generation Persistence and Compatibility Wiring

**Files:**
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `tests/test_application_structure.py`
- Modify: `trace_app/watermark/service.py`
- Modify: `trace_app/watermark/default_operations.py`
- Modify: `trace_app/compat.py`

- [ ] **Step 1: Write failing persistence and public API assertions**

In `test_v4_generation_persists_strict_codec_tag_and_feature_index` in `tests/test_watermark_v4_api.py`, add:

```python
    original_path = main.UPLOAD_DIR / record["original_url"].replace("/uploads/", "")
    watermarked_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    assert record["original_file_md5"] == main.path_md5(original_path)
    assert record["watermarked_file_md5"] == main.path_md5(watermarked_path)
```

In `test_main_exposes_required_python_api` in `tests/test_application_structure.py`, add `"file_md5"` and `"path_md5"` to `required`.

- [ ] **Step 2: Run the new assertions and verify they fail**

Run: `pytest tests/test_watermark_v4_api.py::test_v4_generation_persists_strict_codec_tag_and_feature_index tests/test_application_structure.py::test_main_exposes_required_python_api -v`

Expected: FAIL because the MD5 record fields and compatibility exports do not exist.

- [ ] **Step 3: Add path MD5 to watermark operations and generated records**

Add this field to `WatermarkOperations` in `trace_app/watermark/service.py`:

```python
    path_md5: Callable[[Path], str]
```

Calculate and persist hashes next to the existing SHA-256 fields:

```python
        original_file_md5 = op.path_md5(original_path)
        watermarked_file_md5 = op.path_md5(output_path)
        original_file_sha256 = op.path_sha256(original_path)
        watermarked_file_sha256 = op.path_sha256(output_path)
```

```python
            "original_file_md5": original_file_md5,
            "watermarked_file_md5": watermarked_file_md5,
            "original_file_sha256": original_file_sha256,
            "watermarked_file_sha256": watermarked_file_sha256,
```

Wire `path_md5=fingerprints.path_md5` in `trace_app/watermark/default_operations.py`.

- [ ] **Step 4: Add compatibility wrappers and dependency injection**

Add to `trace_app/compat.py`:

```python
def file_md5(content: bytes) -> str:
    return imaging_fingerprints.file_md5(content)


def path_md5(path: Path) -> str:
    return imaging_fingerprints.path_md5(path)
```

Pass `path_md5=path_md5` when constructing `WatermarkOperations`, pass `file_md5_fn=file_md5` into the compatibility `matched_file_fingerprint` wrapper, and add `file_md5` and `path_md5` to `__all__`.

- [ ] **Step 5: Run persistence, public API, and structure tests**

Run: `pytest tests/test_watermark_v4_api.py::test_v4_generation_persists_strict_codec_tag_and_feature_index tests/test_application_structure.py -v`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit generation and compatibility wiring**

```powershell
git add -- trace_app/watermark/service.py trace_app/watermark/default_operations.py trace_app/compat.py tests/test_watermark_v4_api.py tests/test_application_structure.py
git commit -m "feat: persist MD5 file fingerprints"
```

### Task 3: Original and Watermarked Upload Fast Success

**Files:**
- Modify: `tests/test_watermark_v4_api.py`
- Modify: `tests/test_application_structure.py`
- Modify: `trace_app/watermark/service.py`

- [ ] **Step 1: Replace the API rejection test with a failing dual-success test**

Rename `test_v4_exact_watermarked_fingerprint_succeeds_and_original_rejects` to `test_v4_exact_file_fingerprints_succeed_without_image_decode` and use:

```python
def test_v4_exact_file_fingerprints_succeed_without_image_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(main.app)
    record = _embed_v4(client).json()
    original_path = main.UPLOAD_DIR / record["original_url"].replace("/uploads/", "")
    watermarked_path = main.UPLOAD_DIR / record["download_url"].replace(
        "/uploads/", ""
    )
    monkeypatch.setattr(
        main,
        "load_image_from_bytes",
        lambda content: (_ for _ in ()).throw(AssertionError("unexpected decode")),
    )

    original_response = _extract_bytes(client, original_path.read_bytes(), "original.png")
    watermarked_response = _extract_bytes(
        client, watermarked_path.read_bytes(), "watermarked.png"
    )

    assert original_response.status_code == 200, original_response.text
    assert watermarked_response.status_code == 200, watermarked_response.text
    assert original_response.json()["trace_id"] == record["trace_id"]
    assert watermarked_response.json()["trace_id"] == record["trace_id"]
    assert original_response.json()["matched_file_type"] == "original"
    assert watermarked_response.json()["matched_file_type"] == "watermarked"
    assert original_response.json()["matched_hash_type"] == "file_md5_sha256"
    assert watermarked_response.json()["matched_hash_type"] == "file_md5_sha256"
    assert main.repository.read_detection_stats() == {
        "attempts": 2,
        "successes": 2,
    }
```

Replace the existing service statistics test in `tests/test_application_structure.py` with:

```python
@pytest.mark.parametrize("matched_file_type", ("original", "watermarked"))
def test_watermark_service_extract_upload_fingerprint_uses_repository_stats(
    tmp_path: Path,
    matched_file_type: str,
) -> None:
    repository = _WatermarkRepositorySpy()
    operations = replace(
        main.get_watermark_service().operations,
        matched_file_fingerprint=lambda content, records: {
            "trace_id": "TR-HIT",
            "matched_file_type": matched_file_type,
        },
    )
    service = WatermarkService(
        settings=Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="",
            admin_user="",
            admin_pass="",
        ),
        repository=repository,
        runtime=Runtime(),
        operations=operations,
    )

    result = asyncio.run(
        service.extract_upload(
            UploadFile(filename="matched.png", file=BytesIO(b"matched"))
        )
    )

    assert result["trace_id"] == "TR-HIT"
    assert repository.read_calls == 1
    assert repository.detection_results == [True]
```

- [ ] **Step 2: Run the fast-path tests and verify the original case fails**

Run: `pytest tests/test_watermark_v4_api.py::test_v4_exact_file_fingerprints_succeed_without_image_decode tests/test_application_structure.py -k "extract_upload_fingerprint" -v`

Expected: the original-file cases FAIL with HTTP 404 or a recorded unsuccessful detection.

- [ ] **Step 3: Remove the original-file rejection branch**

Reduce the fingerprint branch in `WatermarkService.extract_upload` to:

```python
        fingerprint_match = op.matched_file_fingerprint(content, records)
        if fingerprint_match:
            self.repository.record_detection_result(True)
            return fingerprint_match
```

Do not move `op.load_image_from_bytes(content)` above this branch.

- [ ] **Step 4: Run the fast-path and V4 API tests**

Run: `pytest tests/test_watermark_v4_api.py tests/test_application_structure.py -k "fingerprint or generation_persists" -v`

Expected: all selected tests PASS, including both file types and statistics assertions.

- [ ] **Step 5: Commit the upload behavior change**

```powershell
git add -- trace_app/watermark/service.py tests/test_watermark_v4_api.py tests/test_application_structure.py
git commit -m "fix: accept exact original file fingerprints"
```

### Task 4: Release Synchronization and Regression Verification

**Files:**
- Regenerate: `release/trace-v4-centos-20260715/trace_app/`
- Regenerate: `release/trace-v4-centos-20260715.zip`
- Regenerate: `release/trace-v4-centos-20260715.zip.sha256`

- [ ] **Step 1: Run the focused source regression suite**

Run: `pytest tests/test_application_structure.py tests/test_aligned_authenticated_detection.py tests/test_watermark_v4_api.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Rebuild the deterministic CentOS release**

Run: `python tools/build_centos_release.py`

Expected: prints one lowercase 64-character SHA-256 digest and updates the release directory, archive, and checksum file.

- [ ] **Step 3: Verify source and release Python modules are byte-identical**

Run:

```powershell
$source = Get-ChildItem trace_app -Recurse -File -Filter *.py
foreach ($file in $source) {
    $relative = $file.FullName.Substring((Resolve-Path .).Path.Length + 1)
    $releaseFile = Join-Path 'release/trace-v4-centos-20260715' $relative
    if (-not (Test-Path $releaseFile)) { throw "Missing release file: $relative" }
    if ((Get-FileHash $file.FullName -Algorithm SHA256).Hash -ne (Get-FileHash $releaseFile -Algorithm SHA256).Hash) {
        throw "Release mismatch: $relative"
    }
}
```

Expected: exits successfully with no output.

- [ ] **Step 4: Run release contract and focused commercial gates**

Run: `pytest tests/test_release_builder.py tests/test_centos_deploy_contract.py tests/test_watermark_v4_quick_matrix.py tests/test_false_positive_gate.py -v`

Expected: all selected tests PASS.

- [ ] **Step 5: Run the complete test suite**

Run: `pytest -q`

Expected: all tests PASS with only the repository's documented skips.

- [ ] **Step 6: Inspect the final diff and commit release artifacts**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only intended source, tests, release artifacts, and the untracked local runtime data are present.

```powershell
git add -- release/trace-v4-centos-20260715 release/trace-v4-centos-20260715.zip release/trace-v4-centos-20260715.zip.sha256
git commit -m "build: refresh CentOS release for MD5 tracing"
```

- [ ] **Step 7: Restart and smoke-test the local server**

Stop the process listening on port 8000, then restart the `mark` worktree application with the SQLite development override:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) { Stop-Process -Id $listener.OwningProcess }
$env:DB_URL = 'sqlite+pysqlite:///data/mark-dev.db'
$server = Start-Process python -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory (Resolve-Path .) -RedirectStandardOutput 'server-mark.stdout.log' -RedirectStandardError 'server-mark.stderr.log' -WindowStyle Hidden -PassThru
$server.Id
```

Run: `Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/`

Expected: HTTP 200. The dual-file endpoint behavior is exercised by `test_v4_exact_file_fingerprints_succeed_without_image_decode`; inspect `server-mark.stderr.log` and confirm it contains no traceback after startup.
