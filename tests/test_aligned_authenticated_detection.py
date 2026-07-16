from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image, ImageDraw

import main


def textured_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    width, height = size
    rng = np.random.default_rng(20260710)
    array = rng.integers(20, 235, size=(height, width, 3), dtype=np.uint8)
    array = cv2.GaussianBlur(array, (0, 0), sigmaX=0.7)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(24):
        x = 20 + (index * 73) % (width - 80)
        y = 20 + (index * 47) % (height - 60)
        draw.rectangle((x, y, x + 45, y + 28), outline=(255, 255, 255), width=3)
    return image


def record_for_target(tmp_path: Path, monkeypatch, target: Image.Image) -> dict:
    upload_dir = tmp_path / "uploads"
    target_path = upload_dir / "watermarked" / "target.png"
    target_path.parent.mkdir(parents=True)
    target.save(target_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    return {
        "id": "aligned-record",
        "trace_id": "TR-ALIGNED-TEST",
        "download_url": "/uploads/watermarked/target.png",
    }


def test_align_query_to_record_recovers_scaled_crop(tmp_path, monkeypatch):
    target = textured_image()
    record = record_for_target(tmp_path, monkeypatch, target)
    scaled = target.resize((960, 720), Image.Resampling.BICUBIC)
    query = scaled.crop((90, 75, 870, 650))

    alignment = main.align_query_to_record(query, record)

    assert alignment is not None
    assert alignment["image"].shape == (480, 640, 3)
    assert alignment["valid_mask"].shape == (480, 640)
    assert alignment["inliers"] >= 18
    assert alignment["ratio"] >= 0.32
    assert 0.05 <= alignment["coverage"] <= 1.0
    assert alignment["target_scale"] == 1.0


def test_align_query_to_record_rejects_unrelated_blank_image(tmp_path, monkeypatch):
    target = textured_image()
    record = record_for_target(tmp_path, monkeypatch, target)

    alignment = main.align_query_to_record(Image.new("RGB", (640, 480), "white"), record)

    assert alignment is None


def test_align_query_to_record_rejects_missing_target(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    record = {"download_url": "/uploads/watermarked/missing.png"}

    assert main.align_query_to_record(textured_image(), record) is None


def test_aligned_tile_iterator_uses_recorded_density_offsets():
    aligned = np.full((240, 240, 3), 128, dtype=np.uint8)
    valid_mask = np.ones((240, 240), dtype=bool)

    low = list(main.iter_aligned_small_trace_tiles(aligned, valid_mask, {"small_crop_trace_density": "low"}))
    medium = list(main.iter_aligned_small_trace_tiles(aligned, valid_mask, {"small_crop_trace_density": "medium"}))
    high = list(main.iter_aligned_small_trace_tiles(aligned, valid_mask, {"small_crop_trace_density": "high"}))

    assert len(low) > 0
    assert len(low) < len(medium) < len(high)
    assert all(item["tile"].shape == (main.SMALL_TRACE_TILE, main.SMALL_TRACE_TILE, 3) for item in high)
    expected_offsets = set(main.small_crop_density_offsets("high"))
    assert {item["offset"] for item in high} == expected_offsets


def test_aligned_tile_iterator_excludes_tiles_below_seventy_percent_coverage():
    aligned = np.full((192, 192, 3), 128, dtype=np.uint8)
    valid_mask = np.ones((192, 192), dtype=bool)
    valid_mask[:40, :96] = False

    tiles = list(
        main.iter_aligned_small_trace_tiles(
            aligned,
            valid_mask,
            {"small_crop_trace_density": "low"},
        )
    )

    assert all(item["coverage"] >= 0.70 for item in tiles)
    assert (0, 0) not in {item["position"] for item in tiles}


def identity_alignment(image: Image.Image, valid_mask: np.ndarray | None = None) -> dict:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "image": array,
        "valid_mask": valid_mask if valid_mask is not None else np.ones(array.shape[:2], dtype=bool),
        "target_scale": 1.0,
        "inliers": 100,
        "ratio": 1.0,
        "coverage": 1.0,
    }


def test_aligned_robust_decoder_recovers_full_trace_with_soft_tile_aggregation():
    trace_id = "TR-ALIGNED-ROBUST"
    original = textured_image((1024, 1024))
    marked = main.embed_robust_watermark(original, trace_id, strength_scale=0.28)
    record = {"id": "record-1", "trace_id": trace_id, "robust_watermark": True}

    result = main.decode_aligned_robust_trace(identity_alignment(marked), record)

    assert result is not None
    assert result["trace_id"] == trace_id
    assert result["bit_errors"] <= 4
    assert result["authenticated_tiles"] >= 2


def test_rs_v2_embedding_keeps_v1_per_tile_damage_budget():
    trace_id = "TR-RS-V2-QUALITY"
    original = textured_image((512, 512))
    original_array = np.asarray(original, dtype=np.int16)

    v1 = np.asarray(
        main.embed_robust_watermark(original, trace_id, strength_scale=1.0),
        dtype=np.int16,
    )
    v2 = np.asarray(
        main.embed_robust_watermark_v2(original, trace_id, strength_scale=1.0),
        dtype=np.int16,
    )
    v1_delta = np.abs(v1 - original_array)
    v2_delta = np.abs(v2 - original_array)

    changed_ratio_delta = abs(
        np.count_nonzero(v2_delta) - np.count_nonzero(v1_delta)
    ) / original_array.size
    assert changed_ratio_delta < 0.001
    assert int(v2_delta.max()) == int(v1_delta.max())


def test_hmac_v3_embedding_keeps_v1_per_tile_damage_budget():
    original = textured_image((512, 512))
    original_array = np.asarray(original, dtype=np.int16)
    auth_code = bytes.fromhex("0123456789abcdef")

    v1 = np.asarray(
        main.embed_robust_watermark(original, "TR-V3-QUALITY", strength_scale=0.74),
        dtype=np.int16,
    )
    v3 = np.asarray(
        main.embed_robust_watermark_v3(original, auth_code, strength_scale=0.74),
        dtype=np.int16,
    )
    v1_delta = np.abs(v1 - original_array)
    v3_delta = np.abs(v3 - original_array)

    changed_ratio_delta = abs(
        np.count_nonzero(v3_delta) - np.count_nonzero(v1_delta)
    ) / original_array.size
    assert changed_ratio_delta < 0.001
    assert int(v3_delta.max()) == int(v1_delta.max())


def test_aligned_rs_v2_decoder_recovers_exact_trace():
    trace_id = "TR-RS-V2-ALIGNED"
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v2(original, trace_id, strength_scale=1.0)
    record = {
        "id": "record-rs-v2",
        "trace_id": trace_id,
        "robust_watermark": True,
        "robust_watermark_version": 2,
    }

    decoded = main.decode_aligned_robust_trace_v2(identity_alignment(marked), record)

    assert decoded is not None
    assert decoded["trace_id"] == trace_id
    assert decoded["phase_tile_counts"] == [8, 8, 8]
    assert decoded["bit_errors"] <= 32
    assert decoded["recovery_method"] in {
        "reed_solomon",
        "expected_codeword_distance",
    }


def test_aligned_rs_v2_decoder_rejects_wrong_record():
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v2(original, "TR-RS-V2-A", strength_scale=1.0)
    wrong_record = {
        "id": "wrong-record",
        "trace_id": "TR-RS-V2-B",
        "robust_watermark": True,
        "robust_watermark_version": 2,
    }

    assert main.decode_aligned_robust_trace_v2(
        identity_alignment(marked),
        wrong_record,
    ) is None


def test_aligned_rs_v2_decoder_requires_two_tiles_per_phase():
    trace_id = "TR-RS-V2-COVERAGE"
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v2(original, trace_id, strength_scale=1.0)
    mask = np.zeros((512, 768), dtype=bool)
    mask[: main.ROBUST_TILE, : main.ROBUST_TILE * 3] = True
    record = {
        "id": "coverage-record",
        "trace_id": trace_id,
        "robust_watermark": True,
        "robust_watermark_version": 2,
    }

    assert main.decode_aligned_robust_trace_v2(
        identity_alignment(marked, mask),
        record,
    ) is None


def test_aligned_rs_v2_decoder_rejects_unwatermarked_image():
    original = textured_image((768, 512))
    record = {
        "id": "unwatermarked-record",
        "trace_id": "TR-RS-V2-ABSENT",
        "robust_watermark": True,
        "robust_watermark_version": 2,
    }

    assert main.decode_aligned_robust_trace_v2(
        identity_alignment(original),
        record,
    ) is None


def test_aligned_hmac_v3_decoder_recovers_exact_candidate():
    auth_code = bytes.fromhex("0123456789abcdef")
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v3(original, auth_code, strength_scale=0.74)
    record = {
        "id": "record-hmac-v3",
        "trace_id": "TR-HMAC-V3-ALIGNED",
        "robust_watermark": True,
        "robust_watermark_version": 3,
        "robust_auth_code": auth_code.hex(),
    }

    decoded = main.decode_aligned_robust_trace_v3(identity_alignment(marked), record)

    assert decoded is not None
    assert decoded["trace_id"] == record["trace_id"]
    assert decoded["bit_errors"] <= 8
    assert decoded["authenticated_tiles"] == 24
    assert sum(count > 0 for count in decoded["phase_tile_counts"]) == 3
    assert decoded["mean_signed_agreement"] > 0


def test_aligned_hmac_v3_decoder_rejects_wrong_candidate_code():
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v3(
        original,
        bytes.fromhex("0123456789abcdef"),
        strength_scale=0.74,
    )
    wrong_record = {
        "trace_id": "TR-HMAC-V3-WRONG",
        "robust_watermark_version": 3,
        "robust_auth_code": "fedcba9876543210",
    }

    assert main.decode_aligned_robust_trace_v3(
        identity_alignment(marked),
        wrong_record,
    ) is None


def test_aligned_hmac_v3_decoder_rejects_missing_code_and_unwatermarked_image():
    original = textured_image((768, 512))
    missing_code = {"trace_id": "TR-HMAC-V3-MISSING", "robust_watermark_version": 3}
    valid_record = {
        "trace_id": "TR-HMAC-V3-ABSENT",
        "robust_watermark_version": 3,
        "robust_auth_code": "0123456789abcdef",
    }

    assert main.decode_aligned_robust_trace_v3(identity_alignment(original), missing_code) is None
    assert main.decode_aligned_robust_trace_v3(identity_alignment(original), valid_record) is None


def test_aligned_hmac_v3_decoder_requires_multiple_phases():
    auth_code = bytes.fromhex("0123456789abcdef")
    original = textured_image((768, 512))
    marked = main.embed_robust_watermark_v3(original, auth_code, strength_scale=0.74)
    mask = np.zeros((512, 768), dtype=bool)
    for tile_y in range(4):
        for tile_x in range(6):
            if (tile_x + 2 * tile_y) % 3 == 0:
                y0 = tile_y * main.ROBUST_TILE
                x0 = tile_x * main.ROBUST_TILE
                mask[y0 : y0 + main.ROBUST_TILE, x0 : x0 + main.ROBUST_TILE] = True
    record = {
        "trace_id": "TR-HMAC-V3-PHASE",
        "robust_watermark_version": 3,
        "robust_auth_code": auth_code.hex(),
    }

    assert main.decode_aligned_robust_trace_v3(
        identity_alignment(marked, mask),
        record,
    ) is None


def test_aligned_robust_decoder_rejects_unwatermarked_image():
    trace_id = "TR-ALIGNED-ROBUST"
    original = textured_image((1024, 1024))
    record = {"id": "record-1", "trace_id": trace_id, "robust_watermark": True}

    assert main.decode_aligned_robust_trace(identity_alignment(original), record) is None


def test_aligned_robust_decoder_requires_multiple_covered_tiles():
    trace_id = "TR-ALIGNED-ROBUST"
    original = textured_image((512, 512))
    marked = main.embed_robust_watermark(original, trace_id, strength_scale=1.0)
    mask = np.zeros((512, 512), dtype=bool)
    mask[: main.ROBUST_TILE, : main.ROBUST_TILE] = True
    record = {"id": "record-1", "trace_id": trace_id, "robust_watermark": True}

    assert main.decode_aligned_robust_trace(identity_alignment(marked, mask), record) is None


def stored_robust_record(tmp_path: Path, monkeypatch, trace_id: str):
    original = textured_image((1024, 1024))
    marked = main.embed_robust_watermark(original, trace_id, strength_scale=0.28)
    upload_dir = tmp_path / "uploads"
    marked_path = upload_dir / "watermarked" / "marked.png"
    original_path = upload_dir / "originals" / "original.png"
    marked_path.parent.mkdir(parents=True)
    original_path.parent.mkdir(parents=True)
    marked.save(marked_path)
    original.save(original_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    return original, marked, {
        "id": "record-1",
        "trace_id": trace_id,
        "user_id": "aligned-user",
        "download_url": "/uploads/watermarked/marked.png",
        "original_url": "/uploads/originals/original.png",
        "robust_watermark": True,
        "created_at": "2026-07-10 00:00:00",
    }


def stored_rs_v2_record(tmp_path: Path, monkeypatch, trace_id: str):
    original = textured_image((1024, 1024))
    marked = main.embed_robust_watermark_v2(original, trace_id, strength_scale=1.0)
    upload_dir = tmp_path / "uploads"
    marked_path = upload_dir / "watermarked" / "marked-v2.png"
    original_path = upload_dir / "originals" / "original-v2.png"
    marked_path.parent.mkdir(parents=True)
    original_path.parent.mkdir(parents=True)
    marked.save(marked_path)
    original.save(original_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    return original, marked, {
        "id": "record-rs-v2",
        "trace_id": trace_id,
        "user_id": "aligned-rs-v2-user",
        "download_url": "/uploads/watermarked/marked-v2.png",
        "original_url": "/uploads/originals/original-v2.png",
        "robust_watermark": True,
        "robust_watermark_version": 2,
        "robust_watermark_codec": "rs_24_8_three_phase",
        "created_at": "2026-07-10 00:00:00",
    }


def stored_hmac_v3_record(tmp_path: Path, monkeypatch, trace_id: str):
    auth_code = bytes.fromhex("0123456789abcdef")
    original = textured_image((1024, 1024))
    marked = main.embed_robust_watermark_v3(original, auth_code, strength_scale=0.74)
    upload_dir = tmp_path / "uploads"
    marked_path = upload_dir / "watermarked" / "marked-v3.png"
    original_path = upload_dir / "originals" / "original-v3.png"
    marked_path.parent.mkdir(parents=True)
    original_path.parent.mkdir(parents=True)
    marked.save(marked_path)
    original.save(original_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    return original, marked, {
        "id": "record-hmac-v3",
        "trace_id": trace_id,
        "user_id": "aligned-hmac-v3-user",
        "download_url": "/uploads/watermarked/marked-v3.png",
        "original_url": "/uploads/originals/original-v3.png",
        "robust_watermark": True,
        "robust_watermark_version": 3,
        "robust_watermark_codec": "hmac64_full_repeat_phase_permutation_v3",
        "robust_auth_code": auth_code.hex(),
        "created_at": "2026-07-10 00:00:00",
    }


def test_aligned_detector_recovers_scaled_crop(tmp_path, monkeypatch):
    trace_id = "TR-ALIGNED-INTEGRATION"
    _, marked, record = stored_robust_record(tmp_path, monkeypatch, trace_id)
    scaled = marked.resize((1536, 1536), Image.Resampling.BICUBIC)
    query = scaled.crop((140, 110, 1390, 1370))

    result = main.detect_aligned_authenticated_watermark(query, records=[record])

    assert result is not None
    assert result["trace_id"] == trace_id
    assert result["mode"] == "aligned_robust_code"
    assert result["code_recovery"]["bit_errors"] <= 4


def test_aligned_rs_v2_detector_recovers_scaled_crop(tmp_path, monkeypatch):
    trace_id = "TR-RS-V2-INTEGRATION"
    _, marked, record = stored_rs_v2_record(tmp_path, monkeypatch, trace_id)
    scaled = marked.resize((1536, 1536), Image.Resampling.BICUBIC)
    query = scaled.crop((140, 110, 1390, 1370))

    result = main.detect_aligned_authenticated_watermark(query, records=[record])

    assert result is not None
    assert result["trace_id"] == trace_id
    assert result["mode"] == "aligned_robust_rs_v2"
    assert result["code_recovery"]["bit_errors"] <= 32


def test_aligned_hmac_v3_detector_recovers_scaled_crop(tmp_path, monkeypatch):
    trace_id = "TR-HMAC-V3-INTEGRATION"
    _, marked, record = stored_hmac_v3_record(tmp_path, monkeypatch, trace_id)
    scaled = marked.resize((1536, 1536), Image.Resampling.BICUBIC)
    query = scaled.crop((140, 110, 1390, 1370))

    result = main.detect_aligned_authenticated_watermark(query, records=[record])

    assert result is not None
    assert result["trace_id"] == trace_id
    assert result["mode"] == "aligned_robust_hmac_v3"
    assert result["code_recovery"]["bit_errors"] <= 8


def test_aligned_detector_rejects_unwatermarked_source_variant(tmp_path, monkeypatch):
    trace_id = "TR-ALIGNED-INTEGRATION"
    original, _, record = stored_robust_record(tmp_path, monkeypatch, trace_id)
    query = original.resize((900, 900), Image.Resampling.BICUBIC)

    assert main.detect_aligned_authenticated_watermark(query, records=[record]) is None


def test_robust_watermark_version_normalization_is_environment_independent(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_ROBUST_WATERMARK_VERSION", "3")

    assert main.normalize_robust_watermark_version(None) == 1
    assert main.normalize_robust_watermark_version("1") == 1
    assert main.normalize_robust_watermark_version("2") == 2
    assert main.normalize_robust_watermark_version("3") == 3
    assert main.normalize_robust_watermark_version("invalid") == 1


def test_aligned_detector_dispatches_v2_without_v1_fallback(monkeypatch):
    record = {
        "trace_id": "TR-RS-V2-DISPATCH",
        "robust_watermark": True,
        "robust_watermark_version": 2,
    }
    alignment = identity_alignment(textured_image((384, 384)))
    monkeypatch.setattr(main, "align_query_to_record", lambda image, candidate: alignment)
    monkeypatch.setattr(main, "decode_aligned_robust_trace_v2", lambda value, candidate: None)
    monkeypatch.setattr(
        main,
        "decode_aligned_robust_trace",
        lambda value, candidate: (_ for _ in ()).throw(AssertionError("v1 fallback")),
    )

    result = main.detect_aligned_authenticated_watermark(
        textured_image((384, 384)),
        records=[record],
    )

    assert result is None


def test_missing_record_version_dispatches_to_v1(monkeypatch):
    record = {"trace_id": "TR-V1-DISPATCH", "robust_watermark": True}
    alignment = identity_alignment(textured_image((384, 384)))
    called = []
    monkeypatch.setattr(main, "align_query_to_record", lambda image, candidate: alignment)
    monkeypatch.setattr(
        main,
        "decode_aligned_robust_trace",
        lambda value, candidate: called.append("v1") or None,
    )
    monkeypatch.setattr(
        main,
        "decode_aligned_robust_trace_v2",
        lambda value, candidate: (_ for _ in ()).throw(AssertionError("v2 dispatch")),
    )

    assert main.detect_aligned_authenticated_watermark(
        textured_image((384, 384)),
        records=[record],
    ) is None
    assert called == ["v1"]


def test_two_authenticated_v2_candidates_return_none(monkeypatch):
    records = [
        {"trace_id": "TR-V2-A", "robust_watermark": True, "robust_watermark_version": 2},
        {"trace_id": "TR-V2-B", "robust_watermark": True, "robust_watermark_version": 2},
    ]
    alignment = identity_alignment(textured_image((384, 384)))
    monkeypatch.setattr(main, "align_query_to_record", lambda image, candidate: alignment)
    monkeypatch.setattr(
        main,
        "decode_aligned_robust_trace_v2",
        lambda value, candidate: {
            "record": candidate,
            "trace_id": candidate["trace_id"],
            "corrected_symbols": 0,
            "erasure_count": 0,
            "bit_errors": 0,
            "recovery_method": "reed_solomon",
            "phase_tile_counts": [2, 2, 2],
            "mean_abs_score": 1.0,
        },
    )

    assert main.detect_aligned_authenticated_watermark(
        textured_image((384, 384)),
        records=records,
    ) is None


def test_legacy_robust_candidates_exclude_v2_records(monkeypatch):
    records = [
        {"trace_id": "TR-V1", "robust_watermark": True},
        {"trace_id": "TR-V2", "robust_watermark": True, "robust_watermark_version": 2},
    ]
    monkeypatch.setattr(main, "read_records", lambda: records)

    candidates = main.legacy_robust_candidate_records()

    assert [record["trace_id"] for record in candidates] == ["TR-V1"]


def test_extract_pipeline_uses_aligned_authenticated_detector(monkeypatch):
    expected = {"trace_id": "TR-PIPELINE-ALIGNED", "mode": "aligned_robust_code"}
    monkeypatch.setattr(main, "v4_candidate_records", lambda: [])
    monkeypatch.setattr(main, "extract_full_lsb", lambda image: None)
    monkeypatch.setattr(main, "is_registered_original_image", lambda image: False)
    monkeypatch.setattr(main, "should_run_frequency_fallbacks", lambda image: True)
    monkeypatch.setattr(main, "detect_dot_matrix_trace", lambda image: None)
    monkeypatch.setattr(main, "detect_aligned_authenticated_watermark", lambda image, **kwargs: expected)
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: [],
            record_detection_result=lambda success: None,
        ),
    )
    monkeypatch.setattr(main.app.state, "aligned_authenticated_detection_enabled", True, raising=False)

    result = main.extract_watermark_from_image(Image.new("RGB", (256, 256), "white"))

    assert result == expected


def test_registered_original_check_does_not_decode_different_sizes(tmp_path, monkeypatch):
    original_path = tmp_path / "uploads" / "originals" / "source.png"
    original_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"placeholder")
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [{"original_url": "/uploads/originals/source.png"}],
    )

    convert_calls = []

    class HeaderOnlyImage:
        size = (200, 150)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            convert_calls.append(mode)
            return self

    monkeypatch.setattr(main.Image, "open", lambda path: HeaderOnlyImage())

    assert main.is_registered_original_image(Image.new("RGB", (100, 100))) is False
    assert convert_calls == []


def test_fingerprint_check_does_not_hash_legacy_record_files(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    for relative in ("originals/legacy.png", "watermarked/legacy.png"):
        path = upload_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy")
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [{
            "trace_id": "TR-LEGACY-NO-HASH",
            "original_url": "/uploads/originals/legacy.png",
            "download_url": "/uploads/watermarked/legacy.png",
        }],
    )
    hash_calls = []
    monkeypatch.setattr(main, "path_sha256", lambda path: hash_calls.append(path) or "NOPE")

    assert main.matched_file_fingerprint(b"not-a-match") is None
    assert hash_calls == []


def test_fingerprint_check_uses_stored_hash_without_file_io(monkeypatch):
    content = b"stored-watermarked-file"
    digest = main.file_sha256(content)
    record = {
        "id": "stored-hash-record",
        "trace_id": "TR-STORED-HASH",
        "watermarked_file_sha256": digest,
        "user_id": "hash-user",
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(
        main,
        "path_sha256",
        lambda path: (_ for _ in ()).throw(AssertionError("unexpected file IO")),
    )

    result = main.matched_file_fingerprint(content)

    assert result is not None
    assert result["trace_id"] == "TR-STORED-HASH"
    assert result["matched_hash_type"] == "file_bytes"


def test_extract_pipeline_skips_dense_fallbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(main, "v4_candidate_records", lambda: [])
    monkeypatch.setattr(main, "extract_full_lsb", lambda image: None)
    monkeypatch.setattr(main, "extract_block_lsb", lambda image: None)
    monkeypatch.setattr(main, "is_registered_original_image", lambda image: False)
    monkeypatch.setattr(main, "should_run_frequency_fallbacks", lambda image: True)
    monkeypatch.setattr(main, "detect_dot_matrix_trace", lambda image: None)
    monkeypatch.setattr(main, "detect_aligned_authenticated_watermark", lambda image, **kwargs: None)
    monkeypatch.setattr(main, "detect_small_crop_trace", lambda image: (_ for _ in ()).throw(AssertionError("dense scan")))
    monkeypatch.setattr(main, "detect_watermark_code", lambda image: (_ for _ in ()).throw(AssertionError("dense scan")))
    monkeypatch.setattr(main, "detect_robust_watermark", lambda image: (_ for _ in ()).throw(AssertionError("dense scan")))
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: [],
            record_detection_result=lambda success: None,
        ),
    )
    monkeypatch.setattr(main.app.state, "aligned_authenticated_detection_enabled", True, raising=False)
    monkeypatch.setattr(main.app.state, "dense_watermark_fallback_enabled", False, raising=False)
    monkeypatch.setattr(main.app.state, "visual_match_fallback_enabled", False)
    monkeypatch.setattr(main.app.state, "visible_watermark_detection_enabled", False)

    with pytest.raises(HTTPException) as exc:
        main.extract_watermark_from_image(Image.new("RGB", (256, 256), "white"))

    assert exc.value.status_code == 404


def test_aligned_candidates_are_ranked_by_aspect_ratio(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    target_dir = upload_dir / "watermarked"
    target_dir.mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(target_dir / "four-three.png")
    Image.new("RGB", (640, 360)).save(target_dir / "wide.png")
    Image.new("RGB", (300, 300)).save(target_dir / "square.png")
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    records = [
        {"trace_id": "wide", "download_url": "/uploads/watermarked/wide.png", "robust_watermark": True},
        {"trace_id": "square", "download_url": "/uploads/watermarked/square.png", "robust_watermark": True},
        {"trace_id": "four-three", "download_url": "/uploads/watermarked/four-three.png", "robust_watermark": True},
    ]

    ranked = main.rank_aligned_candidates(Image.new("RGB", (800, 600)), records)

    assert [record["trace_id"] for record in ranked] == ["four-three", "square", "wide"]


def test_aligned_candidate_ranking_uses_recorded_dimensions_without_opening_files(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "missing-uploads")
    records = [
        {"trace_id": "wide", "image_width": 1600, "image_height": 900, "robust_watermark": True},
        {"trace_id": "four-three", "image_width": 1200, "image_height": 900, "robust_watermark": True},
    ]

    ranked = main.rank_aligned_candidates(Image.new("RGB", (800, 600)), records)

    assert [record["trace_id"] for record in ranked] == ["four-three", "wide"]


def test_aligned_candidates_prefer_content_index_over_aspect_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(main.app.state, "generated_trace_ids", [], raising=False)
    target = textured_image((800, 450))
    query = target.crop((180, 20, 620, 440)).resize((600, 600), Image.Resampling.BICUBIC)
    decoy = textured_image((600, 600)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    target_index = main.save_record_feature_index(target, "target-record")
    decoy_index = main.save_record_feature_index(decoy, "decoy-record")
    records = [
        {
            "id": "decoy-record",
            "trace_id": "decoy",
            "image_width": 600,
            "image_height": 600,
            "feature_index_path": decoy_index,
            "robust_watermark": True,
        },
        {
            "id": "target-record",
            "trace_id": "target",
            "image_width": 800,
            "image_height": 450,
            "feature_index_path": target_index,
            "robust_watermark": True,
        },
    ]

    ranked = main.rank_aligned_candidates(query, records)

    assert ranked[0]["trace_id"] == "target"
    assert ranked[0]["_feature_match_count"] >= 12


def test_recent_unindexed_record_is_reserved_a_candidate_slot(monkeypatch):
    records = [
        {
            "id": f"record-{index}",
            "trace_id": f"TR-{index}",
            "image_width": 1500,
            "image_height": 1000,
            "robust_watermark": True,
        }
        for index in range(12)
    ]
    target = {
        "id": "recent-target",
        "trace_id": "TR-RECENT-TARGET",
        "image_width": 2200,
        "image_height": 1200,
        "robust_watermark": True,
        "created_at": "2026-07-10 22:09:45",
    }
    records.append(target)
    monkeypatch.setattr(
        main.app.state,
        "generated_trace_ids",
        ["TR-RECENT-TARGET"],
        raising=False,
    )

    ranked = main.rank_aligned_candidates(Image.new("RGB", (1500, 1000)), records)

    assert target in ranked[:2]


def test_latest_persisted_record_is_reserved_after_process_restart(monkeypatch):
    latest = {
        "id": "latest-record",
        "trace_id": "TR-LATEST",
        "image_width": 2200,
        "image_height": 1200,
        "robust_watermark": True,
        "created_at": "2026-07-10 22:09:45",
    }
    decoys = [
        {
            "id": f"decoy-{index}",
            "trace_id": f"TR-DECOY-{index}",
            "image_width": 1500,
            "image_height": 1000,
            "robust_watermark": True,
        }
        for index in range(12)
    ]
    monkeypatch.setattr(main.app.state, "generated_trace_ids", [], raising=False)

    ranked = main.rank_aligned_candidates(
        Image.new("RGB", (1500, 1000)),
        [latest, *decoys],
    )

    assert latest in ranked[:2]


def test_recent_backfill_finds_content_match_behind_newer_decoys(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    watermarked_dir = upload_dir / "watermarked"
    watermarked_dir.mkdir(parents=True)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(main.app.state, "generated_trace_ids", [], raising=False)

    target = textured_image((800, 450))
    query = target.crop((180, 20, 620, 440)).resize((600, 600), Image.Resampling.BICUBIC)
    images = [
        textured_image((600, 600)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        Image.new("RGB", (600, 600), (80, 100, 120)),
        target,
    ]
    records = []
    for index, image in enumerate(images):
        image.save(watermarked_dir / f"record-{index}.png")
        records.append({
            "id": f"record-{index}",
            "trace_id": f"TR-{index}",
            "download_url": f"/uploads/watermarked/record-{index}.png",
            "image_width": image.width,
            "image_height": image.height,
            "robust_watermark": True,
            "created_at": f"2026-07-10 22:2{3-index}:00",
        })
    records.extend({
        "id": f"aspect-decoy-{index}",
        "trace_id": f"TR-ASPECT-{index}",
        "image_width": 600,
        "image_height": 600,
        "robust_watermark": True,
    } for index in range(10))

    ranked = main.rank_aligned_candidates(query, records)

    assert ranked[0]["trace_id"] == "TR-2"
    assert ranked[0]["_feature_match_count"] >= 12


def test_robust_strength_has_independent_commercial_default_and_bounds():
    assert main.robust_strength_to_scale(None) == 1.0
    assert main.robust_strength_to_scale("-1") == 0.0
    assert main.robust_strength_to_scale("3") == 2.0
