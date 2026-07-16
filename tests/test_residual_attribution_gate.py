import os
import sys
from pathlib import Path

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/pytest_residual_gate/uploads"
os.environ["DATA_DIR"] = "test_output/pytest_residual_gate/data"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
import numpy as np

import main
from trace_app.watermark import small_crop as small_crop_module
from tests.commercial_negative_benchmark import synthetic_family, synthetic_image


def test_residual_only_candidate_cannot_attribute_a_trace(monkeypatch):
    record = {
        "id": "record-1",
        "trace_id": "TR-RESIDUAL-ONLY",
        "user_id": "test-user",
        "robust_watermark": True,
        "created_at": "2026-07-10 00:00:00",
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(
        main,
        "record_visual_consistency",
        lambda image, candidate: (True, 120, 0.95, 0.80),
    )

    result = main.detect_by_residual_match(Image.new("RGB", (128, 128), "white"))

    assert result is None


def test_residual_candidate_evidence_retains_metrics_without_identity(monkeypatch):
    record = {
        "id": "record-1",
        "trace_id": "TR-RESIDUAL-ONLY",
        "user_id": "test-user",
        "robust_watermark": True,
        "created_at": "2026-07-10 00:00:00",
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(
        main,
        "record_visual_consistency",
        lambda image, candidate: (True, 120, 0.95, 0.80),
    )

    evidence = main.residual_candidate_evidence(Image.new("RGB", (128, 128), "white"))

    assert evidence == {
        "candidate_id": "record-1",
        "candidate_trace_id": "TR-RESIDUAL-ONLY",
        "visual_inliers": 120,
        "visual_ratio": 0.95,
        "residual_score": 0.8,
    }


def test_synthetic_negative_generation_is_deterministic():
    first = np.asarray(synthetic_image(137))
    second = np.asarray(synthetic_image(137))

    assert np.array_equal(first, second)


def test_synthetic_negative_generation_covers_ten_families():
    assert {synthetic_family(index) for index in range(10)} == {
        "solid",
        "gradient",
        "correlated_noise",
        "grid",
        "ui_blocks",
        "periodic",
        "text_edges",
        "radial",
        "checker",
        "low_contrast",
    }


def test_small_crop_detector_rejects_without_scanning_when_visual_candidates_fail(monkeypatch):
    record = {
        "trace_id": "TR-NO-VISUAL-CANDIDATE",
        "robust_watermark": True,
        "watermark_code_version": main.CODE_WATERMARK_VERSION,
        "small_crop_trace_enabled": True,
        "small_crop_trace_version": main.SMALL_TRACE_VERSION,
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(main, "record_visual_consistency", lambda image, candidate: (False, 0, 0.0, 0.0))
    monkeypatch.setattr(
        small_crop_module,
        "iter_small_trace_windows",
        lambda width, height: (_ for _ in ()).throw(AssertionError("window scan must not run")),
    )

    assert main.detect_small_crop_trace(Image.new("RGB", (256, 256), "white")) is None


def test_multiscale_code_detector_rejects_without_scanning_when_visual_candidates_fail(monkeypatch):
    record = {
        "trace_id": "TR-NO-VISUAL-CANDIDATE",
        "robust_watermark": True,
        "watermark_code_version": main.CODE_WATERMARK_VERSION,
    }
    monkeypatch.setattr(main, "read_records", lambda: [record])
    monkeypatch.setattr(main, "record_visual_consistency", lambda image, candidate: (False, 0, 0.0, 0.0))
    monkeypatch.setattr(
        small_crop_module,
        "code_scan_signal_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("code scan must not run")),
    )

    assert main.detect_watermark_code(Image.new("RGB", (256, 256), "white")) is None
