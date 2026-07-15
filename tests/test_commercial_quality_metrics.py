import math

import numpy as np
import pytest
from PIL import Image

from tests.commercial_quality_benchmark import select_recommended_config
from tests.commercial_quality_metrics import metric_distribution, quality_gate, quality_metrics


def test_identical_images_have_perfect_quality():
    image = Image.new("RGB", (64, 64), (100, 120, 140))

    metrics = quality_metrics(image, image)

    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0


def test_perturbed_image_has_finite_quality_metrics():
    original = np.full((64, 64, 3), 128, dtype=np.uint8)
    changed = original.copy()
    changed[16:48, 16:48, 2] = 132

    metrics = quality_metrics(Image.fromarray(original), Image.fromarray(changed))

    assert math.isfinite(metrics["psnr"])
    assert 0.0 < metrics["ssim"] < 1.0
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0
    assert metrics["max_abs_diff"] == 4


def test_quality_metrics_reject_different_image_sizes():
    with pytest.raises(ValueError, match="same dimensions"):
        quality_metrics(Image.new("RGB", (64, 64)), Image.new("RGB", (32, 64)))


def test_quality_gate_requires_both_psnr_and_ssim():
    assert quality_gate({"psnr": 40.0, "ssim": 0.99}) is True
    assert quality_gate({"psnr": 37.99, "ssim": 0.99}) is False
    assert quality_gate({"psnr": 40.0, "ssim": 0.9799}) is False


def test_metric_distribution_reports_required_percentiles():
    rows = [{"psnr": value} for value in (38.0, 39.0, 40.0, 41.0, 42.0)]

    result = metric_distribution(rows, "psnr")

    assert result == {
        "min": 38.0,
        "p5": 38.2,
        "p50": 40.0,
        "p95": 41.8,
        "mean": 40.0,
    }


def test_selector_prefers_least_damage_among_configs_that_pass_trace_gates():
    configs = [
        {
            "fidelity": 0.75,
            "quality_pass": True,
            "wrong": 0,
            "false_positive": 0,
            "probe_recall": 1.0,
            "min_ssim": 0.985,
            "min_psnr": 39.0,
        },
        {
            "fidelity": 0.90,
            "quality_pass": True,
            "wrong": 0,
            "false_positive": 0,
            "probe_recall": 0.90,
            "min_ssim": 0.993,
            "min_psnr": 42.0,
        },
        {
            "fidelity": 0.85,
            "quality_pass": True,
            "wrong": 0,
            "false_positive": 0,
            "probe_recall": 1.0,
            "min_ssim": 0.990,
            "min_psnr": 41.0,
        },
    ]

    selected = select_recommended_config(configs, min_probe_recall=0.95)

    assert selected is not None
    assert selected["fidelity"] == 0.85


@pytest.mark.parametrize(
    "override",
    [
        {"quality_pass": False},
        {"wrong": 1},
        {"false_positive": 1},
        {"probe_recall": 0.94},
    ],
)
def test_selector_rejects_any_config_that_fails_a_hard_gate(override):
    config = {
        "fidelity": 0.80,
        "quality_pass": True,
        "wrong": 0,
        "false_positive": 0,
        "probe_recall": 1.0,
        "min_ssim": 0.99,
        "min_psnr": 40.0,
        **override,
    }

    assert select_recommended_config([config], min_probe_recall=0.95) is None
