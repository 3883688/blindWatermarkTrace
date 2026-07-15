import math

import cv2
import numpy as np
from PIL import Image


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def _ssim(original: np.ndarray, changed: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_original = cv2.GaussianBlur(original, kernel, sigma)
    mu_changed = cv2.GaussianBlur(changed, kernel, sigma)
    mu_original_sq = mu_original * mu_original
    mu_changed_sq = mu_changed * mu_changed
    mu_product = mu_original * mu_changed

    sigma_original_sq = cv2.GaussianBlur(original * original, kernel, sigma) - mu_original_sq
    sigma_changed_sq = cv2.GaussianBlur(changed * changed, kernel, sigma) - mu_changed_sq
    sigma_product = cv2.GaussianBlur(original * changed, kernel, sigma) - mu_product

    numerator = (2 * mu_product + c1) * (2 * sigma_product + c2)
    denominator = (mu_original_sq + mu_changed_sq + c1) * (
        sigma_original_sq + sigma_changed_sq + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def quality_metrics(original: Image.Image, changed: Image.Image) -> dict[str, float | int]:
    if original.size != changed.size:
        raise ValueError("images must have the same dimensions")

    original_array = _rgb_array(original)
    changed_array = _rgb_array(changed)
    difference = changed_array - original_array
    absolute = np.abs(difference)
    mse = float(np.mean(difference * difference))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / rmse)

    return {
        "psnr": round(psnr, 6),
        "ssim": round(_ssim(original_array, changed_array), 6),
        "mae": round(float(np.mean(absolute)), 6),
        "rmse": round(rmse, 6),
        "max_abs_diff": int(np.max(absolute)),
    }


def quality_gate(
    metrics: dict[str, float],
    min_psnr: float = 38.0,
    min_ssim: float = 0.98,
) -> bool:
    return metrics["psnr"] >= min_psnr and metrics["ssim"] >= min_ssim


def metric_distribution(rows: list[dict], field: str) -> dict[str, float]:
    if not rows:
        raise ValueError("metric rows must not be empty")
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "min": round(float(values.min()), 6),
        "p5": round(float(np.percentile(values, 5)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "mean": round(float(values.mean()), 6),
    }
