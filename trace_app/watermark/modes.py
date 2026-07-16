from __future__ import annotations

from collections.abc import Callable


def mode_label(mode: str) -> str:
    labels = {
        "lsb": "仅空间域",
        "dct": "DCT + 空间域",
        "dwt": "DWT + 空间域",
        "fft": "FFT + 空间域",
        "hybrid": "全部算法",
    }
    return labels.get(mode, "DCT + 空间域")


def normalize_mode(raw: str) -> str:
    text = (raw or "").lower()
    if "lsb" in text or "空间" in raw or "最快" in raw:
        return "lsb"
    if "全部" in raw or "hybrid" in text or "最强" in raw:
        return "hybrid"
    if "dwt" in text:
        return "dwt"
    if "fft" in text:
        return "fft"
    return "dct"


def fidelity_to_strength(
    value: str,
    *,
    clamp: Callable[[str | float | None, float, float, float], float],
) -> float:
    fidelity = clamp(value, 0.75, 0.0, 1.0)
    return 1.0 - fidelity * 0.72


def robust_strength_to_scale(
    value: str | float | None,
    default_value: str | float | None,
    *,
    clamp: Callable[[str | float | None, float, float, float], float],
) -> float:
    default = clamp(default_value, 1.0, 0.0, 2.0)
    return clamp(value, default, 0.0, 2.0)
