from __future__ import annotations

import os
import re
from datetime import datetime


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def masked_url(url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:****@", url)


def parse_bool(raw: str | bool | None) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").lower() in {"1", "true", "yes", "on", "启用"}


def env_bool(
    name: str, default: str = "false", legacy_name: str | None = None
) -> bool:
    value = os.getenv(name)
    if value is None and legacy_name:
        value = os.getenv(legacy_name)
    return parse_bool(default if value is None else value)


def clamp_float(
    value: str | float | None, default: float, low: float, high: float
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))
