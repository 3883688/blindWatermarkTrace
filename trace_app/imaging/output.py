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
