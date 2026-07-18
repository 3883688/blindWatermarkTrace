from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, JpegImagePlugin


JPEG_MIN_QUALITY = 90
JPEG_MAX_QUALITY = 95
JPEG_TARGET_RATIO = 1.25
JPEG_SUBSAMPLING_VALUES = (0, 1, 2)
JpegEncoder = Callable[[Image.Image, int], bytes]


@dataclass(frozen=True, slots=True)
class WatermarkedOutput:
    path: Path
    image: Image.Image
    quality: int | None


def _normalize_jpeg_subsampling(value: object) -> int:
    return (
        value
        if type(value) is int and value in JPEG_SUBSAMPLING_VALUES
        else 0
    )


def jpeg_subsampling(image: Image.Image) -> int:
    return _normalize_jpeg_subsampling(JpegImagePlugin.get_sampling(image))


def encode_jpeg(
    image: Image.Image,
    quality: int,
    *,
    subsampling: int = 0,
) -> bytes:
    buffer = BytesIO()
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb_image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=_normalize_jpeg_subsampling(subsampling),
    )
    return buffer.getvalue()


def encode_adaptive_jpeg(
    image: Image.Image,
    source_size: int,
    *,
    subsampling: int = 0,
    encoder: JpegEncoder | None = None,
) -> tuple[bytes, int]:
    if encoder is None:
        effective_image = image if image.mode == "RGB" else image.convert("RGB")

        def default_encoder(image: Image.Image, quality: int) -> bytes:
            return encode_jpeg(image, quality, subsampling=subsampling)

        effective_encoder = default_encoder
    else:
        effective_image = image
        effective_encoder = encoder

    target_size = max(1, int(source_size * JPEG_TARGET_RATIO))
    minimum_content: bytes | None = None
    for quality in range(JPEG_MAX_QUALITY, JPEG_MIN_QUALITY - 1, -1):
        content = effective_encoder(effective_image, quality)
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
    jpeg_subsampling: int = 0,
) -> WatermarkedOutput:
    quality = None
    if jpeg_output:
        path = output_base.with_suffix(".jpg")
        content, quality = encode_adaptive_jpeg(
            image,
            source_size,
            subsampling=jpeg_subsampling,
        )
        path.write_bytes(content)
    else:
        path = output_base.with_suffix(".png")
        image.save(path, format="PNG")
    with Image.open(path) as loaded:
        loaded.load()
        persisted = loaded.copy()
    return WatermarkedOutput(path=path, image=persisted, quality=quality)
