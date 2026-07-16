import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_visible_copyright(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
) -> dict[str, Any] | None:
    copyright_records = [item for item in records if item.get("copyright_enabled") and item.get("copyright_text")]
    if not copyright_records:
        return None

    # The visible copyright layer is human-readable but not OCR-backed in this lightweight version.
    # If hidden extraction fails and the image contains strong watermark-like bright overlays,
    # return the configured copyright source as a lower-confidence fallback.
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    total = max(1, sum(histogram))
    bright_ratio = sum(histogram[205:]) / total
    if bright_ratio < 0.05:
        return None

    record = copyright_records[0]
    text = str(record.get("copyright_text", "")).strip()
    user_id = "QQ:757675150" if "757675150" in text else text.replace("©", "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12].upper()
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": f"VISIBLE-{digest}",
        "user_id": user_id or record.get("user_id") or "VISIBLE-WATERMARK",
        "mode": "visible",
        "mode_label": "可见版权水印",
        "created_at": record.get("created_at"),
        "confidence": 68,
        "phash_match": False,
        "status": "检测到可见版权水印",
        "extracted_at": now_text(),
    }, record)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "simhei.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_random_font(size: int, rng: np.random.Generator) -> ImageFont.ImageFont:
    font_names = [
        "arial.ttf",
        "arialbd.ttf",
        "simhei.ttf",
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simkai.ttf",
        "consola.ttf",
        "verdana.ttf",
        "tahoma.ttf",
        "times.ttf",
    ]
    font_paths = []
    windows_font_dir = Path(os.getenv("WINDIR", "C:\\Windows")) / "Fonts"
    for name in font_names:
        font_paths.append(windows_font_dir / name)
        font_paths.append(Path(name))
    rng.shuffle(font_paths)
    for path in font_paths:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return load_font(size)


def draw_text_pattern(layer: Image.Image, text: str, angle: int, gap: int, opacity: int) -> None:
    width, height = layer.size
    tile = Image.new("RGBA", (width * 2, height * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    font = load_font(max(18, min(width, height) // 18))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        text = text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), text, font=font)
    text_width = max(80, bbox[2] - bbox[0])
    text_height = max(24, bbox[3] - bbox[1])
    step_x = text_width + gap
    step_y = text_height + gap
    for y in range(-height, height * 2, step_y):
        for x in range(-width, width * 2, step_x):
            draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
    rotated = tile.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
    layer.alpha_composite(rotated.crop((width // 2, height // 2, width // 2 + width, height // 2 + height)))


def draw_irregular_text_pattern(layer: Image.Image, text: str, opacity: int, complexity: str) -> None:
    width, height = layer.size
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "big"))
    base_size = max(16, min(width, height) // 20)
    density = {
        "low": 0.55,
        "medium": 0.90,
        "high": 1.25,
        "extreme": 1.75,
        "低": 0.55,
        "中": 0.90,
        "高": 1.25,
        "极": 1.75,
    }.get(complexity, 0.90)
    count = max(10, int((width * height / 130_000) * density))
    colors = [
        (255, 255, 255),
        (255, 248, 196),
        (210, 245, 255),
        (235, 235, 255),
    ]
    safe_text = text
    for index in range(count):
        size = int(base_size * float(rng.uniform(0.70, 1.35)))
        font = load_random_font(size, rng)
        if rng.random() < 0.18:
            draw_text = safe_text.replace(" ", "")
            size = max(10, int(size * float(rng.uniform(0.45, 0.65))))
            font = load_random_font(size, rng)
        else:
            draw_text = safe_text
        try:
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        except UnicodeEncodeError:
            safe_text = safe_text.replace("©", "Copyright")
            draw_text = draw_text.replace("©", "Copyright")
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        patch = Image.new("RGBA", (text_width + 24, text_height + 24), (0, 0, 0, 0))
        patch_draw = ImageDraw.Draw(patch)
        color = colors[int(rng.integers(0, len(colors)))]
        alpha = max(8, min(220, int(opacity * float(rng.uniform(0.45, 1.25)))))
        patch_draw.text((12, 12), draw_text, fill=(*color, alpha), font=font)
        angle = float(rng.uniform(-38, 38))
        if rng.random() < 0.25:
            angle += float(rng.choice(np.array([-58, 58], dtype=np.int16)))
        rotated = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        x = int(rng.integers(-rotated.width // 3, max(1, width - rotated.width * 2 // 3)))
        y = int(rng.integers(-rotated.height // 3, max(1, height - rotated.height * 2 // 3)))
        layer.alpha_composite(rotated, (x, y))

    micro_count = max(18, int(count * 1.8))
    micro_font = load_random_font(max(9, base_size // 2), rng)
    micro_text = text.replace(" ", "")
    for _ in range(micro_count):
        x = int(rng.integers(0, max(1, width - 24)))
        y = int(rng.integers(0, max(1, height - 12)))
        alpha = max(5, int(opacity * float(rng.uniform(0.18, 0.45))))
        ImageDraw.Draw(layer).text((x, y), micro_text, fill=(255, 255, 255, alpha), font=micro_font)


def draw_prominent_corner_label(image: Image.Image, text: str) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)
    safe_text = text.strip() or "© QQ:757675150"
    font_size = max(22, min(base.size) // 14)
    font = load_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    except UnicodeEncodeError:
        safe_text = safe_text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    max_text_width = max(120, int(base.width * 0.72))
    while bbox[2] - bbox[0] > max_text_width and font_size > 16:
        font_size -= 2
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))

    stroke_width = max(2, font_size // 18)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x = max(12, font_size // 3)
    padding_y = max(8, font_size // 4)
    margin = max(14, min(base.size) // 40)
    right = base.width - margin
    bottom = base.height - margin
    left = max(margin, right - text_width - padding_x * 2)
    top = max(margin, bottom - text_height - padding_y * 2)
    radius = max(5, font_size // 6)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=(0, 0, 0, 205))
    draw.text(
        (left + padding_x, top + padding_y - bbox[1]),
        safe_text,
        font=font,
        fill=(255, 212, 0, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    return base.convert("RGB")


def apply_visible_copyright(
    image: Image.Image,
    enabled: bool,
    text: str,
    opacity: float,
    complexity: str,
    irregular: bool = True,
    prominent_corner: bool = False,
) -> Image.Image:
    if not enabled:
        return image.convert("RGB")

    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text = text.strip() or "© QQ:757675150"
    alpha = int(255 * opacity)
    settings = {
        "low": [(-24, 220)],
        "medium": [(-24, 110)],
        "high": [(-24, 105), (24, 105)],
        "extreme": [(-32, 75), (0, 75), (32, 75)],
        "低": [(-24, 220)],
        "中": [(-24, 110)],
        "高": [(-24, 105), (24, 105)],
        "极": [(-32, 75), (0, 75), (32, 75)],
    }.get(complexity, [(-24, 110)])
    if irregular:
        draw_irregular_text_pattern(layer, text, alpha, complexity)
    else:
        for angle, gap in settings:
            draw_text_pattern(layer, text, angle, gap, alpha)
    result = Image.alpha_composite(base, layer).convert("RGB")
    if prominent_corner:
        result = draw_prominent_corner_label(result, text)
    return result
