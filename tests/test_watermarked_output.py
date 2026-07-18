from pathlib import Path

from PIL import Image

from trace_app.imaging.output import (
    JPEG_MIN_QUALITY,
    WatermarkedOutput,
    encode_adaptive_jpeg,
    encode_jpeg,
    save_watermarked_output,
)


def _image() -> Image.Image:
    return Image.effect_noise((320, 240), 32).convert("RGB")


def test_adaptive_jpeg_selects_highest_quality_within_target() -> None:
    def sized_encoder(image: Image.Image, quality: int) -> bytes:
        return bytes(quality * 10)

    content, quality = encode_adaptive_jpeg(
        _image(),
        source_size=744,
        encoder=sized_encoder,
    )

    assert quality == 93
    assert len(content) == 930


def test_adaptive_jpeg_never_drops_below_quality_90() -> None:
    def sized_encoder(image: Image.Image, quality: int) -> bytes:
        return bytes(quality * 10)

    content, quality = encode_adaptive_jpeg(
        _image(),
        source_size=1,
        encoder=sized_encoder,
    )

    assert quality == JPEG_MIN_QUALITY == 90
    assert len(content) == 900


def test_encode_jpeg_produces_real_jpeg() -> None:
    content = encode_jpeg(_image(), 92)

    assert content.startswith(b"\xff\xd8")


def test_save_watermarked_output_returns_persisted_jpeg(tmp_path: Path) -> None:
    result = save_watermarked_output(
        _image(),
        tmp_path / "watermarked",
        jpeg_output=True,
        source_size=1,
    )

    assert isinstance(result, WatermarkedOutput)
    assert result.path.suffix == ".jpg"
    assert result.quality == 90
    assert result.path.exists()
    with Image.open(result.path) as loaded:
        assert loaded.format == "JPEG"
    assert result.image.mode == "RGB"
    assert result.image.getpixel((0, 0))


def test_save_watermarked_output_keeps_png_path_lossless(tmp_path: Path) -> None:
    source = _image()

    result = save_watermarked_output(
        source,
        tmp_path / "watermarked",
        jpeg_output=False,
        source_size=1,
    )

    assert result.path.suffix == ".png"
    assert result.quality is None
    with Image.open(result.path) as loaded:
        assert loaded.format == "PNG"
        assert loaded.convert("RGB").tobytes() == source.tobytes()
