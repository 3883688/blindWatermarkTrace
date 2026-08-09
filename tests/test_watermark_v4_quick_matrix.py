from io import BytesIO
from time import perf_counter

from PIL import Image

from tests.test_watermark_v4_features import _feature_image
from watermark_v4 import V4Config, embed_codeword
from watermark_v4.detector import V4Candidate, V4Detection, detect_v4
from watermark_v4.features import extract_feature_index
from watermark_v4.payload import encode_codeword
from watermark_v4.sync import embed_pilot


def test_v4_quick_positive_attack_matrix() -> None:
    config = V4Config()
    tag = bytes.fromhex("1234abcd5678ef90")
    source = _feature_image((1280, 960), seed=707)
    marked = embed_codeword(
        embed_pilot(source, config),
        encode_codeword(tag),
        config,
    )
    candidate = V4Candidate(
        record_id="quick-matrix-record",
        trace_id="TR-V4-QUICK-MATRIX",
        auth_tag=tag,
        feature_index=extract_feature_index(marked),
    )

    def center_crop(image: Image.Image, ratio: float) -> Image.Image:
        width = round(image.width * ratio)
        height = round(image.height * ratio)
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))

    jpeg_buffer = BytesIO()
    marked.save(jpeg_buffer, format="JPEG", quality=50)
    jpeg_buffer.seek(0)
    with Image.open(jpeg_buffer) as loaded:
        jpeg_50 = loaded.convert("RGB")
    combined = marked.resize((1600, 1200), Image.Resampling.BICUBIC).rotate(
        5.0,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    combined = center_crop(combined, 0.5)
    cases = {
        "intact": marked,
        "scale_0.5": marked.resize((640, 480), Image.Resampling.BICUBIC),
        "scale_0.75": marked.resize((960, 720), Image.Resampling.BICUBIC),
        "scale_1.5": marked.resize((1920, 1440), Image.Resampling.BICUBIC),
        "scale_2.0": marked.resize((2560, 1920), Image.Resampling.BICUBIC),
        "crop_0.3": center_crop(marked, 0.3),
        "crop_0.5": center_crop(marked, 0.5),
        "crop_0.8": center_crop(marked, 0.8),
        "rotate_8": marked.rotate(
            8.0,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        ),
        "jpeg_50": jpeg_50,
        "combined": combined,
    }

    elapsed = {}
    for name, query in cases.items():
        started = perf_counter()
        result = detect_v4(query, (candidate,), config)
        elapsed[name] = perf_counter() - started

        assert type(result) is V4Detection, name
        assert result.trace_id == candidate.trace_id, name
        assert result.bit_errors == 0, name
        assert result.tile_count >= config.minimum_tiles, name
        assert result.phase_count >= config.minimum_phases, name
        assert elapsed[name] < 2.0, name

    assert len(elapsed) == 11

