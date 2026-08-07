import hashlib
import io
from dataclasses import FrozenInstanceError
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np
import pytest
from PIL import Image

import watermark_v4.sync as sync_module
from tests.commercial_quality_metrics import quality_metrics
from watermark_v4 import V4Config, bytes_to_bits, embed_codeword, extract_image_tiles
from watermark_v4.sync import (
    PilotPeakEvidence,
    SyncEstimate,
    _analysis_spectrum,
    _pilot_phases,
    _validate_dimensions,
    detect_pilot,
    embed_pilot,
    pilot_peak_evidence,
    pilot_signal,
)


def _independent_phases(codec: str) -> tuple[float, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{codec}:pilot-phase:{index}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        * (2.0 * np.pi / 2**64)
        for index in range(4)
    )


def _synthetic_rgb(kind: str, width: int = 384, height: int = 256) -> Image.Image:
    if kind == "neutral":
        pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    elif kind == "gradient":
        horizontal = np.linspace(0, 255, width, dtype=np.uint8)
        vertical = np.linspace(0, 255, height, dtype=np.uint8)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = horizontal
        pixels[..., 1] = vertical[:, None]
        pixels[..., 2] = 96
    else:
        pixels = np.random.default_rng(20260723).integers(
            0,
            256,
            size=(height, width, 3),
            dtype=np.uint8,
        )
    return Image.fromarray(pixels)


def test_pilot_phases_match_independent_sha256_vectors_and_are_cached() -> None:
    config = V4Config()
    phases = _pilot_phases(config.codec)

    assert phases is _pilot_phases(config.codec)
    assert phases == _independent_phases(config.codec)
    assert len(phases) == 4
    assert all(type(phase) is float and 0.0 <= phase < 2.0 * np.pi for phase in phases)


def test_pilot_signal_matches_independent_vectorized_formula() -> None:
    config = V4Config()
    height, width = 19, 23
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    expected = np.zeros((height, width), dtype=np.float64)
    for (frequency_x, frequency_y), phase in zip(
        config.pilot_frequency_vectors,
        _independent_phases(config.codec),
    ):
        expected += config.pilot_amplitude * np.sin(
            2.0 * np.pi * (frequency_x * x + frequency_y * y) + phase
        )

    signal = pilot_signal(height, width, config)

    assert signal.shape == (height, width)
    assert signal.dtype == np.dtype(np.float64)
    assert np.isfinite(signal).all()
    np.testing.assert_allclose(signal, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(signal, pilot_signal(height, width, config))


def test_pilot_signal_scales_linearly_with_configured_amplitude() -> None:
    low = pilot_signal(31, 47, V4Config(pilot_amplitude=0.5))
    high = pilot_signal(31, 47, V4Config(pilot_amplitude=1.0))

    np.testing.assert_allclose(high, 2.0 * low, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    ("height", "width"),
    (
        (0, 1),
        (1, 0),
        (-1, 1),
        (1, -1),
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, 1.0),
    ),
)
def test_pilot_signal_rejects_malformed_dimensions(height: object, width: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dimension"):
        pilot_signal(height, width, V4Config())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape",
    (
        (2000, 2001),
        (1, 4097),
        (10_000, 10_000),
    ),
)
def test_pilot_dimensions_have_no_business_pixel_limit(
    shape: tuple[int, int],
) -> None:
    _validate_dimensions(*shape)


@pytest.mark.parametrize("shape", ((2000, 2000), (1, 4096), (1499, 2233)))
def test_pilot_dimension_boundaries_and_current_images_are_allowed(
    shape: tuple[int, int],
) -> None:
    _validate_dimensions(*shape)


def test_chunked_pilot_signal_matches_full_formula_across_chunk_boundaries() -> None:
    config = V4Config()
    height, width = 601, 73
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    expected = np.zeros((height, width), dtype=np.float64)
    for (frequency_x, frequency_y), phase in zip(
        config.pilot_frequency_vectors,
        _independent_phases(config.codec),
    ):
        expected += config.pilot_amplitude * np.sin(
            2.0 * np.pi * (frequency_x * x + frequency_y * y) + phase
        )

    signal = pilot_signal(height, width, config)

    np.testing.assert_allclose(signal, expected, rtol=0.0, atol=1e-13)


def test_embed_pilot_processes_image_above_former_limit_in_row_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (2200, 1900), (96, 128, 160))
    conversion_rows = []
    original_cvt_color = sync_module.cv2.cvtColor

    def forbidden_full_signal(*args: object, **kwargs: object) -> object:
        raise AssertionError("embed_pilot must not allocate a full pilot signal")

    def tracked_cvt_color(values: np.ndarray, code: int) -> np.ndarray:
        conversion_rows.append(values.shape[0])
        return original_cvt_color(values, code)

    monkeypatch.setattr(sync_module, "pilot_signal", forbidden_full_signal)
    monkeypatch.setattr(sync_module.cv2, "cvtColor", tracked_cvt_color)

    embedded = embed_pilot(image, V4Config())

    assert embedded.size == image.size
    assert embedded.mode == "RGB"
    assert embedded.tobytes() != image.tobytes()
    assert conversion_rows
    assert max(conversion_rows) <= sync_module.PILOT_ROW_CHUNK


def test_pilot_sine_temporaries_never_exceed_row_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_shapes: list[tuple[int, ...]] = []
    original_sin = sync_module.np.sin

    def tracked_sin(values: np.ndarray) -> np.ndarray:
        input_shapes.append(values.shape)
        return original_sin(values)

    monkeypatch.setattr(sync_module.np, "sin", tracked_sin)

    signal = pilot_signal(700, 37, V4Config())

    assert signal.shape == (700, 37)
    assert len(input_shapes) == 4 * 3
    assert all(rows <= 256 and columns == 37 for rows, columns in input_shapes)


@pytest.mark.parametrize("config", (None, object()))
def test_pilot_signal_requires_exact_v4_config(config: object) -> None:
    with pytest.raises(TypeError, match="config"):
        pilot_signal(16, 16, config)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ("neutral", "gradient", "noise"))
def test_embed_pilot_is_deterministic_and_preserves_rgb_input(kind: str) -> None:
    image = _synthetic_rgb(kind)
    original = image.tobytes()

    first = embed_pilot(image, V4Config())
    second = embed_pilot(image, V4Config())

    assert type(first) is Image.Image
    assert first.mode == "RGB"
    assert first.size == image.size
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != original
    assert image.tobytes() == original


@pytest.mark.parametrize("level", (0, 255))
def test_embed_pilot_controls_clipping_at_luminance_extremes(level: int) -> None:
    image = Image.new("RGB", (257, 193), (level, level, level))

    embedded = embed_pilot(image, V4Config())
    pixels = np.asarray(embedded)

    assert pixels.dtype == np.dtype(np.uint8)
    assert np.min(pixels) >= 0
    assert np.max(pixels) <= 255


def test_embed_pilot_preserves_rgba_alpha_byte_for_byte() -> None:
    rgb = np.asarray(_synthetic_rgb("noise")).copy()
    alpha = np.random.default_rng(20260724).integers(
        0,
        256,
        size=rgb.shape[:2],
        dtype=np.uint8,
    )
    image = Image.fromarray(np.dstack((rgb, alpha)))
    original = image.tobytes()

    embedded = embed_pilot(image, V4Config())

    assert embedded.mode == "RGBA"
    assert embedded.size == image.size
    np.testing.assert_array_equal(np.asarray(embedded)[..., 3], alpha)
    assert image.tobytes() == original


def test_embed_pilot_preserves_full_resolution_above_analysis_limit() -> None:
    image = _synthetic_rgb("gradient", width=1101, height=129)
    original = np.asarray(image).copy()

    embedded = embed_pilot(image, V4Config())
    changed = np.asarray(embedded)

    assert embedded.size == (1101, 129)
    assert np.any(changed[:, 1024:] != original[:, 1024:])


@pytest.mark.parametrize("mode", ("L", "P", "CMYK"))
def test_embed_pilot_rejects_unsupported_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="RGB"):
        embed_pilot(Image.new(mode, (32, 32)), V4Config())


@pytest.mark.parametrize("image", (None, np.zeros((32, 32, 3), dtype=np.uint8)))
def test_embed_pilot_requires_exact_pil_image(image: object) -> None:
    with pytest.raises(TypeError, match="image"):
        embed_pilot(image, V4Config())  # type: ignore[arg-type]


@pytest.mark.parametrize("config", (None, object()))
def test_embed_pilot_requires_exact_v4_config(config: object) -> None:
    with pytest.raises(TypeError, match="config"):
        embed_pilot(_synthetic_rgb("neutral"), config)  # type: ignore[arg-type]


def test_embed_pilot_converts_color_exactly_once_each_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"forward": 0, "reverse": 0}
    original = sync_module.cv2.cvtColor

    def counted(array: np.ndarray, code: int) -> np.ndarray:
        if code == cv2.COLOR_RGB2YCrCb:
            counts["forward"] += 1
        if code == cv2.COLOR_YCrCb2RGB:
            counts["reverse"] += 1
        return original(array, code)

    monkeypatch.setattr(sync_module.cv2, "cvtColor", counted)

    embed_pilot(_synthetic_rgb("gradient"), V4Config())

    assert counts == {"forward": 1, "reverse": 1}


def _quality_images() -> tuple[tuple[str, Image.Image], ...]:
    with Image.open(Path("img/1.png")) as loaded:
        current = loaded.convert("RGB").resize((512, 344), Image.Resampling.LANCZOS)
    return (
        ("neutral", _synthetic_rgb("neutral")),
        ("gradient", _synthetic_rgb("gradient")),
        ("noise", _synthetic_rgb("noise")),
        ("current-1", current),
    )


@pytest.mark.parametrize(("name", "image"), _quality_images())
def test_pilot_only_quality_gate(
    name: str,
    image: Image.Image,
    record_property: object,
) -> None:
    embedded = embed_pilot(image, V4Config())
    metrics = quality_metrics(image, embedded)
    record_property(name, str(metrics))  # type: ignore[operator]

    assert float(metrics["psnr"]) >= 44.0
    assert float(metrics["ssim"]) >= 0.98


def test_analysis_spectrum_is_centered_finite_complex_and_resized() -> None:
    image = _synthetic_rgb("gradient", width=1500, height=900)

    spectrum, magnitude = _analysis_spectrum(image, V4Config())

    assert spectrum.shape == (614, 1024)
    assert magnitude.shape == spectrum.shape
    assert np.issubdtype(spectrum.dtype, np.complexfloating)
    assert magnitude.dtype == np.dtype(np.float64)
    assert np.isfinite(spectrum).all()
    assert np.isfinite(magnitude).all()


def test_peak_bins_account_for_resize_rotation_and_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (1500, 900), (128, 128, 128))
    analysis_height, analysis_width = 614, 1024
    magnitude = np.ones((analysis_height, analysis_width), dtype=np.float64)
    rotation = 12.0
    scale = 1.15
    radians = np.deg2rad(rotation)
    cosine, sine = np.cos(radians), np.sin(radians)
    expected_bins: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for frequency_x, frequency_y in V4Config().pilot_frequency_vectors:
        rotated_x = (cosine * frequency_x - sine * frequency_y) / scale
        rotated_y = (sine * frequency_x + cosine * frequency_y) / scale
        analysis_x = rotated_x * image.width / analysis_width
        analysis_y = rotated_y * image.height / analysis_height
        positive = (
            round(analysis_height // 2 + analysis_y * analysis_height),
            round(analysis_width // 2 + analysis_x * analysis_width),
        )
        negative = (
            round(analysis_height // 2 - analysis_y * analysis_height),
            round(analysis_width // 2 - analysis_x * analysis_width),
        )
        expected_bins.append((positive, negative))
        magnitude[positive] = 100.0
        magnitude[negative] = 100.0

    monkeypatch.setattr(
        sync_module,
        "_analysis_spectrum",
        lambda *args, **kwargs: (
            np.zeros_like(magnitude, dtype=np.complex128),
            magnitude,
        ),
    )

    evidence = pilot_peak_evidence(
        image,
        V4Config(),
        rotation_degrees=rotation,
        scale=scale,
    )

    assert [(item.positive_bin, item.negative_bin) for item in evidence] == expected_bins
    assert all(item.supported for item in evidence)


@pytest.mark.parametrize(("name", "image"), _quality_images())
def test_intact_embedded_pilot_supports_at_least_three_components(
    name: str,
    image: Image.Image,
    record_property: object,
) -> None:
    evidence = pilot_peak_evidence(embed_pilot(image, V4Config()), V4Config())
    record_property(  # type: ignore[operator]
        name,
        str([(item.positive_ratio, item.negative_ratio) for item in evidence]),
    )

    assert len(evidence) == 4
    assert sum(item.supported for item in evidence) >= 3


@pytest.mark.parametrize(("name", "image"), _quality_images())
def test_original_images_do_not_form_supported_pilot_constellation(
    name: str,
    image: Image.Image,
) -> None:
    evidence = pilot_peak_evidence(image, V4Config())

    assert sum(item.supported for item in evidence) < 3


@pytest.mark.parametrize(
    "image",
    (
        Image.new("RGB", (320, 240), (0, 0, 0)),
        _synthetic_rgb("noise", 320, 240),
    ),
)
def test_blank_and_noise_return_controlled_unsupported_evidence(
    image: Image.Image,
) -> None:
    evidence = pilot_peak_evidence(image, V4Config())

    assert len(evidence) == 4
    assert sum(item.supported for item in evidence) < 3
    assert all(np.isfinite(item.positive_ratio) for item in evidence)
    assert all(np.isfinite(item.negative_ratio) for item in evidence)


def test_peak_evidence_is_frozen_slotted_and_strictly_validated() -> None:
    item = PilotPeakEvidence(0, 3.0, 4.0, True, (10, 20), (30, 40))

    assert not hasattr(item, "__dict__")
    with pytest.raises(FrozenInstanceError):
        item.supported = False  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError), match="component"):
        PilotPeakEvidence(True, 3.0, 4.0, True, (10, 20), (30, 40))
    with pytest.raises((TypeError, ValueError), match="ratio"):
        PilotPeakEvidence(0, float("nan"), 4.0, True, (10, 20), (30, 40))
    with pytest.raises((TypeError, ValueError), match="supported"):
        PilotPeakEvidence(0, 3.0, 4.0, 1, (10, 20), (30, 40))  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError), match="bin"):
        PilotPeakEvidence(0, 3.0, 4.0, True, [10, 20], (30, 40))  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ("L", "P", "CMYK"))
def test_spectral_functions_reject_unsupported_modes(mode: str) -> None:
    image = Image.new(mode, (64, 64))
    with pytest.raises(ValueError, match="RGB"):
        _analysis_spectrum(image, V4Config())
    with pytest.raises(ValueError, match="RGB"):
        pilot_peak_evidence(image, V4Config())


@pytest.mark.parametrize(
    ("rotation", "scale"),
    (
        (float("nan"), 1.0),
        (181.0, 1.0),
        (-181.0, 1.0),
        (True, 1.0),
        (0.0, float("inf")),
        (0.0, 0.24),
        (0.0, 4.01),
        (0.0, False),
    ),
)
def test_peak_scoring_rejects_malformed_geometry(rotation: object, scale: object) -> None:
    with pytest.raises((TypeError, ValueError), match="rotation|scale"):
        pilot_peak_evidence(
            _synthetic_rgb("neutral", 64, 64),
            V4Config(),
            rotation_degrees=rotation,  # type: ignore[arg-type]
            scale=scale,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), True, "later"))
def test_spectral_functions_reject_malformed_deadlines(deadline: object) -> None:
    image = _synthetic_rgb("neutral", 64, 64)
    with pytest.raises((TypeError, ValueError), match="deadline"):
        _analysis_spectrum(image, V4Config(), deadline=deadline)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError), match="deadline"):
        pilot_peak_evidence(image, V4Config(), deadline=deadline)  # type: ignore[arg-type]


def test_spectral_functions_stop_on_expired_deadline() -> None:
    image = _synthetic_rgb("neutral", 64, 64)
    expired = monotonic() - 1.0

    with pytest.raises(TimeoutError, match="deadline"):
        _analysis_spectrum(image, V4Config(), deadline=expired)
    with pytest.raises(TimeoutError, match="deadline"):
        pilot_peak_evidence(image, V4Config(), deadline=expired)


def test_peak_scoring_computes_color_and_fft_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"color": 0, "fft": 0}
    original_color = sync_module.cv2.cvtColor
    original_fft = sync_module.np.fft.fft2

    def counted_color(array: np.ndarray, code: int) -> np.ndarray:
        counts["color"] += 1
        return original_color(array, code)

    def counted_fft(array: np.ndarray) -> np.ndarray:
        counts["fft"] += 1
        return original_fft(array)

    monkeypatch.setattr(sync_module.cv2, "cvtColor", counted_color)
    monkeypatch.setattr(sync_module.np.fft, "fft2", counted_fft)

    pilot_peak_evidence(_synthetic_rgb("gradient"), V4Config())

    assert counts == {"color": 1, "fft": 1}


def test_spectral_scoring_is_not_public_attribution_api() -> None:
    assert "PilotPeakEvidence" not in sync_module.__all__
    assert "pilot_peak_evidence" not in sync_module.__all__


@pytest.mark.parametrize(
    ("rotation_degrees", "scale"),
    (
        (0.0, 0.75),
        (0.0, 1.5),
        (6.0, 1.0),
        (-8.0, 1.0),
        (6.0, 1.25),
    ),
)
def test_detect_pilot_recovers_source_to_query_rotation_and_scale(
    rotation_degrees: float,
    scale: float,
) -> None:
    source = embed_pilot(_synthetic_rgb("gradient", 512, 384), V4Config())
    query = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.BICUBIC,
    )
    if rotation_degrees:
        query = query.rotate(
            rotation_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )

    estimate = detect_pilot(query, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.rotation_degrees == pytest.approx(rotation_degrees, abs=0.5)
    assert estimate.scale == pytest.approx(scale, abs=0.03)
    assert estimate.supported_peaks >= 3
    assert 0.0 < estimate.confidence <= 1.0
    assert 1 <= estimate.evaluated_hypotheses <= 500
    assert estimate.elapsed_seconds >= 0.0


def test_detect_pilot_recovers_combined_resize_rotation_crop_and_jpeg() -> None:
    source = embed_pilot(_synthetic_rgb("gradient", 640, 480), V4Config())
    query = source.resize((768, 576), Image.Resampling.BICUBIC)
    query = query.rotate(5.0, resample=Image.Resampling.BICUBIC, expand=True)
    query = query.crop((45, 35, query.width - 38, query.height - 31))
    encoded = io.BytesIO()
    query.save(encoded, format="JPEG", quality=70)
    encoded.seek(0)
    with Image.open(encoded) as loaded:
        attacked = loaded.convert("RGB")

    estimate = detect_pilot(attacked, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.rotation_degrees == pytest.approx(5.0, abs=0.6)
    assert estimate.scale == pytest.approx(1.2, abs=0.04)
    assert estimate.supported_peaks >= 3


@pytest.mark.parametrize(
    "image",
    (
        _synthetic_rgb("gradient", 384, 256),
        _synthetic_rgb("noise", 384, 256),
        Image.new("RGB", (384, 256), (128, 128, 128)),
    ),
)
def test_detect_pilot_rejects_unwatermarked_inputs(image: Image.Image) -> None:
    assert detect_pilot(image, V4Config()) is None


def test_detect_pilot_rejects_two_separated_equal_pilot_constellations() -> None:
    height, width = 384, 512
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    signal = pilot_signal(height, width, V4Config())
    for (frequency_x, frequency_y), phase in zip(
        V4Config().pilot_frequency_vectors,
        _independent_phases(V4Config().codec),
    ):
        signal += V4Config().pilot_amplitude * np.sin(
            2.0
            * np.pi
            * (frequency_x / 1.5 * x + frequency_y / 1.5 * y)
            + phase
        )
    pixels = np.clip(np.rint(128.0 + signal), 0, 255).astype(np.uint8)
    image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))

    assert detect_pilot(image, V4Config()) is None


def test_detect_pilot_honors_expired_deadline() -> None:
    with pytest.raises(TimeoutError, match="deadline"):
        detect_pilot(
            embed_pilot(_synthetic_rgb("gradient"), V4Config()),
            V4Config(),
            deadline=monotonic() - 1.0,
        )


def test_sync_estimate_is_frozen_slotted_and_strictly_validated() -> None:
    estimate = SyncEstimate(0.0, 1.0, 0.5, 4, 10, 0.1)

    assert not hasattr(estimate, "__dict__")
    with pytest.raises(FrozenInstanceError):
        estimate.scale = 1.1  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError), match="rotation"):
        SyncEstimate(float("nan"), 1.0, 0.5, 4, 10, 0.1)
    with pytest.raises((TypeError, ValueError), match="scale"):
        SyncEstimate(0.0, 2.1, 0.5, 4, 10, 0.1)
    with pytest.raises((TypeError, ValueError), match="confidence"):
        SyncEstimate(0.0, 1.0, 1.1, 4, 10, 0.1)
    with pytest.raises((TypeError, ValueError), match="supported"):
        SyncEstimate(0.0, 1.0, 0.5, True, 10, 0.1)
    with pytest.raises((TypeError, ValueError), match="hypotheses"):
        SyncEstimate(0.0, 1.0, 0.5, 4, 0, 0.1)


@pytest.mark.parametrize(
    ("crop_x", "crop_y"),
    ((0, 0), (37, 59), (165, 143), (128, 128)),
)
def test_detect_pilot_recovers_crop_origin_modulo_tile_size(
    crop_x: int,
    crop_y: int,
) -> None:
    source = embed_pilot(_synthetic_rgb("gradient", 768, 640), V4Config())
    query = source.crop((crop_x, crop_y, crop_x + 384, crop_y + 256))

    estimate = detect_pilot(query, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.offset_x == crop_x % V4Config().tile_size
    assert estimate.offset_y == crop_y % V4Config().tile_size


def test_detect_pilot_recovers_real_image_crop_origin() -> None:
    with Image.open(Path("img/1.png")) as loaded:
        source = loaded.convert("RGB").resize((768, 640), Image.Resampling.LANCZOS)
    marked = embed_pilot(source, V4Config())
    query = marked.crop((37, 59, 421, 315))

    estimate = detect_pilot(query, V4Config())

    assert type(estimate) is SyncEstimate
    assert (estimate.offset_x, estimate.offset_y) == (37, 59)


def test_detect_pilot_recovers_crop_origin_after_half_scale() -> None:
    source = embed_pilot(_synthetic_rgb("gradient", 768, 640), V4Config())
    scaled = source.resize((384, 320), Image.Resampling.BICUBIC)
    query = scaled.crop((108, 61, 300, 189))

    estimate = detect_pilot(query, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.scale == pytest.approx(0.5, abs=0.01)
    assert estimate.offset_x == 216 % V4Config().tile_size
    assert estimate.offset_y == 122 % V4Config().tile_size


def test_detect_pilot_rejects_inconsistent_component_phases_for_offset() -> None:
    height, width = 384, 512
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    signal = np.zeros((height, width), dtype=np.float64)
    corruptions = (0.3, 1.1, -0.7, 2.2)
    for (frequency_x, frequency_y), phase, corruption in zip(
        V4Config().pilot_frequency_vectors,
        _independent_phases(V4Config().codec),
        corruptions,
    ):
        signal += V4Config().pilot_amplitude * np.sin(
            2.0 * np.pi * (frequency_x * x + frequency_y * y)
            + phase
            + corruption
        )
    pixels = np.clip(np.rint(128.0 + signal), 0, 255).astype(np.uint8)
    image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))

    estimate = detect_pilot(image, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.offset_x is None
    assert estimate.offset_y is None


@pytest.mark.parametrize(("name", "image"), _quality_images())
def test_combined_dct_and_pilot_quality_and_codeword_recovery(
    name: str,
    image: Image.Image,
    record_property: object,
) -> None:
    codeword = bytes.fromhex("1020304050607080")
    combined = embed_codeword(embed_pilot(image, V4Config()), codeword, V4Config())
    metrics = quality_metrics(image, combined)
    records = extract_image_tiles(combined, V4Config())
    score_matrix = np.asarray(
        [record.logical_scores for record in records],
        dtype=np.float64,
    )
    aggregate_bits = tuple(
        int(score > 0.0) for score in np.mean(score_matrix, axis=0)
    )
    record_property(name, str(metrics))  # type: ignore[operator]

    if name != "noise":
        assert float(metrics["psnr"]) >= 38.0
    assert float(metrics["ssim"]) >= 0.95
    assert aggregate_bits == bytes_to_bits(codeword)


def test_combined_dct_and_pilot_remains_synchronizable() -> None:
    images = dict(_quality_images())
    combined = embed_codeword(
        embed_pilot(images["gradient"], V4Config()),
        bytes.fromhex("1020304050607080"),
    )
    query = combined.resize(
        (round(combined.width * 1.2), round(combined.height * 1.2)),
        Image.Resampling.BICUBIC,
    )
    query = query.rotate(5.0, resample=Image.Resampling.BICUBIC, expand=True)

    estimate = detect_pilot(query, V4Config())

    assert type(estimate) is SyncEstimate
    assert estimate.rotation_degrees == pytest.approx(5.0, abs=0.6)
    assert estimate.scale == pytest.approx(1.2, abs=0.04)


def test_low_confidence_combined_current_image_defers_to_geometry_fallback() -> None:
    images = dict(_quality_images())
    combined = embed_codeword(
        embed_pilot(images["current-1"], V4Config()),
        bytes.fromhex("1020304050607080"),
    )
    query = combined.resize(
        (round(combined.width * 1.2), round(combined.height * 1.2)),
        Image.Resampling.BICUBIC,
    ).rotate(5.0, resample=Image.Resampling.BICUBIC, expand=True)

    assert detect_pilot(query, V4Config()) is None


def test_1024_synchronization_has_bounded_regression_runtime(
    record_property: object,
) -> None:
    image = _synthetic_rgb("gradient", 1024, 1024)
    combined = embed_codeword(
        embed_pilot(image, V4Config()),
        bytes.fromhex("1020304050607080"),
    )

    started = perf_counter()
    estimate = detect_pilot(combined, V4Config())
    elapsed = perf_counter() - started
    record_property(  # type: ignore[operator]
        "1024-synchronization-regression-seconds",
        f"{elapsed:.6f}",
    )

    assert type(estimate) is SyncEstimate
    assert estimate.rotation_degrees == pytest.approx(0.0, abs=0.5)
    assert estimate.scale == pytest.approx(1.0, abs=0.03)
    assert elapsed < 2.0  # Regression ceiling, not the commercial SLA.


def test_stable_synchronization_api_is_exported_from_package() -> None:
    import watermark_v4

    assert watermark_v4.SyncEstimate is SyncEstimate
    assert watermark_v4.detect_pilot is detect_pilot
    assert watermark_v4.embed_pilot is embed_pilot
