import hashlib
from dataclasses import dataclass
from functools import lru_cache
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from .config import V4Config


PILOT_COMPONENT_COUNT = 4
MAX_OFFSET_ANALYSIS_PIXELS = 4_000_000
MAX_OFFSET_ANALYSIS_SIDE = 4096
PILOT_ROW_CHUNK = 256
TWO_PI = 2.0 * np.pi
PEAK_SUPPORT_RATIO = 2.5
COARSE_ROTATIONS = tuple(float(value) for value in range(-12, 13, 2))
COARSE_SCALES = (
    0.5,
    0.65,
    0.75,
    0.8,
    0.95,
    1.0,
    1.1,
    1.25,
    1.4,
    1.5,
    1.55,
    1.7,
    1.85,
    2.0,
)
REFINE_ROTATION_STEP = 0.5
REFINE_SCALE_STEP = 0.02
AMBIGUOUS_SCORE_MARGIN = 0.05
MIN_SYNC_CONFIDENCE = 0.02


@dataclass(frozen=True, slots=True)
class PilotPeakEvidence:
    component_index: int
    positive_ratio: float
    negative_ratio: float
    supported: bool
    positive_bin: tuple[int, int]
    negative_bin: tuple[int, int]

    def __post_init__(self) -> None:
        if type(self.component_index) is not int or self.component_index not in range(4):
            raise ValueError("component index must be an integer from 0 through 3")
        for ratio in (self.positive_ratio, self.negative_ratio):
            if type(ratio) is not float or not np.isfinite(ratio) or ratio < 0.0:
                raise ValueError("ratio values must be finite nonnegative floats")
        if type(self.supported) is not bool:
            raise TypeError("supported must be a boolean")
        for bin_value in (self.positive_bin, self.negative_bin):
            if (
                type(bin_value) is not tuple
                or len(bin_value) != 2
                or any(type(value) is not int for value in bin_value)
            ):
                raise ValueError("bin values must be integer pairs")


@dataclass(frozen=True, slots=True)
class SyncEstimate:
    rotation_degrees: float
    scale: float
    confidence: float
    supported_peaks: int
    evaluated_hypotheses: int
    elapsed_seconds: float
    offset_x: int | None = None
    offset_y: int | None = None

    def __post_init__(self) -> None:
        real_fields = (
            ("rotation", self.rotation_degrees),
            ("scale", self.scale),
            ("confidence", self.confidence),
            ("elapsed", self.elapsed_seconds),
        )
        for name, value in real_fields:
            if type(value) is not float or not np.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
        if not -12.0 <= self.rotation_degrees <= 12.0:
            raise ValueError("rotation must be between -12 and 12 degrees")
        if not 0.5 <= self.scale <= 2.0:
            raise ValueError("scale must be between 0.5 and 2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed must be nonnegative")
        if type(self.supported_peaks) is not int or not 0 <= self.supported_peaks <= 4:
            raise ValueError("supported peaks must be an integer from 0 through 4")
        if type(self.evaluated_hypotheses) is not int or self.evaluated_hypotheses <= 0:
            raise ValueError("evaluated hypotheses must be a positive integer")
        for name, value in (("offset_x", self.offset_x), ("offset_y", self.offset_y)):
            if value is not None and (type(value) is not int or not 0 <= value < 128):
                raise ValueError(f"{name} must be None or an integer from 0 through 127")


@lru_cache(maxsize=1)
def _pilot_phases(codec: str) -> tuple[float, ...]:
    if type(codec) is not str or not codec:
        raise TypeError("codec must be a nonempty string")
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{codec}:pilot-phase:{index}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        * (TWO_PI / 2**64)
        for index in range(PILOT_COMPONENT_COUNT)
    )


def pilot_signal(height: int, width: int, config: V4Config) -> np.ndarray:
    _validate_dimensions(height, width)
    _validate_config(config)

    x = np.arange(width, dtype=np.float64)[None, :]
    signal = np.zeros((height, width), dtype=np.float64)
    phases = _pilot_phases(config.codec)
    for row_start in range(0, height, PILOT_ROW_CHUNK):
        row_stop = min(row_start + PILOT_ROW_CHUNK, height)
        signal[row_start:row_stop] = _pilot_signal_rows(
            row_start,
            row_stop,
            x,
            phases,
            config,
        )
    if not np.isfinite(signal).all():
        raise ValueError("pilot signal must contain only finite values")
    return signal


def embed_pilot(image: Image.Image, config: V4Config) -> Image.Image:
    _validate_image(image)
    _validate_config(config)

    source = np.asarray(image)
    output = source.copy()
    x = np.arange(image.width, dtype=np.float64)[None, :]
    phases = _pilot_phases(config.codec)
    for row_start in range(0, image.height, PILOT_ROW_CHUNK):
        row_stop = min(row_start + PILOT_ROW_CHUNK, image.height)
        rgb = source[row_start:row_stop, :, :3]
        ycrcb = cv2.cvtColor(
            rgb.astype(np.float32),
            cv2.COLOR_RGB2YCrCb,
        )
        signal = _pilot_signal_rows(
            row_start,
            row_stop,
            x,
            phases,
            config,
        )
        ycrcb[..., 0] = ycrcb[..., 0] + signal
        converted_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        output[row_start:row_stop, :, :3] = np.clip(
            np.rint(converted_rgb),
            0,
            255,
        ).astype(np.uint8)

    return Image.fromarray(output)


def _pilot_signal_rows(
    row_start: int,
    row_stop: int,
    x: np.ndarray,
    phases: tuple[float, ...],
    config: V4Config,
) -> np.ndarray:
    y = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
    chunk = np.zeros((row_stop - row_start, x.shape[1]), dtype=np.float64)
    for (frequency_x, frequency_y), phase in zip(
        config.pilot_frequency_vectors,
        phases,
    ):
        chunk += config.pilot_amplitude * np.sin(
            TWO_PI * (frequency_x * x + frequency_y * y) + phase
        )
    return chunk


def _analysis_spectrum(
    image: Image.Image,
    config: V4Config,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_image(image)
    _validate_config(config)
    _validate_deadline(deadline)
    _check_deadline(deadline)

    rgb = np.asarray(image)[..., :3]
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _check_deadline(deadline)
    if max(image.size) > config.analysis_max_side:
        resize_scale = config.analysis_max_side / max(image.size)
        analysis_width = max(1, round(image.width * resize_scale))
        analysis_height = max(1, round(image.height * resize_scale))
        luminance = cv2.resize(
            luminance,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
    values = luminance.astype(np.float64)
    values -= np.mean(values)
    window_y = np.hanning(values.shape[0])[:, None]
    window_x = np.hanning(values.shape[1])[None, :]
    windowed = values * window_y * window_x
    _check_deadline(deadline)
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(spectrum).astype(np.float64, copy=False)
    if not np.isfinite(spectrum).all() or not np.isfinite(magnitude).all():
        raise ValueError("analysis spectrum must contain only finite values")
    _check_deadline(deadline)
    return spectrum, magnitude


def pilot_peak_evidence(
    image: Image.Image,
    config: V4Config,
    rotation_degrees: float = 0.0,
    scale: float = 1.0,
    deadline: float | None = None,
) -> tuple[PilotPeakEvidence, ...]:
    _validate_image(image)
    _validate_config(config)
    rotation = _validated_rotation(rotation_degrees)
    scale_value = _validated_scale(scale)
    _validate_deadline(deadline)
    _check_deadline(deadline)
    spectrum, magnitude = _analysis_spectrum(image, config, deadline)

    analysis_height, analysis_width = magnitude.shape
    center_y = analysis_height // 2
    center_x = analysis_width // 2
    radians = np.deg2rad(rotation)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    evidence = []
    for component_index, (frequency_x, frequency_y) in enumerate(
        config.pilot_frequency_vectors
    ):
        _check_deadline(deadline)
        rotated_x = (cosine * frequency_x - sine * frequency_y) / scale_value
        rotated_y = (sine * frequency_x + cosine * frequency_y) / scale_value
        analysis_x = rotated_x * image.width / analysis_width
        analysis_y = rotated_y * image.height / analysis_height
        predicted_positive = (
            round(center_y + analysis_y * analysis_height),
            round(center_x + analysis_x * analysis_width),
        )
        predicted_negative = (
            round(center_y - analysis_y * analysis_height),
            round(center_x - analysis_x * analysis_width),
        )
        if abs(analysis_x) >= 0.5 or abs(analysis_y) >= 0.5:
            positive_ratio = 0.0
            negative_ratio = 0.0
            positive_bin = predicted_positive
            negative_bin = predicted_negative
        else:
            positive_ratio, positive_bin = _peak_ratio(
                magnitude, *predicted_positive
            )
            negative_ratio, negative_bin = _peak_ratio(
                magnitude, *predicted_negative
            )
            if magnitude.shape == (image.height, image.width):
                phase_ratio = _phase_aligned_ratio(
                    spectrum,
                    predicted_positive[0],
                    predicted_positive[1],
                    frequency_x,
                    frequency_y,
                    phases=_pilot_phases(config.codec),
                    component_index=component_index,
                )
                positive_ratio = max(positive_ratio, phase_ratio)
                negative_ratio = max(negative_ratio, phase_ratio)
        evidence.append(
            PilotPeakEvidence(
                component_index=component_index,
                positive_ratio=float(positive_ratio),
                negative_ratio=float(negative_ratio),
                supported=(
                    positive_ratio >= PEAK_SUPPORT_RATIO
                    and negative_ratio >= PEAK_SUPPORT_RATIO
                ),
                positive_bin=positive_bin,
                negative_bin=negative_bin,
            )
        )
    return tuple(evidence)


@dataclass(frozen=True, slots=True)
class _SyncHypothesis:
    spectral_rotation: float
    scale: float
    supported_peaks: int
    score: float
    ratios: tuple[float, ...]
    localization_error: float


def detect_pilot(
    image: Image.Image,
    config: V4Config,
    deadline: float | None = None,
) -> SyncEstimate | None:
    _validate_image(image)
    _validate_config(config)
    _validate_deadline(deadline)
    started = monotonic()
    _check_deadline(deadline)
    spectrum, magnitude = _analysis_spectrum(image, config, deadline)

    coarse_hypotheses: list[_SyncHypothesis] = []
    for rotation in COARSE_ROTATIONS:
        for scale in COARSE_SCALES:
            _check_deadline(deadline)
            coarse_hypotheses.append(
                _score_sync_hypothesis(
                    spectrum,
                    magnitude,
                    image.size,
                    config,
                    rotation,
                    scale,
                    search_radius=8,
                )
            )
    ranked_coarse = sorted(
        coarse_hypotheses,
        key=_hypothesis_sort_key,
        reverse=True,
    )
    coarse_modes = [ranked_coarse[0]]
    for item in ranked_coarse[1:]:
        if (
            abs(item.spectral_rotation - coarse_modes[0].spectral_rotation) >= 3.0
            or abs(item.scale - coarse_modes[0].scale) >= 0.10
        ):
            coarse_modes.append(item)
            break

    refined_hypotheses: list[_SyncHypothesis] = []
    seen: set[tuple[float, float]] = set()
    for coarse_mode in coarse_modes:
        refine_rotations = np.arange(
            max(-12.0, coarse_mode.spectral_rotation - 2.0),
            min(12.0, coarse_mode.spectral_rotation + 2.0) + 1e-9,
            REFINE_ROTATION_STEP,
        )
        refine_scales = np.arange(
            max(0.5, coarse_mode.scale - 0.1),
            min(2.0, coarse_mode.scale + 0.1) + 1e-9,
            REFINE_SCALE_STEP,
        )
        for rotation in refine_rotations:
            for scale in refine_scales:
                key = (round(float(rotation), 6), round(float(scale), 6))
                if key in seen:
                    continue
                _check_deadline(deadline)
                seen.add(key)
                refined_hypotheses.append(
                    _score_sync_hypothesis(
                        spectrum,
                        magnitude,
                        image.size,
                        config,
                        float(rotation),
                        float(scale),
                        search_radius=1,
                    )
                )

    best = max(refined_hypotheses, key=_hypothesis_sort_key)
    if best.supported_peaks < 3:
        return None
    alternatives = [
        item
        for item in refined_hypotheses
        if item.supported_peaks >= 3
        and (
            abs(item.spectral_rotation - best.spectral_rotation) >= 3.0
            or abs(item.scale - best.scale) >= 0.10
        )
    ]
    second_score = max((item.score for item in alternatives), default=0.0)
    score_margin = max(0.0, 1.0 - second_score / max(best.score, 1e-12))
    if second_score >= best.score * (1.0 - AMBIGUOUS_SCORE_MARGIN):
        return None
    third_ratio = sorted(best.ratios, reverse=True)[2]
    support_confidence = max(0.0, min(1.0, 1.0 - PEAK_SUPPORT_RATIO / third_ratio))
    confidence = float(max(1e-12, min(1.0, support_confidence * score_margin)))
    if confidence < MIN_SYNC_CONFIDENCE:
        return None
    offset_x: int | None = None
    offset_y: int | None = None
    if abs(best.spectral_rotation) <= 0.5:
        offset_image: Image.Image | None = image
        if abs(best.scale - 1.0) > 0.02:
            normalized_size = (
                max(1, round(image.width / best.scale)),
                max(1, round(image.height / best.scale)),
            )
            normalized_pixels = normalized_size[0] * normalized_size[1]
            if (
                max(normalized_size) <= MAX_OFFSET_ANALYSIS_SIDE
                and normalized_pixels <= MAX_OFFSET_ANALYSIS_PIXELS
            ):
                offset_image = image.resize(
                    normalized_size,
                    Image.Resampling.BICUBIC,
                )
            else:
                offset_image = None
        offset = (
            _estimate_tile_offset(offset_image, config, deadline)
            if offset_image is not None
            else None
        )
        if offset is not None:
            offset_x, offset_y = offset
    return SyncEstimate(
        rotation_degrees=float(-best.spectral_rotation),
        scale=float(best.scale),
        confidence=confidence,
        supported_peaks=best.supported_peaks,
        evaluated_hypotheses=len(coarse_hypotheses) + len(refined_hypotheses),
        elapsed_seconds=float(monotonic() - started),
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _score_sync_hypothesis(
    spectrum: np.ndarray,
    magnitude: np.ndarray,
    image_size: tuple[int, int],
    config: V4Config,
    spectral_rotation: float,
    scale: float,
    *,
    search_radius: int,
) -> _SyncHypothesis:
    image_width, image_height = image_size
    analysis_height, analysis_width = magnitude.shape
    center_y = analysis_height // 2
    center_x = analysis_width // 2
    radians = np.deg2rad(spectral_rotation)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    ratios: list[float] = []
    localization_error = 0.0
    for component_index, (frequency_x, frequency_y) in enumerate(
        config.pilot_frequency_vectors
    ):
        rotated_x = (cosine * frequency_x - sine * frequency_y) / scale
        rotated_y = (sine * frequency_x + cosine * frequency_y) / scale
        analysis_x = rotated_x * image_width / analysis_width
        analysis_y = rotated_y * image_height / analysis_height
        if abs(analysis_x) >= 0.5 or abs(analysis_y) >= 0.5:
            ratios.append(0.0)
            continue
        positive = (
            round(center_y + analysis_y * analysis_height),
            round(center_x + analysis_x * analysis_width),
        )
        negative = (
            round(center_y - analysis_y * analysis_height),
            round(center_x - analysis_x * analysis_width),
        )
        positive_ratio, positive_bin = _peak_ratio(
            magnitude, *positive, search_radius=search_radius
        )
        negative_ratio, negative_bin = _peak_ratio(
            magnitude, *negative, search_radius=search_radius
        )
        localization_error += (
            (positive_bin[0] - positive[0]) ** 2
            + (positive_bin[1] - positive[1]) ** 2
            + (negative_bin[0] - negative[0]) ** 2
            + (negative_bin[1] - negative[1]) ** 2
        )
        ratio = min(positive_ratio, negative_ratio)
        ratios.append(float(ratio))
    ratio_tuple = tuple(ratios)
    return _SyncHypothesis(
        spectral_rotation=float(spectral_rotation),
        scale=float(scale),
        supported_peaks=sum(value >= PEAK_SUPPORT_RATIO for value in ratio_tuple),
        score=float(
            sum(np.log1p(value) for value in ratio_tuple)
            - 0.01 * localization_error
        ),
        ratios=ratio_tuple,
        localization_error=float(localization_error),
    )


def _hypothesis_sort_key(item: _SyncHypothesis) -> tuple[float, int]:
    return item.score, item.supported_peaks


def _estimate_tile_offset(
    image: Image.Image,
    config: V4Config,
    deadline: float | None,
) -> tuple[int, int] | None:
    tile_size = config.tile_size
    if image.width < tile_size or image.height < tile_size:
        return None
    _check_deadline(deadline)
    grayscale = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2GRAY)
    stride = tile_size // 2
    x_starts = tuple(range(0, image.width - tile_size + 1, stride))
    y_starts = tuple(range(0, image.height - tile_size + 1, stride))
    if len(x_starts) * len(y_starts) < 4:
        return None

    window = np.hanning(tile_size)
    window_2d = window[:, None] * window[None, :]
    component_values: list[list[complex]] = [
        [] for _ in range(PILOT_COMPONENT_COUNT)
    ]
    for start_y in y_starts:
        _check_deadline(deadline)
        blocks = np.stack(
            [
                grayscale[
                    start_y : start_y + tile_size,
                    start_x : start_x + tile_size,
                ]
                for start_x in x_starts
            ]
        ).astype(np.float64)
        blocks -= np.mean(blocks, axis=(1, 2), keepdims=True)
        spectra = np.fft.fft2(blocks * window_2d, axes=(-2, -1))
        for component_index, (frequency_x, frequency_y) in enumerate(
            config.pilot_frequency_vectors
        ):
            frequency_column = round(frequency_x * tile_size)
            frequency_row = round(frequency_y * tile_size)
            values = spectra[:, frequency_row, frequency_column]
            alignment = np.asarray(
                [
                    np.exp(
                        -1j
                        * TWO_PI
                        * (frequency_x * start_x + frequency_y * start_y)
                    )
                    for start_x in x_starts
                ]
            )
            component_values[component_index].extend(values * alignment)

    coefficient_phases = []
    for values in component_values:
        array = np.asarray(values)
        robust_coefficient = complex(
            float(np.median(array.real)),
            float(np.median(array.imag)),
        )
        coefficient_phases.append(float(np.angle(robust_coefficient)))

    offsets_y, offsets_x = np.mgrid[0:tile_size, 0:tile_size]
    residuals = []
    phases = _pilot_phases(config.codec)
    for observed_phase, base_phase, (frequency_x, frequency_y) in zip(
        coefficient_phases,
        phases,
        config.pilot_frequency_vectors,
    ):
        residuals.append(
            observed_phase
            - (base_phase - np.pi / 2.0)
            - TWO_PI * (frequency_x * offsets_x + frequency_y * offsets_y)
        )
    phase_error = np.mean(1.0 - np.cos(np.stack(residuals)), axis=0)
    best_flat_index = int(np.argmin(phase_error))
    best_y, best_x = np.unravel_index(best_flat_index, phase_error.shape)
    if float(phase_error[best_y, best_x]) > 0.002:
        return None
    _check_deadline(deadline)
    return int(best_x), int(best_y)


def _phase_aligned_ratio(
    spectrum: np.ndarray,
    predicted_row: int,
    predicted_column: int,
    frequency_x: float,
    frequency_y: float,
    *,
    phases: tuple[float, ...],
    component_index: int,
) -> float:
    height, width = spectrum.shape
    center_y = height // 2
    center_x = width // 2
    fractional_x = (predicted_column - center_x) - frequency_x * width
    fractional_y = (predicted_row - center_y) - frequency_y * height
    expected_phase = (
        phases[component_index]
        - np.pi / 2.0
        - np.pi * fractional_x * (width - 1) / width
        - np.pi * fractional_y * (height - 1) / height
    )

    row_start = max(0, predicted_row - 8)
    row_stop = min(height, predicted_row + 9)
    column_start = max(0, predicted_column - 8)
    column_stop = min(width, predicted_column + 9)
    local = spectrum[row_start:row_stop, column_start:column_stop]
    projected = np.real(local * np.exp(-1j * expected_phase))
    target_row = predicted_row - row_start
    target_column = predicted_column - column_start
    target = max(0.0, float(projected[target_row, target_column]))

    rows = np.arange(row_start, row_stop)[:, None]
    columns = np.arange(column_start, column_stop)[None, :]
    background = projected[
        (np.abs(rows - predicted_row) > 2)
        | (np.abs(columns - predicted_column) > 2)
    ]
    dispersion = float(np.std(np.abs(background))) if background.size else 0.0
    if target <= 0.0:
        return 0.0
    return target / max(dispersion, np.finfo(np.float64).eps)


def _peak_ratio(
    magnitude: np.ndarray,
    predicted_row: int,
    predicted_column: int,
    *,
    search_radius: int = 1,
) -> tuple[float, tuple[int, int]]:
    height, width = magnitude.shape
    search_row_start = max(0, predicted_row - search_radius)
    search_row_stop = min(height, predicted_row + search_radius + 1)
    search_column_start = max(0, predicted_column - search_radius)
    search_column_stop = min(width, predicted_column + search_radius + 1)
    search = magnitude[
        search_row_start:search_row_stop,
        search_column_start:search_column_stop,
    ]
    peak_offset = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_bin = (
        search_row_start + int(peak_offset[0]),
        search_column_start + int(peak_offset[1]),
    )
    peak = float(magnitude[peak_bin])

    row_start = max(0, peak_bin[0] - 8)
    row_stop = min(height, peak_bin[0] + 9)
    column_start = max(0, peak_bin[1] - 8)
    column_stop = min(width, peak_bin[1] + 9)
    local = magnitude[row_start:row_stop, column_start:column_stop]
    rows = np.arange(row_start, row_stop)[:, None]
    columns = np.arange(column_start, column_stop)[None, :]
    background = local[
        (np.abs(rows - peak_bin[0]) > 2)
        | (np.abs(columns - peak_bin[1]) > 2)
    ]
    local_median = float(np.median(background)) if background.size else 0.0
    if peak <= 0.0:
        return 0.0, peak_bin
    return peak / max(local_median, np.finfo(np.float64).eps), peak_bin


def _validate_dimensions(height: int, width: int) -> None:
    if type(height) is not int or type(width) is not int:
        raise TypeError("pilot dimensions must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("pilot dimensions must be positive")


def _validate_config(config: V4Config) -> None:
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config instance")


def _validate_image(image: Image.Image) -> None:
    if type(image) is not Image.Image:
        raise TypeError("image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")


def _validated_rotation(value: float) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError("rotation must be a finite number")
    rotation = float(value)
    if not -180.0 <= rotation <= 180.0:
        raise ValueError("rotation must be between -180 and 180 degrees")
    return rotation


def _validated_scale(value: float) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError("scale must be a finite number")
    scale = float(value)
    if not 0.25 <= scale <= 4.0:
        raise ValueError("scale must be between 0.25 and 4")
    return scale


def _validate_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise TypeError("deadline must be a finite monotonic timestamp")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("FFT synchronization deadline expired")


__all__ = ("SyncEstimate", "detect_pilot", "embed_pilot", "pilot_signal")
