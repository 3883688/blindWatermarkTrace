"""Detect face and foot/shoe regions for optional V4 watermark reinforcement."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
REGION_ANALYSIS_MAX_SIDE = 1024
PERSON_MODEL_SIZE = 416
POSE_INPUT_WIDTH = 192
POSE_INPUT_HEIGHT = 256
PERSON_CONFIDENCE_THRESHOLD = 0.3
KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
MAX_PEOPLE = 8
_DETECTOR_LOCK = Lock()
_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
_PERSON_MODEL_PATH = _MODEL_DIRECTORY / "yolox-tiny-humanart-person.onnx"
_POSE_MODEL_PATH = _MODEL_DIRECTORY / "dwpose-s-wholebody.onnx"


@dataclass(frozen=True, slots=True)
class ProtectedRegion:
    kind: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    def __post_init__(self) -> None:
        if self.kind not in {"face", "foot_shoe"}:
            raise ValueError("unsupported protected region kind")
        if not (0 <= self.left < self.right and 0 <= self.top < self.bottom):
            raise ValueError("protected region bounds are invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("protected region confidence is invalid")


def detect_protected_regions(rgb: np.ndarray) -> tuple[ProtectedRegion, ...]:
    """Find face and foot/shoe regions using local person and pose models.

    CUDA is selected automatically when ONNX Runtime exposes it. The built-in
    OpenCV detectors remain an offline fallback when either model is unavailable
    or model inference fails; the caller's full-image watermark is unaffected.
    """
    image = _validated_rgb(rgb)
    try:
        with _DETECTOR_LOCK:
            return _detect_with_pose_models(image)
    except Exception as exc:
        LOGGER.warning(
            "protected-region ONNX inference failed; using OpenCV fallback: %s",
            exc,
        )
        with _DETECTOR_LOCK:
            return _detect_with_builtin_detectors(image)


def _detect_with_pose_models(image: np.ndarray) -> tuple[ProtectedRegion, ...]:
    person_session, pose_session = _onnx_sessions()
    people = _detect_people_onnx(image, person_session)
    regions: list[ProtectedRegion] = []
    for person_box, person_confidence in people:
        points, scores = _estimate_wholebody_pose(image, person_box, pose_session)
        regions.extend(
            _regions_from_keypoints(
                points,
                scores,
                person_box,
                person_confidence,
                image.shape[1],
                image.shape[0],
            )
        )
    return _deduplicated(regions)


def _detect_people_onnx(
    image: np.ndarray, session: Any
) -> list[tuple[tuple[float, float, float, float], float]]:
    height, width = image.shape[:2]
    ratio = min(PERSON_MODEL_SIZE / height, PERSON_MODEL_SIZE / width)
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (max(1, round(width * ratio)), max(1, round(height * ratio))),
        interpolation=cv2.INTER_LINEAR,
    )
    model_input = np.full(
        (PERSON_MODEL_SIZE, PERSON_MODEL_SIZE, 3), 114, dtype=np.uint8
    )
    model_input[: resized.shape[0], : resized.shape[1]] = resized
    batch = np.ascontiguousarray(
        model_input.transpose(2, 0, 1)[np.newaxis], dtype=np.float32
    )
    outputs = session.run(None, {session.get_inputs()[0].name: batch})
    detections = np.asarray(outputs[0])[0]
    labels = np.asarray(outputs[1])[0] if len(outputs) > 1 else np.zeros(len(detections))

    people: list[tuple[tuple[float, float, float, float], float]] = []
    for detection, label in zip(detections, labels, strict=False):
        if detection.shape[0] < 5 or int(label) != 0:
            continue
        confidence = float(detection[4])
        if not np.isfinite(confidence) or confidence < PERSON_CONFIDENCE_THRESHOLD:
            continue
        left, top, right, bottom = (float(value) / ratio for value in detection[:4])
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(float(width), right), min(float(height), bottom)
        if right - left < 8 or bottom - top < 8:
            continue
        people.append(((left, top, right, bottom), min(1.0, confidence)))
    people.sort(key=lambda item: item[1], reverse=True)
    return people[:MAX_PEOPLE]


def _estimate_wholebody_pose(
    image: np.ndarray,
    person_box: tuple[float, float, float, float],
    session: Any,
) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = person_box
    center_x, center_y = (left + right) * 0.5, (top + bottom) * 0.5
    scale_width, scale_height = (right - left) * 1.25, (bottom - top) * 1.25
    model_aspect = POSE_INPUT_WIDTH / POSE_INPUT_HEIGHT
    if scale_width / scale_height > model_aspect:
        scale_height = scale_width / model_aspect
    else:
        scale_width = scale_height * model_aspect

    source_left = center_x - scale_width * 0.5
    source_top = center_y - scale_height * 0.5
    transform = np.array(
        [
            [
                POSE_INPUT_WIDTH / scale_width,
                0.0,
                -source_left * POSE_INPUT_WIDTH / scale_width,
            ],
            [
                0.0,
                POSE_INPUT_HEIGHT / scale_height,
                -source_top * POSE_INPUT_HEIGHT / scale_height,
            ],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        transform,
        (POSE_INPUT_WIDTH, POSE_INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)
    crop = (
        crop - np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
    ) / np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
    batch = np.ascontiguousarray(crop.transpose(2, 0, 1)[np.newaxis])
    simcc_x, simcc_y = session.run(None, {session.get_inputs()[0].name: batch})[:2]
    simcc_x, simcc_y = np.asarray(simcc_x)[0], np.asarray(simcc_y)[0]
    x_indices = np.argmax(simcc_x, axis=1).astype(np.float32) / 2.0
    y_indices = np.argmax(simcc_y, axis=1).astype(np.float32) / 2.0
    points = np.column_stack(
        (
            source_left + x_indices / POSE_INPUT_WIDTH * scale_width,
            source_top + y_indices / POSE_INPUT_HEIGHT * scale_height,
        )
    )
    scores = (np.max(simcc_x, axis=1) + np.max(simcc_y, axis=1)) * 0.5
    return points, scores


def _regions_from_keypoints(
    points: np.ndarray,
    scores: np.ndarray,
    person_box: tuple[float, float, float, float],
    person_confidence: float,
    image_width: int,
    image_height: int,
) -> list[ProtectedRegion]:
    if points.shape != (133, 2) or scores.shape != (133,):
        raise RuntimeError("whole-body pose model returned an unexpected shape")
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    regions: list[ProtectedRegion] = []

    face = _confident_points(points, scores, range(23, 91))
    if len(face) >= 4:
        confidence = min(person_confidence, float(np.mean(face[:, 2])))
        bounds = _point_bounds(
            face[:, :2],
            image_width,
            image_height,
            padding_x=max(person_width * 0.025, np.ptp(face[:, 0]) * 0.18),
            padding_y=max(person_height * 0.02, np.ptp(face[:, 1]) * 0.18),
        )
        regions.append(ProtectedRegion("face", *bounds, _unit_confidence(confidence)))

    for indices in ((15, 17, 18, 19), (16, 20, 21, 22)):
        foot = _confident_points(points, scores, indices)
        if len(foot) < 2:
            continue
        confidence = min(person_confidence, float(np.mean(foot[:, 2])))
        span_x = float(np.ptp(foot[:, 0]))
        span_y = float(np.ptp(foot[:, 1]))
        bounds = _point_bounds(
            foot[:, :2],
            image_width,
            image_height,
            padding_x=max(person_width * 0.055, span_x * 0.45),
            padding_y=max(person_height * 0.025, span_y * 0.30),
        )
        regions.append(
            ProtectedRegion("foot_shoe", *bounds, _unit_confidence(confidence))
        )
    return regions


def _confident_points(
    points: np.ndarray, scores: np.ndarray, indices: object
) -> np.ndarray:
    selected = np.asarray(list(indices) if not isinstance(indices, tuple) else indices)
    mask = scores[selected] >= KEYPOINT_CONFIDENCE_THRESHOLD
    return np.column_stack((points[selected][mask], scores[selected][mask]))


def _point_bounds(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    padding_x: float,
    padding_y: float,
) -> tuple[int, int, int, int]:
    left = max(
        0, min(image_width - 1, int(np.floor(np.min(points[:, 0]) - padding_x)))
    )
    top = max(
        0, min(image_height - 1, int(np.floor(np.min(points[:, 1]) - padding_y)))
    )
    right = max(
        left + 1,
        min(image_width, int(np.ceil(np.max(points[:, 0]) + padding_x))),
    )
    bottom = max(
        top + 1,
        min(image_height, int(np.ceil(np.max(points[:, 1]) + padding_y))),
    )
    return left, top, right, bottom


def _unit_confidence(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@lru_cache(maxsize=1)
def _onnx_sessions() -> tuple[Any, Any]:
    if not _PERSON_MODEL_PATH.is_file() or not _POSE_MODEL_PATH.is_file():
        raise OSError("protected-region ONNX models are unavailable")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is unavailable") from exc
    available = set(ort.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    if not providers:
        raise RuntimeError("onnxruntime has no supported execution provider")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return (
        ort.InferenceSession(str(_PERSON_MODEL_PATH), options, providers=providers),
        ort.InferenceSession(str(_POSE_MODEL_PATH), options, providers=providers),
    )


def _detect_with_builtin_detectors(image: np.ndarray) -> tuple[ProtectedRegion, ...]:
    analysis, scale = _analysis_image(image)
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)

    face_boxes = list(
        _frontal_face_detector().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
    )
    profile_boxes = list(
        _profile_face_detector().detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(24, 24),
        )
    )
    mirrored_profile_boxes = list(
        _profile_face_detector().detectMultiScale(
            cv2.flip(gray, 1),
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(24, 24),
        )
    )
    people_boxes, people_weights = _people_detector().detectMultiScale(
        cv2.cvtColor(analysis, cv2.COLOR_RGB2BGR),
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    height, width = image.shape[:2]
    regions: list[ProtectedRegion] = []
    for box in face_boxes + profile_boxes:
        left, top, right, bottom = _scaled_bounds(box, scale, width, height, 0.12)
        regions.append(ProtectedRegion("face", left, top, right, bottom, 0.85))
    analysis_width = analysis.shape[1]
    for x, y, box_width, box_height in mirrored_profile_boxes:
        mirrored_box = (analysis_width - x - box_width, y, box_width, box_height)
        left, top, right, bottom = _scaled_bounds(
            mirrored_box, scale, width, height, 0.12
        )
        regions.append(ProtectedRegion("face", left, top, right, bottom, 0.85))

    for box, weight in zip(people_boxes, people_weights, strict=False):
        x, y, box_width, box_height = (float(value) for value in box)
        # The bottom 24% covers ankles, feet and shoes while excluding most clothing.
        foot_top = y + box_height * 0.76
        center = x + box_width * 0.5
        overlap = box_width * 0.08
        confidence = max(0.5, min(0.95, 0.5 + float(weight) * 0.08))
        for start, end in ((x, center + overlap), (center - overlap, x + box_width)):
            bounds = _scaled_bounds(
                (start, foot_top, end - start, y + box_height - foot_top),
                scale,
                width,
                height,
                0.18,
            )
            regions.append(ProtectedRegion("foot_shoe", *bounds, confidence))

    return _deduplicated(regions)


def reinforced_tiles(
    regions: tuple[ProtectedRegion, ...],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> frozenset[tuple[int, int]]:
    """Return complete V4 tiles intersecting any protected region."""
    if image_width <= 0 or image_height <= 0 or tile_size <= 0:
        raise ValueError("image and tile dimensions must be positive")
    columns = image_width // tile_size
    rows = image_height // tile_size
    selected: set[tuple[int, int]] = set()
    for region in regions:
        first_x = max(0, region.left // tile_size)
        last_x = min(columns - 1, (region.right - 1) // tile_size)
        first_y = max(0, region.top // tile_size)
        last_y = min(rows - 1, (region.bottom - 1) // tile_size)
        if first_x > last_x or first_y > last_y:
            continue
        selected.update(
            (tile_x, tile_y)
            for tile_y in range(first_y, last_y + 1)
            for tile_x in range(first_x, last_x + 1)
        )
    return frozenset(selected)


def _validated_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.dtype("uint8") or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("region detection input must be uint8 RGB")
    if min(image.shape[:2]) <= 0:
        raise ValueError("region detection input must be non-empty")
    return np.ascontiguousarray(image)


def _analysis_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= REGION_ANALYSIS_MAX_SIDE:
        return image, 1.0
    scale = REGION_ANALYSIS_MAX_SIDE / longest
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _scaled_bounds(
    box: object,
    scale: float,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in box)
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    left = min(image_width - 1, max(0, int((x - padding_x) / scale)))
    top = min(image_height - 1, max(0, int((y - padding_y) / scale)))
    right = min(
        image_width,
        max(left + 1, int(np.ceil((x + width + padding_x) / scale))),
    )
    bottom = min(
        image_height,
        max(top + 1, int(np.ceil((y + height + padding_y) / scale))),
    )
    return left, top, right, bottom


def _deduplicated(regions: list[ProtectedRegion]) -> tuple[ProtectedRegion, ...]:
    accepted: list[ProtectedRegion] = []
    for candidate in sorted(regions, key=lambda item: item.confidence, reverse=True):
        if any(
            candidate.kind == existing.kind
            and _intersection_over_union(candidate, existing) >= 0.55
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return tuple(accepted)


def _intersection_over_union(first: ProtectedRegion, second: ProtectedRegion) -> float:
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    return intersection / max(1, first_area + second_area - intersection)


@lru_cache(maxsize=1)
def _frontal_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("OpenCV frontal face detector is unavailable")
    return detector


@lru_cache(maxsize=1)
def _profile_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    if detector.empty():
        raise RuntimeError("OpenCV profile face detector is unavailable")
    return detector


@lru_cache(maxsize=1)
def _people_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


__all__ = ("ProtectedRegion", "detect_protected_regions", "reinforced_tiles")
