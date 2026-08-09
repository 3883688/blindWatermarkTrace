"""Detect face and foot/shoe regions for optional V4 watermark reinforcement."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np


REGION_ANALYSIS_MAX_SIDE = 1024
_DETECTOR_LOCK = Lock()


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
    """Find frontal/profile faces and the foot zones of detected people.

    OpenCV ships the face cascades and pedestrian HOG descriptor with the runtime,
    so this path remains available in offline deployments and adds no model download.
    A person's lower leg box is split into left/right foot-shoe regions so that a
    single large person box does not reinforce unrelated image content.
    """
    image = _validated_rgb(rgb)
    analysis, scale = _analysis_image(image)
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)

    with _DETECTOR_LOCK:
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
