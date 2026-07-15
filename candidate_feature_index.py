from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FEATURE_INDEX_MAX_SIDE = 640
FEATURE_INDEX_MAX_DESCRIPTORS = 768
FEATURE_DESCRIPTOR_BYTES = 32


def extract_feature_descriptors(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    scale = min(1.0, FEATURE_INDEX_MAX_SIDE / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale)))),
            Image.Resampling.BICUBIC,
        )
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(
        nfeatures=FEATURE_INDEX_MAX_DESCRIPTORS,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=7,
    )
    _, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or descriptors.size == 0:
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    return np.ascontiguousarray(
        descriptors[:FEATURE_INDEX_MAX_DESCRIPTORS],
        dtype=np.uint8,
    )


def save_feature_descriptors(path: Path, descriptors: np.ndarray) -> None:
    normalized = np.asarray(descriptors, dtype=np.uint8)
    if normalized.ndim != 2 or normalized.shape[1] != FEATURE_DESCRIPTOR_BYTES:
        raise ValueError("ORB descriptors must have shape (n, 32)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, descriptors=normalized)


def load_feature_descriptors(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            descriptors = np.asarray(payload["descriptors"], dtype=np.uint8)
    except (OSError, KeyError, ValueError):
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    if descriptors.ndim != 2 or descriptors.shape[1] != FEATURE_DESCRIPTOR_BYTES:
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    return np.ascontiguousarray(descriptors, dtype=np.uint8)


def descriptor_match_score(
    query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
) -> tuple[int, float]:
    query = np.asarray(query_descriptors, dtype=np.uint8)
    target = np.asarray(target_descriptors, dtype=np.uint8)
    if (
        query.ndim != 2
        or target.ndim != 2
        or query.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        or target.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        or len(query) < 2
        or len(target) < 2
    ):
        return 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query, target, k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    count = len(good)
    quality = count / max(1, min(len(query), len(target)))
    return count, round(float(quality), 6)
