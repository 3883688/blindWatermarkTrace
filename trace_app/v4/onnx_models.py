"""Verified ONNX adapters for the production V4 visual models."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trace_app.v4.deadlines import Deadline
from trace_app.v4.geometry import GeometryEvidence, validate_homography
from trace_app.v4.models import ModelRegistryError, validate_dino_embeddings


SessionFactory = Callable[[Path], object]
MINIMUM_MATCHES = 24
MODEL_MAX_SIDE = 512


def _verified_model(path: str | Path) -> Path:
    model_path = Path(path).resolve(strict=True)
    checksum_path = model_path.with_suffix(model_path.suffix + ".sha256")
    try:
        fields = checksum_path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise ModelRegistryError("ONNX checksum file is unavailable") from exc
    expected = fields[0].lower() if fields else ""
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ModelRegistryError("invalid ONNX SHA-256")
    with model_path.open("rb") as model_file:
        actual = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual != expected:
        raise ModelRegistryError(f"SHA-256 mismatch for ONNX model {model_path.name}")
    return model_path


def _session(path: Path, factory: SessionFactory | None) -> object:
    if factory is not None:
        return factory(path)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ModelRegistryError("onnxruntime is required for V4 models") from exc
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class DinoOnnxModels:
    """Expose the existing DINOv2 ONNX model through the V4 infer contract."""

    def __init__(self, path: str | Path, *, session_factory: SessionFactory | None = None) -> None:
        self.path = _verified_model(path)
        self._session = _session(self.path, session_factory)
        try:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            valid = (
                len(inputs) == 1
                and inputs[0].name == "pixel_values"
                and inputs[0].type == "tensor(float)"
                and len(outputs) == 1
                and outputs[0].name == "last_hidden_state"
                and outputs[0].type == "tensor(float)"
            )
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise ModelRegistryError("invalid DINOv2 ONNX signature")

    def infer(self, name: str, batch: object) -> np.ndarray:
        array = np.asarray(batch)
        if (
            name != "dinov2_vits14"
            or array.dtype != np.dtype("float32")
            or array.ndim != 4
            or array.shape[1:] != (3, 224, 224)
            or len(array) < 1
            or not np.isfinite(array).all()
        ):
            raise ModelRegistryError("invalid DINOv2 ONNX input")
        try:
            output = np.asarray(
                self._session.run(None, {"pixel_values": np.ascontiguousarray(array)})[0],
                dtype=np.float32,
            )
        except Exception as exc:
            raise ModelRegistryError("DINOv2 ONNX inference failed") from exc
        vectors = output[:, 0, :] if output.ndim == 3 else output
        return validate_dino_embeddings(vectors)


class LightGlueOnnxMatcher:
    """Run the pinned joint SuperPoint/LightGlue ONNX geometry pipeline."""

    def __init__(self, path: str | Path, *, session_factory: SessionFactory | None = None) -> None:
        self.path = _verified_model(path)
        self._session = _session(self.path, session_factory)
        try:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            valid = (
                len(inputs) == 1
                and inputs[0].name == "images"
                and inputs[0].type == "tensor(float)"
                and [item.name for item in outputs] == ["keypoints", "matches", "mscores"]
            )
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise ModelRegistryError("invalid SuperPoint/LightGlue ONNX signature")

    def match_geometry(
        self, query: Image.Image, target: Image.Image, deadline: Deadline
    ) -> GeometryEvidence | None:
        deadline.check("lightglue_prepare")
        batch, scales, sizes = _prepare_pair(query, target)
        try:
            keypoints, matches, scores = self._session.run(None, {"images": batch})
        except Exception as exc:
            raise ModelRegistryError("SuperPoint/LightGlue ONNX inference failed") from exc
        deadline.check("lightglue_inference")
        points = _matched_points(keypoints, matches, scores, scales, sizes)
        if points is None:
            return None
        points0, points1 = points
        matrix, mask = cv2.findHomography(
            points0.reshape(-1, 1, 2), points1.reshape(-1, 1, 2), cv2.RANSAC, 3.0
        )
        deadline.check("lightglue_ransac")
        if matrix is None or mask is None:
            return None
        inliers_mask = np.asarray(mask).reshape(-1).astype(bool)
        inliers = int(np.count_nonzero(inliers_mask))
        ratio = inliers / len(points0)
        if inliers < MINIMUM_MATCHES or ratio < 0.5:
            return None
        try:
            checked = validate_homography(
                np.asarray(matrix, dtype=np.float64),
                query_size=query.size,
                target_size=target.size,
            )
        except ValueError:
            return None
        projected = cv2.perspectiveTransform(
            points0[inliers_mask].reshape(-1, 1, 2), checked
        ).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(projected - points1[inliers_mask], axis=1)))
        if not np.isfinite(error) or error > 3.0:
            return None
        return GeometryEvidence(checked, "lightglue", inliers, ratio, error)


def _prepare_pair(
    query: Image.Image, target: Image.Image
) -> tuple[np.ndarray, tuple[tuple[float, float], ...], tuple[tuple[int, int], ...]]:
    arrays: list[np.ndarray] = []
    scales: list[tuple[float, float]] = []
    sizes: list[tuple[int, int]] = []
    for image in (query, target):
        if not isinstance(image, Image.Image) or min(image.size) <= 0:
            raise ValueError("LightGlue inputs must be valid images")
        scale = min(1.0, MODEL_MAX_SIDE / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        gray = image.convert("L").resize(size, Image.Resampling.BILINEAR)
        arrays.append(np.asarray(gray, dtype=np.float32) / np.float32(255.0))
        scales.append((size[0] / image.width, size[1] / image.height))
        sizes.append(size)
    width = max(size[0] for size in sizes)
    height = max(size[1] for size in sizes)
    batch = np.zeros((2, 1, height, width), dtype=np.float32)
    for index, (array, size) in enumerate(zip(arrays, sizes, strict=True)):
        batch[index, 0, : size[1], : size[0]] = array
    return np.ascontiguousarray(batch), tuple(scales), tuple(sizes)


def _matched_points(
    keypoints: object,
    matches: object,
    scores: object,
    scales: tuple[tuple[float, float], ...],
    sizes: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    keypoints = np.asarray(keypoints)
    matches = np.asarray(matches)
    scores = np.asarray(scores)
    if (
        keypoints.dtype != np.dtype("int64")
        or keypoints.ndim != 3
        or keypoints.shape[0] != 2
        or keypoints.shape[2] != 2
        or matches.dtype != np.dtype("int64")
        or matches.ndim != 2
        or matches.shape[1] != 3
        or scores.dtype != np.dtype("float32")
        or scores.shape != (len(matches),)
        or not np.isfinite(scores).all()
    ):
        raise ModelRegistryError("malformed SuperPoint/LightGlue ONNX output")
    selected = matches[(matches[:, 0] == 0) & (scores >= 0.2)]
    if len(selected) < MINIMUM_MATCHES:
        return None
    indexes0, indexes1 = selected[:, 1], selected[:, 2]
    if (
        np.any(indexes0 < 0)
        or np.any(indexes1 < 0)
        or np.any(indexes0 >= keypoints.shape[1])
        or np.any(indexes1 >= keypoints.shape[1])
    ):
        raise ModelRegistryError("LightGlue match index is out of range")
    points0 = keypoints[0, indexes0].astype(np.float32)
    points1 = keypoints[1, indexes1].astype(np.float32)
    valid = np.ones(len(points0), dtype=bool)
    for points, (width, height) in ((points0, sizes[0]), (points1, sizes[1])):
        valid &= (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
    points0 = (points0[valid] + 0.5) / np.asarray(scales[0], dtype=np.float32) - 0.5
    points1 = (points1[valid] + 0.5) / np.asarray(scales[1], dtype=np.float32) - 0.5
    if len(points0) < MINIMUM_MATCHES:
        return None
    return np.ascontiguousarray(points0), np.ascontiguousarray(points1)


__all__ = ("DinoOnnxModels", "LightGlueOnnxMatcher")
