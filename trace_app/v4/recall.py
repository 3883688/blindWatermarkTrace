"""Deterministic multi-view DINO recall for V4 source groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from uuid import UUID

import cv2
import numpy as np
from PIL import Image

from trace_app.v4.deadlines import Deadline
from trace_app.v4.domain import OwnerScope
from trace_app.v4.repository import RecalledGroup


VIEW_POLICY_VERSION = "v4-multiview-1"
MAX_RECALLED_GROUPS = 40
_DINO_SIDE = 224
_DINO_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
_DINO_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


@dataclass(frozen=True, slots=True, order=True)
class ViewBox:
    kind: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class RecallCandidate:
    source_group_id: UUID
    best_distance: float
    matching_view_count: int
    distance_consistency: float


class RecallRepository(Protocol):
    def recall_groups(
        self,
        scope: OwnerScope,
        embedding: Sequence[float],
        *,
        group_limit: int,
        neighbor_limit: int,
    ) -> tuple[RecalledGroup, ...]: ...


class DinoModels(Protocol):
    def infer(self, name: str, *args: object) -> object: ...


def generate_view_boxes(width: int, height: int) -> tuple[ViewBox, ...]:
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    boxes = [ViewBox("full", 0, 0, width, height)]
    seen = {(0, 0, width, height)}
    for fraction, kind in ((0.75, "overlap_75"), (0.5, "overlap_50")):
        crop_width = max(1, round(width * fraction))
        crop_height = max(1, round(height * fraction))
        x_positions = (0, (width - crop_width) // 2, width - crop_width)
        y_positions = (0, (height - crop_height) // 2, height - crop_height)
        for y in y_positions:
            for x in x_positions:
                key = (x, y, crop_width, crop_height)
                if key not in seen:
                    boxes.append(ViewBox(kind, x, y, crop_width, crop_height))
                    seen.add(key)
    return tuple(boxes)


def build_dino_batch(image: Image.Image) -> tuple[np.ndarray, tuple[ViewBox, ...]]:
    if type(image) is not Image.Image:
        raise TypeError("DINO source must be an exact PIL Image")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    boxes = generate_view_boxes(image.width, image.height)
    batch = np.empty((len(boxes), 3, _DINO_SIDE, _DINO_SIDE), dtype=np.float32)
    for index, box in enumerate(boxes):
        crop = rgb[box.y : box.y + box.height, box.x : box.x + box.width]
        resized = cv2.resize(crop, (_DINO_SIDE, _DINO_SIDE), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / np.float32(255.0) - _DINO_MEAN) / _DINO_STD
        batch[index] = np.transpose(normalized, (2, 0, 1))
    if not np.isfinite(batch).all():
        raise ValueError("DINO preprocessing produced non-finite values")
    return batch, boxes


def recall_source_groups(
    scope: OwnerScope,
    embeddings: np.ndarray,
    repository: RecallRepository,
    deadline: Deadline,
) -> tuple[RecallCandidate, ...]:
    vectors = np.asarray(embeddings)
    if vectors.dtype != np.dtype("float32") or vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError("query embeddings must be float32 with shape (n, 384)")
    if vectors.shape[0] < 1 or not np.isfinite(vectors).all():
        raise ValueError("query embeddings must be non-empty and finite")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("query embeddings must be unit-normalized")
    observations: dict[UUID, list[float]] = {}
    for vector in vectors:
        deadline.check("dino_recall_query")
        rows = repository.recall_groups(
            scope,
            vector.tolist(),
            group_limit=MAX_RECALLED_GROUPS,
            neighbor_limit=400,
        )
        deadline.check("dino_recall_query")
        seen_in_view: set[UUID] = set()
        for row in rows:
            distance = float(row.best_distance)
            if row.source_group_id in seen_in_view or not np.isfinite(distance) or distance < 0:
                continue
            observations.setdefault(row.source_group_id, []).append(distance)
            seen_in_view.add(row.source_group_id)
    candidates = [
        RecallCandidate(
            source_group_id=group_id,
            best_distance=min(distances),
            matching_view_count=len(distances),
            distance_consistency=float(np.std(np.asarray(distances, dtype=np.float64))),
        )
        for group_id, distances in observations.items()
    ]
    candidates.sort(
        key=lambda item: (
            item.best_distance,
            -item.matching_view_count,
            item.distance_consistency,
            item.source_group_id.int,
        )
    )
    return tuple(candidates[:MAX_RECALLED_GROUPS])


def recall_image(
    scope: OwnerScope,
    image: Image.Image,
    models: DinoModels,
    repository: RecallRepository,
    deadline: Deadline,
) -> tuple[RecallCandidate, ...]:
    deadline.check("dino_preprocess")
    batch, _ = build_dino_batch(image)
    deadline.check("dino_inference")
    embeddings = models.infer("dinov2_vits14", batch)
    deadline.check("dino_inference")
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("DINO inference must return a NumPy array")
    return recall_source_groups(scope, embeddings, repository, deadline)


__all__ = (
    "MAX_RECALLED_GROUPS",
    "RecallCandidate",
    "VIEW_POLICY_VERSION",
    "ViewBox",
    "build_dino_batch",
    "generate_view_boxes",
    "recall_image",
    "recall_source_groups",
)
