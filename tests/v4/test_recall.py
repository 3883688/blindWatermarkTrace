from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from trace_app.v4.deadlines import Deadline
from trace_app.v4.domain import OwnerScope
from trace_app.v4.recall import (
    VIEW_POLICY_VERSION,
    build_dino_batch,
    generate_view_boxes,
    recall_image,
    recall_source_groups,
)
from trace_app.v4.repository import RecalledGroup


def _current_image() -> Image.Image:
    paths = sorted(path for path in Path("img").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    assert paths, "img/ must contain at least one supported test image"
    with Image.open(paths[0]) as image:
        return image.convert("RGB")


def test_view_policy_is_deterministic_full_image_and_overlapping() -> None:
    image = _current_image()

    first = generate_view_boxes(image.width, image.height)
    second = generate_view_boxes(image.width, image.height)

    assert first == second
    assert first[0].kind == "full"
    assert (first[0].x, first[0].y, first[0].width, first[0].height) == (0, 0, image.width, image.height)
    assert len(first) > 1
    assert len({box for box in first}) == len(first)
    assert all(box.x >= 0 and box.y >= 0 and box.x + box.width <= image.width and box.y + box.height <= image.height for box in first)
    assert VIEW_POLICY_VERSION.startswith("v4-")


def test_dino_views_are_batched_as_finite_normalized_float32() -> None:
    batch, boxes = build_dino_batch(_current_image())

    assert batch.dtype == np.float32
    assert batch.shape == (len(boxes), 3, 224, 224)
    assert np.isfinite(batch).all()


class _RecallRepository:
    def __init__(self, rows: list[tuple[RecalledGroup, ...]]) -> None:
        self.rows = iter(rows)
        self.calls: list[tuple[OwnerScope, tuple[float, ...], int, int]] = []

    def recall_groups(self, scope, embedding, *, group_limit, neighbor_limit):
        self.calls.append((scope, tuple(embedding), group_limit, neighbor_limit))
        return next(self.rows)


def _row(value: int, distance: float, views: int = 1) -> RecalledGroup:
    return RecalledGroup(UUID(int=value), distance, views, 0.0)


def test_multi_view_recall_aggregates_stably_and_preserves_owner_scope() -> None:
    repository = _RecallRepository(
        [
            (_row(2, 0.08), _row(1, 0.10)),
            (_row(1, 0.07), _row(2, 0.08)),
            (_row(2, 0.09), _row(1, 0.07)),
        ]
    )
    embeddings = np.eye(3, 384, dtype=np.float32)
    scope = OwnerScope(user_id=9)

    result = recall_source_groups(scope, embeddings, repository, Deadline.after(10))

    assert [item.source_group_id for item in result] == [UUID(int=1), UUID(int=2)]
    assert result[0].matching_view_count == 3
    assert result[0].best_distance == 0.07
    assert all(call[0] == scope for call in repository.calls)
    assert all(call[2:] == (40, 400) for call in repository.calls)


def test_recall_caps_distinct_groups_at_40_with_uuid_tie_break() -> None:
    rows = tuple(_row(value, 0.1) for value in range(50, 0, -1))
    repository = _RecallRepository([rows])

    result = recall_source_groups(
        OwnerScope(user_id=1),
        np.full((1, 384), 1.0 / np.sqrt(384), dtype=np.float32),
        repository,
        Deadline.after(10),
    )

    assert len(result) == 40
    assert [item.source_group_id.int for item in result] == list(range(1, 41))


def test_recall_rejects_non_unit_dino_embeddings() -> None:
    with np.testing.assert_raises_regex(ValueError, "unit-normalized"):
        recall_source_groups(
            OwnerScope(user_id=1),
            np.ones((1, 384), dtype=np.float32),
            _RecallRepository([()]),
            Deadline.after(10),
        )


def test_image_recall_batches_all_views_through_dino_once() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []
    repository = _RecallRepository([() for _ in generate_view_boxes(*_current_image().size)])

    class Models:
        def infer(self, name, batch):
            calls.append((name, batch.shape))
            return np.eye(len(batch), 384, dtype=np.float32)

    result = recall_image(
        OwnerScope(user_id=4), _current_image(), Models(), repository, Deadline.after(10)
    )

    assert result == ()
    assert len(calls) == 1
    assert calls[0][0] == "dinov2_vits14"
