from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from PIL import Image

from trace_app.v4.deadlines import Deadline
from trace_app.v4.generation import (
    AuthTagCollision,
    EncodedImages,
    GenerationRequest,
    GroupArtifacts,
    StagedMedia,
    V4GenerationService,
)
from trace_app.v4.keys import KeyRing
from trace_app.v4.repository import EmbeddingInput, FeatureInput, StoredSourceGroup
from watermark_v4.payload import CODEC_ID


def _image_bytes() -> bytes:
    paths = sorted(path for path in Path("img").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    assert paths
    return paths[0].read_bytes()


def _decode(content: bytes, deadline: Deadline) -> np.ndarray:
    deadline.check("test_decode")
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _encode(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return output.getvalue()


class _Media:
    def __init__(self) -> None:
        self.staged: list[StagedMedia] = []
        self.discarded: list[str] = []
        self.promoted: list[str] = []

    def stage_bytes(self, *, owner_user_id, variant, content_type, content):
        item = StagedMedia(
            media_id=f"media-{len(self.staged)}",
            owner_user_id=owner_user_id,
            variant=variant,
            storage_key=f"{variant}/opaque-{len(self.staged)}.bin",
            content_type=content_type,
            content=content,
        )
        self.staged.append(item)
        return item

    def discard(self, item):
        self.discarded.append(item.media_id)

    def promote(self, item):
        self.promoted.append(item.media_id)


class _Repository:
    def __init__(self) -> None:
        self.groups: dict[tuple[int, bytes], StoredSourceGroup] = {}
        self.records = []
        self.audits: list[tuple[str, str]] = []
        self.collide_once = False

    def find_source_group(self, owner_user_id, source_hash):
        return self.groups.get((owner_user_id, source_hash))

    def auth_tag_exists(self, source_group_id, tag):
        return False

    def commit_generation(self, unit):
        if self.collide_once:
            self.collide_once = False
            raise AuthTagCollision("collision")
        key = (unit.group.owner_user_id, unit.group.original_image_sha256)
        existing = self.groups.get(key)
        created = existing is None
        group = existing or StoredSourceGroup(
            id=unit.provisional_group_id,
            owner_user_id=unit.group.owner_user_id,
            original_image_sha256=unit.group.original_image_sha256,
            image_width=unit.group.image_width,
            image_height=unit.group.image_height,
            original_media_id=unit.group.original_media_id,
            model_version=unit.group.model_version,
            feature_schema_version=unit.group.feature_schema_version,
            status="active",
        )
        self.groups[key] = group
        record = replace(unit.record, source_group_id=group.id)
        self.records.append((record, created, unit.group_artifacts, unit.media))
        return record, created

    def append_generation_failure(self, *, owner_user_id, correlation_id, reason):
        self.audits.append((reason, str(correlation_id)))


def _artifacts() -> GroupArtifacts:
    return GroupArtifacts(
        embeddings=(EmbeddingInput(0, "full", [1.0] + [0.0] * 383, "dino-v1"),),
        features=(
            FeatureInput("orb", "1", "orb-v1", b"orb", hashlib.sha256(b"orb").digest()),
            FeatureInput("superpoint", "1", "sp-v1", b"sp", hashlib.sha256(b"sp").digest()),
        ),
        model_version="dino-v1",
        feature_schema_version="1",
    )


def _service(repository, media, *, build_calls, traces=None, fail_embed=False):
    trace_values = iter(traces or ("TRACE-1", "TRACE-2", "TRACE-3"))

    def build(rgb, deadline):
        build_calls.append(rgb.shape)
        return _artifacts()

    def embed(rgb, tag, deadline):
        if fail_embed:
            raise RuntimeError("secret internal detail")
        changed = rgb.copy()
        changed[0, 0, 0] ^= np.uint8(tag[0])
        return EncodedImages(_encode(changed), _encode(changed[: min(32, len(changed)), : min(32, changed.shape[1])]))

    return V4GenerationService(
        repository=repository,
        key_ring=KeyRing({"key-2026": b"k" * 32}, "key-2026"),
        media=media,
        decode_rgb=_decode,
        build_group_artifacts=build,
        embed=embed,
        trace_id_factory=lambda: next(trace_values),
    )


def test_generation_groups_by_canonical_rgb_per_owner_and_reuses_features() -> None:
    repository, media, build_calls = _Repository(), _Media(), []
    service = _service(repository, media, build_calls=build_calls)
    request = GenerationRequest(7, _image_bytes(), "image/png")

    first = service.generate(request, Deadline.after(10))
    second = service.generate(request, Deadline.after(10))
    other_owner = service.generate(replace(request, owner_user_id=8), Deadline.after(10))

    assert first.record.source_group_id == second.record.source_group_id
    assert other_owner.record.source_group_id != first.record.source_group_id
    assert len(build_calls) == 2
    assert all(item[0].codec == CODEC_ID for item in repository.records)
    assert all(len(item[0].auth_tag) == 8 for item in repository.records)
    assert all(item.content_type == "image/png" for item in media.staged[1::3])
    assert len(media.promoted) == 9


def test_group_artifacts_reject_feature_checksum_mismatch() -> None:
    good = _artifacts()
    with pytest.raises(ValueError, match="checksum"):
        GroupArtifacts(
            good.embeddings,
            (replace(good.features[0], feature_sha256=b"x" * 32), good.features[1]),
            good.model_version,
            good.feature_schema_version,
        )

    with pytest.raises(ValueError, match="384-dimensional"):
        GroupArtifacts(
            (replace(good.embeddings[0], embedding=[1.0, 0.0]),),
            good.features,
            good.model_version,
            good.feature_schema_version,
        )


def test_auth_tag_collision_discards_staged_media_and_retries_trace() -> None:
    repository, media, build_calls = _Repository(), _Media(), []
    repository.collide_once = True
    service = _service(repository, media, build_calls=build_calls, traces=("COLLIDE", "RETRY"))

    result = service.generate(GenerationRequest(7, _image_bytes(), "image/png"), Deadline.after(10))

    assert result.record.trace_id == "RETRY"
    assert len(media.discarded) == 3
    assert len(media.promoted) == 3
    assert len(repository.records) == 1


def test_failure_cleans_every_staged_object_and_writes_redacted_audit() -> None:
    repository, media, build_calls = _Repository(), _Media(), []
    service = _service(repository, media, build_calls=build_calls, fail_embed=True)

    with pytest.raises(RuntimeError, match="secret internal detail"):
        service.generate(GenerationRequest(7, _image_bytes(), "image/png"), Deadline.after(10))

    assert repository.records == []
    assert media.promoted == []
    assert repository.audits and repository.audits[0][0] == "generation_failed"
    assert "secret" not in repr(repository.audits)


def test_partial_media_staging_failure_discards_earlier_objects() -> None:
    repository, build_calls = _Repository(), []

    class FailingMedia(_Media):
        def stage_bytes(self, **kwargs):
            if len(self.staged) == 1:
                raise OSError("staging full")
            return super().stage_bytes(**kwargs)

    media = FailingMedia()
    service = _service(repository, media, build_calls=build_calls)

    with pytest.raises(OSError, match="staging full"):
        service.generate(GenerationRequest(7, _image_bytes(), "image/png"), Deadline.after(10))

    assert media.discarded == ["media-0"]
    assert repository.records == []


def test_commit_happens_only_after_output_decodes_and_hashes() -> None:
    events: list[str] = []
    repository, media = _Repository(), _Media()

    class OrderedRepository(_Repository):
        def commit_generation(self, unit):
            events.append("commit")
            return super().commit_generation(unit)

    repository = OrderedRepository()

    def decode(content, deadline):
        events.append("decode")
        return _decode(content, deadline)

    service = V4GenerationService(
        repository=repository,
        key_ring=KeyRing({"key-2026": b"k" * 32}, "key-2026"),
        media=media,
        decode_rgb=decode,
        build_group_artifacts=lambda rgb, deadline: events.append("models") or _artifacts(),
        embed=lambda rgb, tag, deadline: events.append("embed") or EncodedImages(_encode(rgb), _encode(rgb[:8, :8])),
        trace_id_factory=lambda: "TRACE",
    )

    service.generate(GenerationRequest(7, _image_bytes(), "image/png"), Deadline.after(10))

    assert events.index("commit") > events.index("models")
    assert events.index("commit") > events.index("embed")
    assert events.count("decode") == 3
