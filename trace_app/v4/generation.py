"""Transactional, source-group-first orchestration for V4 generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol
from uuid import UUID, uuid4

import numpy as np

from trace_app.v4.deadlines import Deadline
from trace_app.v4.keys import KeyRing
from trace_app.v4.recall import VIEW_POLICY_VERSION
from trace_app.v4.repository import (
    AuthTagCollision,
    EmbeddingInput,
    FeatureInput,
    MediaObjectInput,
    SourceGroupInput,
    StoredSourceGroup,
    V4RecordInput,
)
from watermark_v4.payload import AuthContext, CODEC_ID


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    owner_user_id: int
    content: bytes
    content_type: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.owner_user_id, bool) or self.owner_user_id <= 0:
            raise ValueError("owner_user_id must be positive")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("generation content must be non-empty bytes")
        if not isinstance(self.content_type, str) or not self.content_type.startswith("image/"):
            raise ValueError("generation content type must be an image")


@dataclass(frozen=True, slots=True)
class EncodedImages:
    watermarked: bytes
    thumbnail: bytes

    def __post_init__(self) -> None:
        if not self.watermarked or not self.thumbnail:
            raise ValueError("encoded V4 output and thumbnail must be non-empty")


@dataclass(frozen=True, slots=True)
class GroupArtifacts:
    embeddings: tuple[EmbeddingInput, ...]
    features: tuple[FeatureInput, ...]
    model_version: str
    feature_schema_version: str
    view_policy_version: str = VIEW_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.embeddings or {item.feature_kind for item in self.features} != {"orb", "superpoint"}:
            raise ValueError("new source groups require DINO, ORB, and SuperPoint features")
        if len({item.view_index for item in self.embeddings}) != len(self.embeddings):
            raise ValueError("source group embedding view indexes must be unique")
        for item in self.embeddings:
            try:
                vector = np.asarray(item.embedding, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("DINO embeddings must be 384-dimensional unit vectors") from exc
            if (
                vector.shape != (384,)
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-4)
            ):
                raise ValueError("DINO embeddings must be 384-dimensional unit vectors")
        if any(
            hashlib.sha256(item.feature_bytes).digest() != item.feature_sha256
            for item in self.features
        ):
            raise ValueError("source group feature checksum mismatch")
        if not self.model_version or not self.feature_schema_version or not self.view_policy_version:
            raise ValueError("source group artifact versions must be non-empty")


@dataclass(frozen=True, slots=True)
class StagedMedia:
    media_id: str
    owner_user_id: int
    variant: str
    storage_key: str
    content_type: str
    content: bytes = field(repr=False)

    @property
    def media_input(self) -> MediaObjectInput:
        return MediaObjectInput(
            id=self.media_id,
            owner_user_id=self.owner_user_id,
            variant=self.variant,
            storage_key=self.storage_key,
            content_type=self.content_type,
            byte_size=len(self.content),
            sha256=hashlib.sha256(self.content).digest(),
            status="active",
        )


@dataclass(frozen=True, slots=True)
class GenerationUnit:
    provisional_group_id: UUID
    group: SourceGroupInput
    group_artifacts: GroupArtifacts | None
    media: tuple[StagedMedia, StagedMedia, StagedMedia]
    record: V4RecordInput
    correlation_id: UUID


@dataclass(frozen=True, slots=True)
class GenerationResult:
    record: object
    source_group_created: bool


class GenerationRepository(Protocol):
    def find_source_group(self, owner_user_id: int, source_hash: bytes) -> StoredSourceGroup | None: ...
    def auth_tag_exists(self, source_group_id: UUID, tag: bytes) -> bool: ...
    def commit_generation(self, unit: GenerationUnit) -> tuple[object, bool]: ...
    def append_generation_failure(
        self, *, owner_user_id: int, correlation_id: UUID, reason: str
    ) -> None: ...


class GenerationMedia(Protocol):
    def stage_bytes(
        self,
        *,
        owner_user_id: int,
        variant: str,
        content_type: str,
        content: bytes,
    ) -> StagedMedia: ...
    def discard(self, item: StagedMedia) -> None: ...
    def promote(self, item: StagedMedia) -> None: ...


DecodeRgb = Callable[[bytes, Deadline], np.ndarray]
BuildArtifacts = Callable[[np.ndarray, Deadline], GroupArtifacts]
Embed = Callable[[np.ndarray, bytes, Deadline], EncodedImages]


def canonical_rgb_sha256(rgb: np.ndarray) -> bytes:
    array = np.asarray(rgb)
    if array.dtype != np.dtype("uint8") or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("decoded image must be uint8 RGB")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("decoded image dimensions must be positive")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


class V4GenerationService:
    def __init__(
        self,
        *,
        repository: GenerationRepository,
        key_ring: KeyRing,
        media: GenerationMedia,
        decode_rgb: DecodeRgb,
        build_group_artifacts: BuildArtifacts,
        embed: Embed,
        trace_id_factory: Callable[[], str],
        max_tag_attempts: int = 16,
    ) -> None:
        if max_tag_attempts <= 0:
            raise ValueError("tag attempts must be positive")
        self.repository = repository
        self.key_ring = key_ring
        self.media = media
        self.decode_rgb = decode_rgb
        self.build_group_artifacts = build_group_artifacts
        self.embed = embed
        self.trace_id_factory = trace_id_factory
        self.max_tag_attempts = max_tag_attempts

    def generate(self, request: GenerationRequest, deadline: Deadline) -> GenerationResult:
        correlation_id = uuid4()
        staged: list[StagedMedia] = []
        try:
            deadline.check("generation_decode_input")
            source_rgb = self.decode_rgb(request.content, deadline)
            source_hash = canonical_rgb_sha256(source_rgb)
            existing = self.repository.find_source_group(request.owner_user_id, source_hash)
            artifacts = None if existing is not None else self.build_group_artifacts(source_rgb, deadline)
            model_version = existing.model_version if existing is not None else artifacts.model_version
            feature_version = (
                existing.feature_schema_version if existing is not None else artifacts.feature_schema_version
            )
            provisional_group_id = existing.id if existing is not None else uuid4()

            for _ in range(self.max_tag_attempts):
                deadline.check("generation_allocate_tag")
                trace_id = self.trace_id_factory()
                context = AuthContext(
                    CODEC_ID,
                    self.key_ring.active_key_id,
                    request.owner_user_id,
                    source_hash,
                    trace_id,
                )
                tag = self.key_ring.sign(context)
                if existing is not None and self.repository.auth_tag_exists(existing.id, tag):
                    continue
                encoded = self.embed(source_rgb, tag, deadline)
                watermarked_rgb = self.decode_rgb(encoded.watermarked, deadline)
                canonical_rgb_sha256(watermarked_rgb)
                self.decode_rgb(encoded.thumbnail, deadline)

                staged.append(
                    self.media.stage_bytes(
                        owner_user_id=request.owner_user_id,
                        variant="original",
                        content_type=request.content_type,
                        content=request.content,
                    )
                )
                staged.append(
                    self.media.stage_bytes(
                        owner_user_id=request.owner_user_id,
                        variant="watermarked",
                        content_type="image/png",
                        content=encoded.watermarked,
                    )
                )
                staged.append(
                    self.media.stage_bytes(
                        owner_user_id=request.owner_user_id,
                        variant="thumbnail",
                        content_type="image/png",
                        content=encoded.thumbnail,
                    )
                )
                group = SourceGroupInput(
                    owner_user_id=request.owner_user_id,
                    original_image_sha256=source_hash,
                    image_width=int(source_rgb.shape[1]),
                    image_height=int(source_rgb.shape[0]),
                    original_media_id=staged[0].media_id,
                    model_version=model_version,
                    feature_schema_version=feature_version,
                    view_policy_version=(
                        existing.view_policy_version
                        if existing is not None
                        else artifacts.view_policy_version
                    ),
                )
                record = V4RecordInput(
                    id=uuid4(),
                    source_group_id=provisional_group_id,
                    owner_user_id=request.owner_user_id,
                    trace_id=trace_id,
                    codec=CODEC_ID,
                    auth_tag=tag,
                    key_id=self.key_ring.active_key_id,
                    original_file_md5=hashlib.md5(request.content).digest(),
                    original_file_sha256=hashlib.sha256(request.content).digest(),
                    watermarked_file_md5=hashlib.md5(encoded.watermarked).digest(),
                    watermarked_file_sha256=hashlib.sha256(encoded.watermarked).digest(),
                    original_pixel_sha256=source_hash,
                    watermarked_pixel_sha256=canonical_rgb_sha256(watermarked_rgb),
                    output_media_id=staged[1].media_id,
                    thumbnail_media_id=staged[2].media_id,
                    evidence_uuid=correlation_id,
                    metadata_json=dict(request.metadata),
                )
                unit = GenerationUnit(
                    provisional_group_id,
                    group,
                    artifacts,
                    tuple(staged),
                    record,
                    correlation_id,
                )
                try:
                    stored_record, created = self.repository.commit_generation(unit)
                except AuthTagCollision:
                    self._discard_all(staged)
                    staged = []
                    continue
                for item in staged:
                    self.media.promote(item)
                staged = []
                return GenerationResult(stored_record, created)
            raise RuntimeError("unable to allocate a unique V4 authentication tag")
        except Exception:
            self._discard_all(staged)
            try:
                self.repository.append_generation_failure(
                    owner_user_id=request.owner_user_id,
                    correlation_id=correlation_id,
                    reason="generation_failed",
                )
            except Exception:
                pass
            raise

    def _discard_all(self, staged: list[StagedMedia]) -> None:
        for item in staged:
            try:
                self.media.discard(item)
            except Exception:
                pass


__all__ = (
    "AuthTagCollision",
    "EncodedImages",
    "GenerationRequest",
    "GenerationResult",
    "GenerationUnit",
    "GroupArtifacts",
    "StagedMedia",
    "V4GenerationService",
    "canonical_rgb_sha256",
)
