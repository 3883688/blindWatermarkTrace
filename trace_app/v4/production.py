"""CPU-only production primitives for the V4 generation pipeline."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Mapping
from uuid import UUID

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from trace_app.v4.deadlines import Deadline
from trace_app.v4.detection import V4DetectionService
from trace_app.v4.domain import OwnerScope
from trace_app.v4.features import deserialize_features, serialize_features
from trace_app.v4.generation import EncodedImages, GroupArtifacts, V4GenerationService
from trace_app.v4.geometry import ConfirmedGroup, FeatureSet, match_orb_ransac
from trace_app.v4.keys import KeyRing
from trace_app.v4.recall import VIEW_POLICY_VERSION, build_dino_batch, recall_image
from trace_app.v4.region_protection import detect_protected_regions, reinforced_tiles
from trace_app.imaging.visible_mark import apply_visible_copyright
from trace_app.v4.repository import EmbeddingInput, FeatureInput
from watermark_v4.config import V4Config
from watermark_v4.dct import embed_codeword, extract_image_tiles
from watermark_v4.observation import V4Observation, extract_observation
from watermark_v4.payload import encode_codeword
from watermark_v4.sync import embed_pilot


DINO_MODEL_VERSION = "dinov2_vits14"
ORB_MODEL_VERSION = "opencv_orb_v1"
SUPERPOINT_MODEL_VERSION = "superpoint_lightglue_v1"
FEATURE_SCHEMA_VERSION = 1
THUMBNAIL_MAX_SIDE = 512
DEFAULT_WATERMARKED_JPEG_QUALITY = 80


@dataclass(frozen=True, slots=True)
class VisibleCopyrightConfig:
    enabled: bool = False
    text: str = "© QQ:757675150"
    opacity: float = 0.16
    complexity: str = "medium"
    irregular: bool = True
    prominent_corner: bool = False


def _parse_bool(value: object, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "启用"}


def _pilot_amplitude_from_metadata(metadata: Mapping[str, object]) -> float:
    try:
        value = float(metadata.get("pilot_amplitude", 0.75))
    except (TypeError, ValueError):
        return 0.75
    if not np.isfinite(value):
        return 0.75
    return max(0.25, min(1.25, value))


def _output_quality_from_metadata(metadata: Mapping[str, object]) -> int:
    try:
        value = int(str(metadata.get("output_quality", "80")).strip())
    except (TypeError, ValueError):
        return DEFAULT_WATERMARKED_JPEG_QUALITY
    return max(60, min(95, value))


def visible_copyright_from_metadata(
    default: VisibleCopyrightConfig,
    metadata: Mapping[str, object],
) -> VisibleCopyrightConfig:
    """Apply per-request visible-watermark overrides to deployment defaults."""
    try:
        opacity = float(metadata.get("copyright_opacity", default.opacity))
    except (TypeError, ValueError):
        opacity = default.opacity
    return VisibleCopyrightConfig(
        enabled=_parse_bool(metadata.get("copyright_enabled"), default.enabled),
        text=str(metadata.get("copyright_text") or default.text).strip() or default.text,
        opacity=max(0.02, min(0.90, opacity)),
        complexity=str(metadata.get("copyright_complexity") or default.complexity),
        irregular=_parse_bool(
            metadata.get("copyright_irregular_enabled"), default.irregular
        ),
        prominent_corner=_parse_bool(
            metadata.get("copyright_prominent_corner_enabled"),
            default.prominent_corner,
        ),
    )


@dataclass(frozen=True, slots=True)
class ProductionServices:
    generation: V4GenerationService
    detection: V4DetectionService


def decode_rgb(content: bytes, deadline: Deadline) -> np.ndarray:
    """Decode untrusted image bytes into a contiguous uint8 RGB array."""
    if not isinstance(content, bytes) or not content:
        raise ValueError("image content must be non-empty bytes")
    deadline.check("decode_open")
    try:
        with Image.open(BytesIO(content)) as source:
            source.load()
            deadline.check("decode_pixels")
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("unsupported or corrupt image") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError("decoded image must be non-empty RGB")
    deadline.check("decode_complete")
    return np.ascontiguousarray(rgb)


def encode_v4_images(
    rgb: np.ndarray,
    tag: bytes,
    deadline: Deadline,
    *,
    visible_copyright: VisibleCopyrightConfig | None = None,
    protected_region_enhancement: bool = False,
    pilot_amplitude: float = 0.75,
    output_quality: int = DEFAULT_WATERMARKED_JPEG_QUALITY,
) -> EncodedImages:
    """Embed a V4 HMAC64 tag and return a compact JPEG plus a PNG thumbnail."""
    array = np.asarray(rgb)
    if array.dtype != np.dtype("uint8") or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("V4 embedding input must be uint8 RGB")
    if type(tag) is not bytes or len(tag) != 8:
        raise ValueError("V4 authentication tag must contain exactly 8 bytes")
    if type(output_quality) is not int or not 60 <= output_quality <= 95:
        raise ValueError("JPEG output quality must be an integer from 60 through 95")
    deadline.check("embed_prepare")
    config = V4Config(pilot_amplitude=pilot_amplitude)
    regions = detect_protected_regions(array) if protected_region_enhancement else ()
    source = Image.fromarray(np.ascontiguousarray(array))
    if visible_copyright is not None:
        source = apply_visible_copyright(
            source,
            visible_copyright.enabled,
            visible_copyright.text,
            visible_copyright.opacity,
            visible_copyright.complexity,
            visible_copyright.irregular,
            visible_copyright.prominent_corner,
        )
    pilot_source = embed_pilot(source, config)
    codeword = encode_codeword(tag)
    marked = embed_codeword(pilot_source, codeword, config)
    if regions:
        selected_tiles = reinforced_tiles(
            regions,
            image_width=source.width,
            image_height=source.height,
            tile_size=config.tile_size,
        )
        if selected_tiles:
            reinforced_config = replace(
                config,
                dct_margin=min(config.dct_margin_range[1], config.dct_margin + 2.0),
            )
            reinforced = embed_codeword(
                pilot_source,
                codeword,
                reinforced_config,
                tile_coordinates=selected_tiles,
            )
            marked_array = np.asarray(marked).copy()
            reinforced_array = np.asarray(reinforced)
            for tile_x, tile_y in selected_tiles:
                left = tile_x * config.tile_size
                top = tile_y * config.tile_size
                marked_array[
                    top : top + config.tile_size,
                    left : left + config.tile_size,
                ] = reinforced_array[
                    top : top + config.tile_size,
                    left : left + config.tile_size,
                ]
            marked = Image.fromarray(marked_array)
    deadline.check("embed_complete")

    output = BytesIO()
    marked.save(
        output,
        format="JPEG",
        quality=output_quality,
        subsampling=0,
        optimize=True,
        progressive=True,
    )
    thumbnail = marked.copy()
    thumbnail.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE), Image.Resampling.LANCZOS)
    thumbnail_output = BytesIO()
    thumbnail.save(thumbnail_output, format="PNG", optimize=False)
    deadline.check("embed_encode")
    return EncodedImages(
        output.getvalue(), thumbnail_output.getvalue(), "image/jpeg"
    )


def build_group_artifacts(
    rgb: np.ndarray,
    deadline: Deadline,
    dino_models: object,
) -> GroupArtifacts:
    """Build normalized DINO views and bounded geometry feature envelopes."""
    array = np.asarray(rgb)
    if array.dtype != np.dtype("uint8") or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("artifact input must be uint8 RGB")
    image = Image.fromarray(np.ascontiguousarray(array))

    deadline.check("artifacts_dino_prepare")
    batch, boxes = build_dino_batch(image)
    vectors = np.asarray(dino_models.infer(DINO_MODEL_VERSION, batch), dtype=np.float32)
    if vectors.shape != (len(boxes), 384) or not np.isfinite(vectors).all():
        raise ValueError("DINO inference returned invalid embeddings")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("DINO inference returned a zero embedding")
    vectors = np.ascontiguousarray(vectors / norms.astype(np.float32))
    embeddings = tuple(
        EmbeddingInput(index, box.kind, vector.tolist(), DINO_MODEL_VERSION)
        for index, (box, vector) in enumerate(zip(boxes, vectors, strict=True))
    )

    deadline.check("artifacts_orb")
    orb = _orb_feature_set(array)
    superpoint_points = np.empty((0, 2), dtype=np.float32)
    superpoint_descriptors = np.empty((0, 256), dtype=np.float32)
    features = (
        _feature("orb", ORB_MODEL_VERSION, orb.points, orb.descriptors),
        _feature(
            "superpoint",
            SUPERPOINT_MODEL_VERSION,
            superpoint_points,
            superpoint_descriptors,
        ),
    )
    deadline.check("artifacts_complete")
    return GroupArtifacts(
        embeddings=embeddings,
        features=features,
        model_version=DINO_MODEL_VERSION,
        feature_schema_version=str(FEATURE_SCHEMA_VERSION),
        view_policy_version=VIEW_POLICY_VERSION,
    )


def create_production_services(
    *,
    repository: object,
    media: object,
    key_ring: KeyRing,
    dino_models: object,
    lightglue_matcher: object,
    visible_copyright: VisibleCopyrightConfig | None = None,
) -> ProductionServices:
    """Wire the verified CPU models into the V4 service contracts."""
    confirmer = _GeometryConfirmer(repository, media, lightglue_matcher)

    def recall(scope: OwnerScope, rgb: np.ndarray, deadline: Deadline) -> tuple[UUID, ...]:
        candidates = recall_image(
            scope,
            Image.fromarray(np.ascontiguousarray(rgb)),
            dino_models,
            repository,
            deadline,
        )
        confirmer.set_scope(scope)
        return tuple(item.source_group_id for item in candidates)

    def observe(rgb: np.ndarray, confirmed: ConfirmedGroup, deadline: Deadline):
        group = repository.get_source_group(confirmer.scope, confirmed.source_group_id)
        if group is None:
            return None
        return extract_aligned_observation(
            rgb,
            confirmed,
            target_size=(group.image_width, group.image_height),
            deadline=deadline,
        )

    default_visible_copyright = visible_copyright or VisibleCopyrightConfig()

    def embed_with_metadata(rgb, tag, deadline, metadata):
        return encode_v4_images(
            rgb,
            tag,
            deadline,
            visible_copyright=visible_copyright_from_metadata(
                default_visible_copyright, metadata
            ),
            protected_region_enhancement=_parse_bool(
                metadata.get("protected_region_enhancement"), False
            ),
            pilot_amplitude=_pilot_amplitude_from_metadata(metadata),
            output_quality=_output_quality_from_metadata(metadata),
        )

    generation = V4GenerationService(
        repository=repository,
        key_ring=key_ring,
        media=media,
        decode_rgb=decode_rgb,
        build_group_artifacts=lambda rgb, deadline: build_group_artifacts(
            rgb, deadline, dino_models
        ),
        embed=lambda rgb, tag, deadline: encode_v4_images(
            rgb,
            tag,
            deadline,
            visible_copyright=visible_copyright,
        ),
        embed_with_metadata=embed_with_metadata,
        trace_id_factory=lambda: secrets.token_urlsafe(24),
    )
    detection = V4DetectionService(
        repository=repository,
        key_ring=key_ring,
        decode_rgb=decode_rgb,
        recall_groups=recall,
        confirm_group=confirmer,
        extract_observation=observe,
    )
    return ProductionServices(generation, detection)


class _GeometryConfirmer:
    def __init__(self, repository: object, media: object, lightglue: object) -> None:
        self.repository = repository
        self.media = media
        self.lightglue = lightglue
        self.scope = OwnerScope(1)
        self._query_identity: int | None = None
        self._query_orb: FeatureSet | None = None

    def set_scope(self, scope: OwnerScope) -> None:
        self.scope = scope

    def __call__(
        self, query_rgb: np.ndarray, source_group_id: UUID, deadline: Deadline
    ) -> ConfirmedGroup | None:
        query = self._query_features(query_rgb)
        stored = self.repository.get_features_for_groups(self.scope, (source_group_id,))
        orb_row = next((item for item in stored if item.feature_kind == "orb"), None)
        if orb_row is None:
            return None
        target = _stored_feature_set(orb_row)
        deadline.check("geometry_orb")
        evidence = match_orb_ransac(query, target)
        if evidence is None:
            evidence = self._lightglue(query_rgb, source_group_id, deadline)
        if evidence is None:
            return None
        return ConfirmedGroup(
            source_group_id,
            evidence.homography,
            evidence.method,
            evidence.inliers,
            evidence.ratio,
            evidence.reprojection_error,
        )

    def _query_features(self, rgb: np.ndarray) -> FeatureSet:
        identity = id(rgb)
        if identity != self._query_identity or self._query_orb is None:
            self._query_identity = identity
            self._query_orb = _orb_feature_set(rgb)
        return self._query_orb

    def _lightglue(self, query_rgb: np.ndarray, source_group_id: UUID, deadline: Deadline):
        group = self.repository.get_source_group(self.scope, source_group_id)
        if group is None or not group.original_media_id:
            return None
        media_object = self.repository.get_media(group.original_media_id)
        if media_object is None:
            return None
        path = self.media.resolve_storage_key(media_object.storage_key)
        try:
            with Image.open(path) as source:
                target = source.convert("RGB").copy()
        except OSError:
            return None
        return self.lightglue.match_geometry(
            Image.fromarray(np.ascontiguousarray(query_rgb)), target, deadline
        )


def _orb_feature_set(rgb: np.ndarray) -> FeatureSet:
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        points = np.empty((0, 2), dtype=np.float32)
        descriptors = np.empty((0, 32), dtype=np.uint8)
    else:
        points = np.asarray([item.pt for item in keypoints], dtype=np.float32)
        descriptors = np.ascontiguousarray(descriptors, dtype=np.uint8)
    return FeatureSet(points, descriptors, int(rgb.shape[1]), int(rgb.shape[0]), "orb")


def _stored_feature_set(row: object) -> FeatureSet:
    arrays = deserialize_features(
        row.feature_bytes,
        expected_feature_kind=row.feature_kind,
        expected_schema_version=int(row.schema_version),
        expected_model_version=row.model_version,
    )
    return FeatureSet(
        arrays["points"],
        arrays["descriptors"],
        row.image_width,
        row.image_height,
        row.feature_kind,
    )


def extract_aligned_observation(
    query_rgb: np.ndarray,
    confirmed: ConfirmedGroup,
    *,
    target_size: tuple[int, int],
    deadline: Deadline,
) -> V4Observation | None:
    """Warp a confirmed query into source coordinates and read its V4 codeword."""
    query = np.asarray(query_rgb)
    target_width, target_height = target_size
    if query.dtype != np.dtype("uint8") or query.ndim != 3 or query.shape[2] != 3:
        raise ValueError("observation input must be uint8 RGB")
    if min(target_width, target_height) <= 0:
        raise ValueError("observation target dimensions must be positive")
    deadline.check("observation_warp")
    warped = cv2.warpPerspective(
        query,
        np.asarray(confirmed.homography, dtype=np.float64),
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpPerspective(
        np.full(query.shape[:2], 255, dtype=np.uint8),
        np.asarray(confirmed.homography, dtype=np.float64),
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    deadline.check("observation_extract")
    config = V4Config()
    tiles = extract_image_tiles(Image.fromarray(warped), config)
    selected = []
    coverages = []
    for tile in tiles:
        top = tile.tile_y * config.tile_size
        left = tile.tile_x * config.tile_size
        coverage = float(
            np.mean(mask[top : top + config.tile_size, left : left + config.tile_size] > 0)
        )
        if coverage >= config.minimum_coverage:
            selected.append(tile)
            coverages.append(coverage)
    deadline.check("observation_aggregate")
    if len(selected) < config.minimum_tiles:
        return None
    return extract_observation(
        tuple(selected),
        coverages=tuple(coverages),
        minimum_tiles_per_class=max(1, config.minimum_tiles // 2),
        minimum_phases=config.minimum_phases,
        minimum_coverage=float(config.minimum_coverage),
    )


def _feature(
    kind: str,
    model_version: str,
    points: np.ndarray,
    descriptors: np.ndarray,
) -> FeatureInput:
    payload = serialize_features(
        {"points": points, "descriptors": descriptors},
        feature_kind=kind,
        schema_version=FEATURE_SCHEMA_VERSION,
        model_version=model_version,
    )
    return FeatureInput(
        feature_kind=kind,
        schema_version=str(FEATURE_SCHEMA_VERSION),
        model_version=model_version,
        feature_bytes=payload,
        feature_sha256=hashlib.sha256(payload).digest(),
    )


__all__ = (
    "build_group_artifacts",
    "create_production_services",
    "decode_rgb",
    "encode_v4_images",
    "VisibleCopyrightConfig",
    "visible_copyright_from_metadata",
    "extract_aligned_observation",
    "ProductionServices",
)
