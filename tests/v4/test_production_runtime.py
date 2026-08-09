from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import trace_app.v4.production as production_module

from trace_app.v4.deadlines import Deadline
from trace_app.v4.features import deserialize_features
from trace_app.v4.production import (
    VisibleCopyrightConfig,
    build_group_artifacts,
    create_production_services,
    decode_rgb,
    encode_v4_images,
    extract_aligned_observation,
)
from trace_app.v4.detection import V4DetectionService
from trace_app.v4.generation import V4GenerationService
from trace_app.v4.geometry import ConfirmedGroup
from trace_app.v4.keys import KeyRing
from trace_app.v4.region_protection import ProtectedRegion
from watermark_v4.payload import encode_codeword


class _Dino:
    def infer(self, name: str, batch: object) -> np.ndarray:
        assert name == "dinov2_vits14"
        count = len(np.asarray(batch))
        vectors = np.zeros((count, 384), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


class _LightGlue:
    def match_geometry(self, query, target, deadline):
        return None


def _png(size: tuple[int, int] = (384, 384)) -> bytes:
    y, x = np.indices((size[1], size[0]))
    array = np.stack(
        ((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 11) % 256),
        axis=2,
    ).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def test_cpu_runtime_decodes_and_embeds_v4_codeword() -> None:
    deadline = Deadline.after(30)
    rgb = decode_rgb(_png(), deadline)

    encoded = encode_v4_images(rgb, b"12345678", deadline)

    assert rgb.dtype == np.uint8 and rgb.shape == (384, 384, 3)
    assert Image.open(BytesIO(encoded.watermarked)).format == "PNG"
    assert Image.open(BytesIO(encoded.thumbnail)).format == "PNG"
    assert encoded.watermarked != _png()


def test_cpu_runtime_can_enable_configured_visible_copyright_layer() -> None:
    deadline = Deadline.after(30)
    rgb = decode_rgb(_png(), deadline)
    encoded = encode_v4_images(
        rgb,
        b"12345678",
        deadline,
        visible_copyright=VisibleCopyrightConfig(
            enabled=True,
            text="© 3883688",
            prominent_corner=True,
        ),
    )
    marked = decode_rgb(encoded.watermarked, deadline)
    assert np.count_nonzero(marked != rgb) > np.count_nonzero(
        decode_rgb(
            encode_v4_images(rgb, b"12345678", deadline).watermarked,
            deadline,
        )
        != rgb
    )


def test_region_enhancement_only_replaces_detected_complete_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deadline = Deadline.after(30)
    rgb = decode_rgb(_png(), deadline)
    monkeypatch.setattr(
        production_module,
        "detect_protected_regions",
        lambda _rgb: (ProtectedRegion("face", 20, 20, 80, 80, 0.9),),
    )

    baseline = decode_rgb(
        encode_v4_images(rgb, b"12345678", deadline).watermarked,
        deadline,
    )
    reinforced = decode_rgb(
        encode_v4_images(
            rgb,
            b"12345678",
            deadline,
            protected_region_enhancement=True,
        ).watermarked,
        deadline,
    )

    difference = np.any(baseline != reinforced, axis=2)
    assert np.any(difference[:128, :128])
    assert not np.any(difference[128:, :])
    assert not np.any(difference[:128, 128:])


def test_group_artifacts_contain_dino_orb_and_superpoint_rows() -> None:
    rgb = decode_rgb(_png(), Deadline.after(30))

    artifacts = build_group_artifacts(rgb, Deadline.after(30), _Dino())

    assert len(artifacts.embeddings) >= 1
    assert {item.feature_kind for item in artifacts.features} == {"orb", "superpoint"}
    for item in artifacts.features:
        arrays = deserialize_features(
            item.feature_bytes,
            expected_feature_kind=item.feature_kind,
            expected_schema_version=1,
            expected_model_version=item.model_version,
        )
        assert set(arrays) == {"points", "descriptors"}


def test_aligned_v4_observation_recovers_the_embedded_rs_codeword() -> None:
    deadline = Deadline.after(30)
    rgb = decode_rgb(_png(), deadline)
    tag = b"12345678"
    encoded = encode_v4_images(rgb, tag, deadline)
    marked = decode_rgb(encoded.watermarked, deadline)
    confirmed = ConfirmedGroup(
        source_group_id=__import__("uuid").UUID(int=1),
        homography=np.eye(3, dtype=np.float64),
        method="orb_ransac",
        inliers=100,
        ratio=1.0,
        reprojection_error=0.0,
    )

    observation = extract_aligned_observation(
        marked,
        confirmed,
        target_size=(rgb.shape[1], rgb.shape[0]),
        deadline=deadline,
    )

    assert observation is not None
    assert observation.observed_codeword == encode_codeword(tag)


def test_production_services_wire_v4_generation_and_detection_only() -> None:
    services = create_production_services(
        repository=object(),
        media=object(),
        key_ring=KeyRing({"active": b"k" * 32}, "active"),
        dino_models=_Dino(),
        lightglue_matcher=_LightGlue(),
    )

    assert isinstance(services.generation, V4GenerationService)
    assert isinstance(services.detection, V4DetectionService)


def test_runtime_requirements_exclude_legacy_and_test_dependencies() -> None:
    requirements = (__import__("pathlib").Path(__file__).resolve().parents[2] / "requirements.txt").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in ("torch", "nvidia", "pytest", "httpx", "pymysql", "pywavelets"):
        assert forbidden not in requirements
