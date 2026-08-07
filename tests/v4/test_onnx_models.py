from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trace_app.v4.deadlines import Deadline
from trace_app.v4.models import ModelRegistryError
from trace_app.v4.onnx_models import DinoOnnxModels, LightGlueOnnxMatcher


class _Node:
    def __init__(self, name: str, shape: list[object], node_type: str) -> None:
        self.name = name
        self.shape = shape
        self.type = node_type


def _artifact(tmp_path: Path, name: str, content: bytes = b"onnx") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    path.with_suffix(path.suffix + ".sha256").write_text(
        sha256(content).hexdigest() + "  " + name + "\n", encoding="ascii"
    )
    return path


class _DinoSession:
    def get_inputs(self):
        return [_Node("pixel_values", ["batch_size", 3, 224, 224], "tensor(float)")]

    def get_outputs(self):
        return [_Node("last_hidden_state", ["batch_size", 257, 384], "tensor(float)")]

    def run(self, _outputs, inputs):
        batch = inputs["pixel_values"]
        tokens = np.zeros((len(batch), 257, 384), dtype=np.float32)
        tokens[:, 0, :2] = (3.0, 4.0)
        return [tokens]


class _LightGlueSession:
    def get_inputs(self):
        return [_Node("images", ["batch_size", 1, "height", "width"], "tensor(float)")]

    def get_outputs(self):
        return [
            _Node("keypoints", ["batch_size", 1024, 2], "tensor(int64)"),
            _Node("matches", ["matches", 3], "tensor(int64)"),
            _Node("mscores", ["matches"], "tensor(float)"),
        ]

    def run(self, _outputs, _inputs):
        points0 = np.asarray(
            [[20 * x, 20 * y] for y in range(5) for x in range(6)], dtype=np.int64
        )
        points1 = points0 + np.asarray([8, 12], dtype=np.int64)
        keypoints = np.stack((points0, points1))
        matches = np.asarray(
            [[0, index, index] for index in range(len(points0))], dtype=np.int64
        )
        scores = np.full((len(matches),), 0.95, dtype=np.float32)
        return [keypoints, matches, scores]


def test_dino_onnx_returns_normalized_cls_embeddings(tmp_path: Path) -> None:
    models = DinoOnnxModels(
        _artifact(tmp_path, "dino.onnx"), session_factory=lambda _path: _DinoSession()
    )

    result = models.infer(
        "dinov2_vits14", np.ones((2, 3, 224, 224), dtype=np.float32)
    )

    assert result.shape == (2, 384)
    np.testing.assert_allclose(result[:, :2], [[0.6, 0.8], [0.6, 0.8]])


def test_joint_lightglue_onnx_returns_geometry_evidence(tmp_path: Path) -> None:
    matcher = LightGlueOnnxMatcher(
        _artifact(tmp_path, "lightglue.onnx"),
        session_factory=lambda _path: _LightGlueSession(),
    )

    evidence = matcher.match_geometry(
        Image.new("RGB", (256, 256)),
        Image.new("RGB", (256, 256)),
        Deadline.after(10),
    )

    assert evidence is not None
    assert evidence.method == "lightglue"
    assert evidence.inliers == 30
    assert evidence.ratio == 1.0


def test_onnx_models_reject_checksum_mismatch(tmp_path: Path) -> None:
    path = _artifact(tmp_path, "dino.onnx")
    path.write_bytes(b"changed")

    with pytest.raises(ModelRegistryError, match="SHA-256"):
        DinoOnnxModels(path, session_factory=lambda _path: _DinoSession())
