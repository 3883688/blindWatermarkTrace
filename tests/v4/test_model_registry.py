from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from trace_app.v4.models import (
    ModelRegistry,
    ModelRegistryError,
    validate_dino_embeddings,
    validate_lightglue_output,
    validate_superpoint_output,
)


def _model_entry(tmp_path: Path, name: str) -> dict[str, object]:
    weights = tmp_path / f"{name}.safetensors"
    weights.write_bytes(f"local {name} placeholder".encode("ascii"))
    if name == "lightglue":
        input_shape, output_shape = [None, 256], [None, 2]
    elif name == "superpoint":
        input_shape, output_shape = [None, 3, 224, 224], [None, 256]
    else:
        input_shape, output_shape = [None, 3, 224, 224], [None, 384]
    return {
        "name": name,
        "version": "pinned-test",
        "framework": "torch",
        "weights": weights.name,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "input": {
            "dtype": "float32",
            "shape": input_shape,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "output": {"dtype": "float32" if name != "lightglue" else "int64", "shape": output_shape},
    }


def _write_manifest(tmp_path: Path) -> Path:
    manifest = {
        "schema_version": 1,
        "models": [_model_entry(tmp_path, name) for name in ("dinov2_vits14", "superpoint", "lightglue")],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_registry_accepts_verified_local_safetensors(tmp_path: Path) -> None:
    registry = ModelRegistry.from_manifest(_write_manifest(tmp_path))

    spec = registry.require("dinov2_vits14")

    assert spec.weights_path.name == "dinov2_vits14.safetensors"
    assert spec.output.shape == (None, 384)
    assert registry.ready is False


@pytest.mark.parametrize("missing", ["version", "framework", "input", "output"])
def test_registry_requires_complete_model_metadata(tmp_path: Path, missing: str) -> None:
    path = _write_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["models"][0][missing]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelRegistryError, match=missing):
        ModelRegistry.from_manifest(path)


def test_registry_fails_closed_when_weights_change(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    (tmp_path / "dinov2_vits14.safetensors").write_bytes(b"replaced")

    with pytest.raises(ModelRegistryError, match="SHA-256"):
        ModelRegistry.from_manifest(path)


def test_registry_requires_all_v4_models(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["models"] = document["models"][:-1]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="lightglue"):
        ModelRegistry.from_manifest(path)


@pytest.mark.parametrize("weights", ["https://example.invalid/dino.safetensors", "../outside.safetensors", "dino.pkl"])
def test_registry_rejects_remote_escaping_or_pickle_weights(
    tmp_path: Path, weights: str
) -> None:
    path = _write_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["models"][0]["weights"] = weights
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelRegistryError):
        ModelRegistry.from_manifest(path)


def test_dino_embeddings_are_finite_normalized_and_exact_shape() -> None:
    embeddings = validate_dino_embeddings(np.ones((3, 384), dtype=np.float32))
    assert embeddings.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    with pytest.raises(ModelRegistryError):
        validate_dino_embeddings(np.ones((3, 383), dtype=np.float32))
    with pytest.raises(ModelRegistryError):
        validate_dino_embeddings(np.full((1, 384), np.nan, dtype=np.float32))
    with pytest.raises(ModelRegistryError, match="float32"):
        validate_dino_embeddings(np.ones((1, 384), dtype=np.float64))

    large = validate_dino_embeddings(
        np.full((1, 384), np.finfo(np.float32).max, dtype=np.float32)
    )
    np.testing.assert_allclose(np.linalg.norm(large, axis=1), 1.0, atol=1e-6)


def test_feature_matcher_outputs_are_strictly_validated() -> None:
    points = np.zeros((8, 2), dtype=np.float32)
    descriptors = np.zeros((8, 256), dtype=np.float32)
    validate_superpoint_output(points, descriptors)
    with pytest.raises(ModelRegistryError):
        validate_superpoint_output(points, descriptors[:-1])
    with pytest.raises(ModelRegistryError, match="float32"):
        validate_superpoint_output(points.astype(np.float64), descriptors)

    matches = np.array([[0, 1], [2, 3]], dtype=np.int64)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    validate_lightglue_output(matches, scores, query_count=4, train_count=5)
    with pytest.raises(ModelRegistryError):
        validate_lightglue_output(matches + 9, scores, query_count=4, train_count=5)
    with pytest.raises(ModelRegistryError, match="float32"):
        validate_lightglue_output(matches, scores.astype(np.float64), query_count=4, train_count=5)


def test_registry_is_ready_only_after_deterministic_smoke_check(tmp_path: Path) -> None:
    registry = ModelRegistry.from_manifest(_write_manifest(tmp_path))
    with pytest.raises(ModelRegistryError, match="loaded"):
        registry.run_smoke_checks({})
    assert registry.ready is False


def test_registry_becomes_ready_after_all_loaded_models_pass_smoke_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry.from_manifest(_write_manifest(tmp_path))
    _install_fake_torch(monkeypatch, {"active": False})
    registry.load_model(
        "dinov2_vits14",
        _FakeModel(lambda batch: np.ones((len(batch), 384), dtype=np.float32)),
        state_loader=lambda _: {},
    )
    registry.load_model(
        "superpoint",
        _FakeModel(
            lambda batch: (
                np.zeros((8, 2), dtype=np.float32),
                np.zeros((8, 256), dtype=np.float32),
            )
        ),
        state_loader=lambda _: {},
    )
    registry.load_model(
        "lightglue",
        _FakeModel(
            lambda query, train: (
                np.array([[0, 0]], dtype=np.int64),
                np.array([0.9], dtype=np.float32),
            )
        ),
        state_loader=lambda _: {},
    )
    registry.run_smoke_checks(
        {
            "dinov2_vits14": (np.ones((1, 3, 224, 224), dtype=np.float32),),
            "superpoint": (np.ones((1, 3, 224, 224), dtype=np.float32),),
            "lightglue": (
                np.ones((2, 256), dtype=np.float32),
                np.ones((3, 256), dtype=np.float32),
            ),
        }
    )

    assert registry.ready is True


class _FakeModel:
    def __init__(self, output: Callable[..., object]) -> None:
        self.output = output
        self.evaluating = False

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        assert state == {}
        assert strict is True

    def eval(self) -> None:
        self.evaluating = True

    def parameters(self) -> tuple[object, ...]:
        return ()

    def __call__(self, *args: object) -> object:
        assert self.evaluating
        return self.output(*args)


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, state: dict[str, bool]) -> None:
    @contextmanager
    def inference_mode():
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=inference_mode))


def test_registry_rejects_public_inference_before_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry.from_manifest(_write_manifest(tmp_path))
    state = {"active": False}
    _install_fake_torch(monkeypatch, state)
    model = _FakeModel(lambda batch: np.ones((len(batch), 384), dtype=np.float32))
    registry.load_model("dinov2_vits14", model, state_loader=lambda _: {})

    with pytest.raises(ModelRegistryError, match="not ready"):
        registry.infer("dinov2_vits14", np.ones((2, 3, 224, 224), dtype=np.float32))


def test_model_load_rechecks_weight_hash_to_close_toc_tou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry.from_manifest(_write_manifest(tmp_path))
    registry.require("dinov2_vits14").weights_path.write_bytes(b"replaced after registration")
    _install_fake_torch(monkeypatch, {"active": False})

    with pytest.raises(ModelRegistryError, match="SHA-256"):
        registry.load_model("dinov2_vits14", _FakeModel(lambda _: None), state_loader=lambda _: {})
