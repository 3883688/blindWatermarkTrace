"""Pinned local model registry and strict V4 inference output contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
from urllib.parse import urlparse

import numpy as np


class ModelRegistryError(RuntimeError):
    """Raised when a model artifact or inference result is not trustworthy."""


REQUIRED_V4_MODELS = frozenset({"dinov2_vits14", "superpoint", "lightglue"})


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str
    shape: tuple[int | None, ...]
    normalization: Mapping[str, tuple[float, ...]] | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    version: str
    framework: str
    weights_path: Path
    sha256: str
    input: TensorContract
    output: TensorContract


def _required(mapping: dict[str, object], name: str) -> object:
    if name not in mapping:
        raise ModelRegistryError(f"missing required model field: {name}")
    return mapping[name]


def _tensor_contract(raw: object, field: str, *, require_normalization: bool) -> TensorContract:
    if not isinstance(raw, dict):
        raise ModelRegistryError(f"{field} metadata must be an object")
    dtype = _required(raw, "dtype")
    shape = _required(raw, "shape")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not shape:
        raise ModelRegistryError(f"invalid {field} dtype/shape metadata")
    if any(size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0) for size in shape):
        raise ModelRegistryError(f"invalid {field} shape metadata")
    normalization = raw.get("normalization")
    if require_normalization and not isinstance(normalization, dict):
        raise ModelRegistryError("input normalization metadata is required")
    normalized = None
    if normalization is not None:
        if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
            raise ModelRegistryError("invalid input normalization metadata")
        mean, std = normalization["mean"], normalization["std"]
        if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != len(std) or not mean:
            raise ModelRegistryError("invalid input normalization metadata")
        normalized = MappingProxyType({"mean": tuple(float(x) for x in mean), "std": tuple(float(x) for x in std)})
    return TensorContract(dtype=dtype, shape=tuple(shape), normalization=normalized)


def _finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ModelRegistryError(f"{name} must be a numeric array")
    if not np.isfinite(array).all():
        raise ModelRegistryError(f"{name} must contain finite values")
    return array


def _validate_contract(value: object, contract: TensorContract, name: str) -> np.ndarray:
    array = _finite_array(value, name)
    if array.dtype.name != contract.dtype:
        raise ModelRegistryError(f"{name} must use {contract.dtype} dtype")
    if array.ndim != len(contract.shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, contract.shape)
    ):
        raise ModelRegistryError(f"{name} does not match pinned shape {contract.shape}")
    return array


def _outputs_equal(left: object, right: object) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(_outputs_equal(a, b) for a, b in zip(left, right))
    return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)


def validate_dino_embeddings(value: object) -> np.ndarray:
    array = _finite_array(value, "DINO embeddings")
    if array.dtype != np.dtype("float32"):
        raise ModelRegistryError("DINO embeddings must use float32 dtype")
    if array.ndim != 2 or array.shape[1] != 384 or array.shape[0] < 1:
        raise ModelRegistryError("DINO embeddings must have shape (n, 384)")
    stable = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(stable, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ModelRegistryError("DINO embeddings must have non-zero norm")
    output = np.ascontiguousarray(stable / norms, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ModelRegistryError("DINO normalization produced non-finite values")
    return output


def validate_superpoint_output(points: object, descriptors: object) -> tuple[np.ndarray, np.ndarray]:
    point_array = _finite_array(points, "SuperPoint points")
    descriptor_array = _finite_array(descriptors, "SuperPoint descriptors")
    if point_array.dtype != np.dtype("float32") or descriptor_array.dtype != np.dtype("float32"):
        raise ModelRegistryError("SuperPoint outputs must use float32 dtype")
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ModelRegistryError("SuperPoint points must have shape (n, 2)")
    if descriptor_array.ndim != 2 or descriptor_array.shape[0] != point_array.shape[0] or descriptor_array.shape[1] < 1:
        raise ModelRegistryError("SuperPoint descriptors must have shape (n, d)")
    return point_array, descriptor_array


def validate_lightglue_output(
    matches: object,
    scores: object,
    *,
    query_count: int,
    train_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    match_array = _finite_array(matches, "LightGlue matches")
    score_array = _finite_array(scores, "LightGlue scores")
    if match_array.dtype != np.dtype("int64") or match_array.ndim != 2 or match_array.shape[1] != 2:
        raise ModelRegistryError("LightGlue matches must be int64 with shape (n, 2)")
    if score_array.dtype != np.dtype("float32"):
        raise ModelRegistryError("LightGlue scores must use float32 dtype")
    if score_array.ndim != 1 or score_array.shape[0] != match_array.shape[0]:
        raise ModelRegistryError("LightGlue scores must have shape (n,)")
    if np.any(score_array < 0) or np.any(score_array > 1):
        raise ModelRegistryError("LightGlue scores must be within [0, 1]")
    if match_array.size and (
        np.any(match_array[:, 0] < 0)
        or np.any(match_array[:, 0] >= query_count)
        or np.any(match_array[:, 1] < 0)
        or np.any(match_array[:, 1] >= train_count)
    ):
        raise ModelRegistryError("LightGlue match index is out of range")
    return match_array, score_array


class ModelRegistry:
    def __init__(self, specs: Mapping[str, ModelSpec]) -> None:
        self._specs = MappingProxyType(dict(specs))
        self._loaded_models: dict[str, Callable[..., object]] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ModelRegistry":
        path = Path(manifest_path).resolve(strict=True)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRegistryError("invalid model manifest") from exc
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ModelRegistryError("unsupported model manifest schema_version")
        models = document.get("models")
        if not isinstance(models, list) or not models:
            raise ModelRegistryError("model manifest must contain models")
        root = path.parent.resolve()
        specs: dict[str, ModelSpec] = {}
        for raw in models:
            if not isinstance(raw, dict):
                raise ModelRegistryError("model entry must be an object")
            name = _required(raw, "name")
            version = _required(raw, "version")
            framework = _required(raw, "framework")
            weights = _required(raw, "weights")
            digest = _required(raw, "sha256")
            input_contract = _tensor_contract(_required(raw, "input"), "input", require_normalization=True)
            output_contract = _tensor_contract(_required(raw, "output"), "output", require_normalization=False)
            if not all(isinstance(item, str) and item for item in (name, version, framework, weights, digest)):
                raise ModelRegistryError("model string fields must be non-empty")
            parsed = urlparse(weights)
            candidate = Path(weights)
            if parsed.scheme or candidate.is_absolute() or candidate.suffix != ".safetensors":
                raise ModelRegistryError("weights must be a relative local .safetensors path")
            try:
                weights_path = (root / candidate).resolve(strict=True)
                weights_path.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ModelRegistryError("weights path is missing or escapes the manifest directory") from exc
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ModelRegistryError("sha256 must be an exact lowercase SHA-256 digest")
            with weights_path.open("rb") as weights_file:
                actual = hashlib.file_digest(weights_file, "sha256").hexdigest()
            if actual != digest:
                raise ModelRegistryError(f"SHA-256 mismatch for model {name}")
            if name in specs:
                raise ModelRegistryError(f"duplicate model name: {name}")
            specs[name] = ModelSpec(name, version, framework, weights_path, digest, input_contract, output_contract)
        if missing := REQUIRED_V4_MODELS - specs.keys():
            raise ModelRegistryError(f"required V4 models are missing: {sorted(missing)}")
        return cls(specs)

    def require(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ModelRegistryError(f"required model is unavailable: {name}") from exc

    @staticmethod
    def _read_verified_artifact(spec: ModelSpec) -> bytes:
        try:
            artifact = spec.weights_path.read_bytes()
        except OSError as exc:
            raise ModelRegistryError(f"model weights are unavailable: {spec.name}") from exc
        actual = hashlib.sha256(artifact).hexdigest()
        if actual != spec.sha256:
            raise ModelRegistryError(f"SHA-256 mismatch for model {spec.name}")
        return artifact

    def load_model(
        self,
        name: str,
        module: object,
        *,
        state_loader: Callable[[bytes], object] | None = None,
    ) -> object:
        spec = self.require(name)
        artifact = self._read_verified_artifact(spec)
        if state_loader is None:
            try:
                from safetensors.torch import load
            except ImportError as exc:
                raise ModelRegistryError("torch and safetensors are required for model loading") from exc
            state = load(artifact)
        else:
            state = state_loader(artifact)
        try:
            module.load_state_dict(state, strict=True)
            module.eval()
        except (AttributeError, RuntimeError) as exc:
            raise ModelRegistryError(f"model state is incompatible: {name}") from exc
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        if not callable(module):
            raise ModelRegistryError(f"loaded model is not callable: {name}")
        self._loaded_models[name] = module
        self._ready = False
        return module

    def infer(self, name: str, *args: object, **kwargs: object) -> object:
        if not self._ready:
            raise ModelRegistryError("model registry is not ready")
        return self._infer_loaded(name, *args, **kwargs)

    def _infer_loaded(self, name: str, *args: object, **kwargs: object) -> object:
        spec = self.require(name)
        try:
            module = self._loaded_models[name]
        except KeyError as exc:
            raise ModelRegistryError(f"model is not loaded: {name}") from exc
        if name == "lightglue":
            if len(args) != 2 or kwargs:
                raise ModelRegistryError("LightGlue requires query and train descriptors")
            query = _validate_contract(args[0], spec.input, "LightGlue query descriptors")
            train = _validate_contract(args[1], spec.input, "LightGlue train descriptors")
        else:
            if len(args) != 1 or kwargs:
                raise ModelRegistryError(f"{name} requires exactly one input tensor")
            _validate_contract(args[0], spec.input, f"{name} input")
        try:
            import torch
        except ImportError as exc:
            raise ModelRegistryError("torch is required for model inference") from exc
        with torch.inference_mode():
            output = module(*args, **kwargs)
        if name == "dinov2_vits14":
            _validate_contract(output, spec.output, "DINO output")
            return validate_dino_embeddings(output)
        if name == "superpoint":
            if not isinstance(output, tuple) or len(output) != 2:
                raise ModelRegistryError("SuperPoint must return points and descriptors")
            points, descriptors = validate_superpoint_output(*output)
            _validate_contract(descriptors, spec.output, "SuperPoint descriptors")
            return points, descriptors
        if name == "lightglue":
            if not isinstance(output, tuple) or len(output) != 2:
                raise ModelRegistryError("LightGlue must return matches and scores")
            matches, scores = validate_lightglue_output(
                *output, query_count=query.shape[0], train_count=train.shape[0]
            )
            _validate_contract(matches, spec.output, "LightGlue matches")
            return matches, scores
        raise ModelRegistryError(f"unsupported V4 model: {name}")

    def run_smoke_checks(self, inputs: Mapping[str, tuple[object, ...]]) -> None:
        if self._loaded_models.keys() != self._specs.keys():
            raise ModelRegistryError("every pinned model must be loaded before smoke checks")
        if inputs.keys() != self._specs.keys():
            raise ModelRegistryError("smoke inputs must cover every pinned model")
        for name in sorted(inputs):
            first = self._infer_loaded(name, *inputs[name])
            second = self._infer_loaded(name, *inputs[name])
            if not _outputs_equal(first, second):
                raise ModelRegistryError(f"model smoke inference is not deterministic: {name}")
        self._ready = True


__all__ = (
    "ModelRegistry",
    "ModelRegistryError",
    "ModelSpec",
    "TensorContract",
    "REQUIRED_V4_MODELS",
    "validate_dino_embeddings",
    "validate_lightglue_output",
    "validate_superpoint_output",
)
