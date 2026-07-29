"""Non-executable, bounded serialization for V4 model features."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors


MAGIC = b"V4FT"
ENVELOPE_VERSION = 1
_HEADER = struct.Struct(">4sBIQ32s")
_ALLOWED_DTYPES = frozenset(
    {"bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64"}
)


class FeatureEnvelopeError(ValueError):
    """Raised when a feature envelope is unsafe or violates its contract."""


@dataclass(frozen=True, slots=True)
class FeatureLimits:
    max_metadata_bytes: int = 64 * 1024
    max_payload_bytes: int = 32 * 1024 * 1024
    max_arrays: int = 8
    max_rank: int = 8
    max_array_elements: int = 5_000_000

    def __post_init__(self) -> None:
        for name in self.__slots__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


_FEATURE_SCHEMAS: Mapping[str, Mapping[str, tuple[str, tuple[int | None, ...]]]] = MappingProxyType(
    {
        "dino_embedding": MappingProxyType({"embedding": ("float32", (None, 384))}),
        "superpoint": MappingProxyType(
            {
                "points": ("float32", (None, 2)),
                "descriptors": ("float32", (None, 256)),
            }
        ),
        "orb": MappingProxyType(
            {
                "points": ("float32", (None, 2)),
                "descriptors": ("uint8", (None, 32)),
            }
        ),
    }
)


def _feature_schema(feature_kind: str) -> Mapping[str, tuple[str, tuple[int | None, ...]]]:
    try:
        return _FEATURE_SCHEMAS[feature_kind]
    except (KeyError, TypeError) as exc:
        raise FeatureEnvelopeError("unknown feature kind") from exc


def _validate_fixed_schema(
    feature_kind: str, arrays: Mapping[str, np.ndarray]
) -> None:
    schema = _feature_schema(feature_kind)
    if arrays.keys() != schema.keys():
        raise FeatureEnvelopeError(f"{feature_kind} array names do not match fixed schema")
    leading_counts: set[int] = set()
    for name, (dtype, shape) in schema.items():
        array = arrays[name]
        if array.dtype.name != dtype:
            raise FeatureEnvelopeError(f"array {name!r} dtype does not match fixed schema")
        if array.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(array.shape, shape)):
            raise FeatureEnvelopeError(f"array {name!r} shape does not match fixed schema")
        if shape and shape[0] is None:
            leading_counts.add(array.shape[0])
    if len(leading_counts) > 1:
        raise FeatureEnvelopeError(f"{feature_kind} arrays must share their leading count")


def _validate_array(name: str, value: np.ndarray, limits: FeatureLimits) -> np.ndarray:
    if not isinstance(name, str) or not name or len(name) > 128:
        raise FeatureEnvelopeError("array name must be a non-empty bounded string")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FeatureEnvelopeError(f"array {name!r} uses forbidden object dtype")
    dtype = array.dtype.name
    if dtype not in _ALLOWED_DTYPES:
        raise FeatureEnvelopeError(f"array {name!r} uses unsupported dtype {dtype}")
    if array.ndim > limits.max_rank:
        raise FeatureEnvelopeError(f"array {name!r} rank exceeds limit")
    if array.size > limits.max_array_elements:
        raise FeatureEnvelopeError(f"array {name!r} element count exceeds limit")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise FeatureEnvelopeError(f"array {name!r} must contain finite values")
    return np.ascontiguousarray(array)


def serialize_features(
    arrays: Mapping[str, np.ndarray],
    *,
    feature_kind: str,
    schema_version: int,
    model_version: str,
    limits: FeatureLimits | None = None,
) -> bytes:
    active_limits = limits or FeatureLimits()
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version <= 0:
        raise FeatureEnvelopeError("schema_version must be a positive integer")
    if not isinstance(model_version, str) or not model_version or len(model_version) > 128:
        raise FeatureEnvelopeError("model_version must be a non-empty bounded string")
    if not arrays or len(arrays) > active_limits.max_arrays:
        raise FeatureEnvelopeError("array count is outside allowed limits")

    normalized = {name: _validate_array(name, value, active_limits) for name, value in arrays.items()}
    _validate_fixed_schema(feature_kind, normalized)
    metadata = {
        "schema_version": schema_version,
        "model_version": model_version,
        "feature_kind": feature_kind,
        "format": "safetensors",
        "compression": "none",
        "arrays": {
            name: {"dtype": array.dtype.name, "shape": list(array.shape), "count": int(array.size)}
            for name, array in sorted(normalized.items())
        },
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > active_limits.max_metadata_bytes:
        raise FeatureEnvelopeError("metadata length exceeds limit")
    payload = save_safetensors(normalized)
    if len(payload) > active_limits.max_payload_bytes:
        raise FeatureEnvelopeError("payload length exceeds limit")
    checksum = hashlib.sha256(payload).digest()
    return _HEADER.pack(MAGIC, ENVELOPE_VERSION, len(metadata_bytes), len(payload), checksum) + metadata_bytes + payload


def _parse_metadata(raw: bytes) -> dict[str, object]:
    try:
        metadata = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureEnvelopeError("invalid canonical metadata") from exc
    if not isinstance(metadata, dict) or _canonical_json(metadata) != raw:
        raise FeatureEnvelopeError("metadata is not canonical JSON")
    return metadata


def _declared_arrays(metadata: dict[str, object], limits: FeatureLimits) -> dict[str, dict[str, object]]:
    if metadata.get("format") != "safetensors" or metadata.get("compression") != "none":
        raise FeatureEnvelopeError("only uncompressed safetensors payloads are accepted")
    declared = metadata.get("arrays")
    if not isinstance(declared, dict) or not declared or len(declared) > limits.max_arrays:
        raise FeatureEnvelopeError("array count is outside allowed limits")
    checked: dict[str, dict[str, object]] = {}
    for name, item in declared.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise FeatureEnvelopeError("invalid array metadata")
        dtype, shape, count = item.get("dtype"), item.get("shape"), item.get("count")
        if dtype not in _ALLOWED_DTYPES:
            raise FeatureEnvelopeError(f"array {name!r} has unsupported dtype")
        if not isinstance(shape, list) or len(shape) > limits.max_rank:
            raise FeatureEnvelopeError(f"array {name!r} has invalid rank")
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise FeatureEnvelopeError(f"array {name!r} has invalid shape")
        computed = math.prod(shape)
        if isinstance(count, bool) or not isinstance(count, int) or count != computed:
            raise FeatureEnvelopeError(f"array {name!r} has invalid element count")
        if count > limits.max_array_elements:
            raise FeatureEnvelopeError(f"array {name!r} element count exceeds limit")
        checked[name] = item
    return checked


def deserialize_features(
    envelope: bytes,
    *,
    expected_feature_kind: str,
    expected_schema_version: int,
    expected_model_version: str,
    limits: FeatureLimits | None = None,
) -> Mapping[str, np.ndarray]:
    active_limits = limits or FeatureLimits()
    if not isinstance(envelope, bytes) or len(envelope) < _HEADER.size:
        raise FeatureEnvelopeError("truncated feature envelope")
    magic, version, metadata_length, payload_length, checksum = _HEADER.unpack_from(envelope)
    if magic != MAGIC or version != ENVELOPE_VERSION:
        raise FeatureEnvelopeError("unsupported feature envelope")
    if metadata_length > active_limits.max_metadata_bytes:
        raise FeatureEnvelopeError("metadata length exceeds limit")
    if payload_length > active_limits.max_payload_bytes:
        raise FeatureEnvelopeError("payload length exceeds limit")
    expected_length = _HEADER.size + metadata_length + payload_length
    if len(envelope) != expected_length:
        raise FeatureEnvelopeError("feature envelope length mismatch")

    metadata_end = _HEADER.size + metadata_length
    metadata = _parse_metadata(envelope[_HEADER.size:metadata_end])
    _feature_schema(expected_feature_kind)
    if metadata.get("feature_kind") != expected_feature_kind:
        raise FeatureEnvelopeError("feature kind mismatch")
    if metadata.get("schema_version") != expected_schema_version:
        raise FeatureEnvelopeError("schema version mismatch")
    if metadata.get("model_version") != expected_model_version:
        raise FeatureEnvelopeError("model version mismatch")
    declared = _declared_arrays(metadata, active_limits)
    payload = envelope[metadata_end:]
    if not hashlib.sha256(payload).digest() == checksum:
        raise FeatureEnvelopeError("payload checksum mismatch")
    try:
        arrays = load_safetensors(payload)
    except Exception as exc:
        raise FeatureEnvelopeError("invalid safetensors payload") from exc
    if arrays.keys() != declared.keys():
        raise FeatureEnvelopeError("payload array names do not match metadata")
    for name, array in arrays.items():
        item = declared[name]
        expected_shape = tuple(item["shape"])
        if array.dtype.name != item["dtype"] or array.shape != expected_shape or array.size != item["count"]:
            raise FeatureEnvelopeError(f"array {name!r} does not match declared dtype/shape/count")
        _validate_array(name, array, active_limits)
    _validate_fixed_schema(expected_feature_kind, arrays)
    return MappingProxyType(arrays)


__all__ = (
    "ENVELOPE_VERSION",
    "FeatureEnvelopeError",
    "FeatureLimits",
    "deserialize_features",
    "serialize_features",
)
