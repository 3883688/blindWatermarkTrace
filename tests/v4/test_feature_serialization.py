from __future__ import annotations

import struct

import numpy as np
import pytest

from trace_app.v4.features import (
    FeatureEnvelopeError,
    FeatureLimits,
    deserialize_features,
    serialize_features,
)


def test_feature_envelope_round_trips_without_pickle_or_compression() -> None:
    arrays = {
        "embedding": np.arange(384, dtype=np.float32).reshape(1, 384),
    }

    encoded = serialize_features(
        arrays, feature_kind="dino_embedding", schema_version=1, model_version="dino-test"
    )
    decoded = deserialize_features(
        encoded,
        expected_feature_kind="dino_embedding",
        expected_schema_version=1,
        expected_model_version="dino-test",
    )

    assert encoded.startswith(b"V4FT")
    assert b"pickle" not in encoded.lower()
    np.testing.assert_array_equal(decoded["embedding"], arrays["embedding"])


def test_serializer_rejects_object_and_non_finite_arrays() -> None:
    with pytest.raises(FeatureEnvelopeError, match="object"):
        serialize_features(
            {"embedding": np.array([object()], dtype=object)},
            feature_kind="dino_embedding",
            schema_version=1,
            model_version="v1",
        )
    with pytest.raises(FeatureEnvelopeError, match="finite"):
        serialize_features(
            {"embedding": np.full((1, 384), np.inf, dtype=np.float32)},
            feature_kind="dino_embedding",
            schema_version=1,
            model_version="v1",
        )
    with pytest.raises(FeatureEnvelopeError, match="dtype"):
        serialize_features(
            {"embedding": np.ones((1, 384), dtype=np.float64)},
            feature_kind="dino_embedding",
            schema_version=1,
            model_version="v1",
        )


def test_deserializer_rejects_schema_or_model_mismatch() -> None:
    encoded = serialize_features(
        {"embedding": np.ones((1, 384), dtype=np.float32)},
        feature_kind="dino_embedding",
        schema_version=1,
        model_version="v1",
    )
    with pytest.raises(FeatureEnvelopeError, match="schema"):
        deserialize_features(encoded, expected_feature_kind="dino_embedding", expected_schema_version=2, expected_model_version="v1")
    with pytest.raises(FeatureEnvelopeError, match="model"):
        deserialize_features(encoded, expected_feature_kind="dino_embedding", expected_schema_version=1, expected_model_version="v2")

    with pytest.raises(FeatureEnvelopeError, match="feature kind"):
        deserialize_features(encoded, expected_feature_kind="superpoint", expected_schema_version=1, expected_model_version="v1")


def test_deserializer_rejects_checksum_corruption() -> None:
    encoded = bytearray(
        serialize_features(
            {"embedding": np.ones((1, 384), dtype=np.float32)},
            feature_kind="dino_embedding",
            schema_version=1,
            model_version="v1",
        )
    )
    encoded[-1] ^= 0x01
    with pytest.raises(FeatureEnvelopeError, match="checksum"):
        deserialize_features(bytes(encoded), expected_feature_kind="dino_embedding", expected_schema_version=1, expected_model_version="v1")


def test_deserializer_rejects_declared_length_before_allocation() -> None:
    encoded = bytearray(
        serialize_features(
            {"embedding": np.ones((1, 384), dtype=np.float32)},
            feature_kind="dino_embedding",
            schema_version=1,
            model_version="v1",
        )
    )
    # Fixed header: magic, envelope version, metadata length, payload length, checksum.
    struct.pack_into(">Q", encoded, 9, 2**63)
    with pytest.raises(FeatureEnvelopeError, match="payload length"):
        deserialize_features(
            bytes(encoded),
            expected_feature_kind="dino_embedding",
            expected_schema_version=1,
            expected_model_version="v1",
            limits=FeatureLimits(max_payload_bytes=1024),
        )


def test_array_shape_count_and_fixed_schema_are_checked() -> None:
    encoded = serialize_features(
        {"embedding": np.ones((1, 384), dtype=np.float32)},
        feature_kind="dino_embedding",
        schema_version=1,
        model_version="v1",
    )
    with pytest.raises(FeatureEnvelopeError, match="element count"):
        deserialize_features(
            encoded,
            expected_feature_kind="dino_embedding",
            expected_schema_version=1,
            expected_model_version="v1",
            limits=FeatureLimits(max_array_elements=128),
        )


def test_trailing_bytes_and_truncated_envelopes_are_rejected() -> None:
    encoded = serialize_features(
        {"embedding": np.ones((1, 384), dtype=np.float32)},
        feature_kind="dino_embedding",
        schema_version=1,
        model_version="v1",
    )
    for damaged in (encoded + b"trailing", encoded[:-1]):
        with pytest.raises(FeatureEnvelopeError):
            deserialize_features(damaged, expected_feature_kind="dino_embedding", expected_schema_version=1, expected_model_version="v1")


def test_unknown_feature_kind_and_self_declared_layout_are_rejected() -> None:
    with pytest.raises(FeatureEnvelopeError, match="feature kind"):
        serialize_features(
            {"x": np.ones((1,), dtype=np.float32)},
            feature_kind="attacker_defined",
            schema_version=1,
            model_version="v1",
        )
