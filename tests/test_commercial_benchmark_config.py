import importlib


def test_embedding_form_includes_independent_robust_strength(monkeypatch):
    monkeypatch.setenv("SMALL_CROP_TRACE_STRENGTH", "0.35")
    monkeypatch.setenv("SMALL_CROP_TRACE_DENSITY", "medium")
    monkeypatch.setenv("ROBUST_WATERMARK_STRENGTH", "1.0")
    monkeypatch.setenv("ROBUST_WATERMARK_VERSION", "2")

    config = importlib.import_module("tests.commercial_benchmark_config")
    form = config.build_embedding_form("benchmark-user", "1.0")

    assert form == {
        "user_id": "benchmark-user",
        "mode": "dct",
        "fidelity_level": "1.0",
        "small_crop_trace_enabled": "true",
        "small_crop_trace_strength": "0.35",
        "small_crop_trace_density": "medium",
        "robust_watermark_strength": "1.0",
        "robust_watermark_version": "2",
        "dot_matrix_trace_enabled": "false",
        "copyright_enabled": "false",
    }


def test_embedding_form_defaults_to_quality_candidate(monkeypatch):
    monkeypatch.delenv("SMALL_CROP_TRACE_STRENGTH", raising=False)
    monkeypatch.delenv("SMALL_CROP_TRACE_DENSITY", raising=False)
    monkeypatch.delenv("ROBUST_WATERMARK_STRENGTH", raising=False)
    monkeypatch.delenv("ROBUST_WATERMARK_VERSION", raising=False)

    config = importlib.import_module("tests.commercial_benchmark_config")
    form = config.build_embedding_form("benchmark-user", 1.0)

    assert form["small_crop_trace_strength"] == "0.35"
    assert form["small_crop_trace_density"] == "medium"
    assert form["robust_watermark_strength"] == "1.0"
    assert form["robust_watermark_version"] == "1"
