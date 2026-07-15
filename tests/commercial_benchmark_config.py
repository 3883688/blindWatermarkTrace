import os


def build_embedding_form(user_id: str, fidelity_level: str | float) -> dict[str, str]:
    return {
        "user_id": user_id,
        "mode": "dct",
        "fidelity_level": str(fidelity_level),
        "small_crop_trace_enabled": "true",
        "small_crop_trace_strength": os.getenv("SMALL_CROP_TRACE_STRENGTH", "0.35"),
        "small_crop_trace_density": os.getenv("SMALL_CROP_TRACE_DENSITY", "medium"),
        "robust_watermark_strength": os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0"),
        "robust_watermark_version": os.getenv("ROBUST_WATERMARK_VERSION", "1"),
        "dot_matrix_trace_enabled": "false",
        "copyright_enabled": "false",
    }
