from pathlib import Path

import main


EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/site-logo.png"),
    ("GET", "/favicon.ico"),
    ("GET", "/favico.ico"),
    ("POST", "/auth/login"),
    ("GET", "/api/roles"),
    ("PUT", "/api/roles/{role_key}"),
    ("GET", "/api/users"),
    ("POST", "/api/users"),
    ("PUT", "/api/users/{username}"),
    ("DELETE", "/api/users/{username}"),
    ("POST", "/api/watermark/embed"),
    ("POST", "/api/watermark/extract"),
    ("POST", "/api/watermark/extract-url"),
    ("GET", "/api/dashboard-stats"),
    ("GET", "/api/images"),
    ("DELETE", "/api/images/{image_id}"),
    ("POST", "/api/dev/reset"),
}


def test_main_exposes_expected_routes() -> None:
    actual_routes = {
        (method, route.path)
        for route in main.app.routes
        for method in getattr(route, "methods", set())
    }

    assert EXPECTED_ROUTES <= actual_routes


def test_main_exposes_required_python_api() -> None:
    required = {
        "app",
        "ensure_dirs",
        "embed_robust_watermark",
        "embed_robust_watermark_v2",
        "embed_robust_watermark_v3",
        "detect_aligned_authenticated_watermark",
        "extract_watermark_from_image",
        "align_query_to_record",
        "file_sha256",
    }

    assert required <= set(dir(main))


def test_main_still_contains_watermark_endpoint_implementation() -> None:
    source = Path("main.py").read_text(encoding="utf-8")

    assert "def embed_watermark(" in source
