from __future__ import annotations

import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trace_app.api import auth, dashboard, images, users, watermark
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.auth.service import AuthService
from trace_app.config import (
    DEFAULT_WATERMARK_AUTH_KEY,
    Settings,
    settings as default_settings,
)
from trace_app.database.connection import create_runtime
from trace_app.database.repositories import Repository
from trace_app.dependencies import get_optional_current_user, get_repository
from trace_app.management.service import ManagementService
from trace_app.media import (
    derive_media_signing_key,
    resolve_media_path,
    user_can_access_media,
    verify_media_signature,
)
from trace_app.runtime import dispose_engine, dispose_runtime
from trace_app.watermark.service import WatermarkService
from trace_app.watermark.default_operations import build_default_operations

ServiceFactory = Callable[[], object]


def running_pytest() -> bool:
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None


def ensure_directories(settings: Settings) -> None:
    settings.original_dir.mkdir(parents=True, exist_ok=True)
    settings.watermarked_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnail_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    try:
        yield
    finally:
        dispose_runtime(app.state.runtime)


def _parse_bool(raw: str | bool | None) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").lower() in {"1", "true", "yes", "on", "启用"}


def _env_bool(name: str, default: str, legacy_name: str | None = None) -> bool:
    value = os.getenv(name)
    if value is None and legacy_name:
        value = os.getenv(legacy_name)
    return _parse_bool(default if value is None else value)


def _clamp_float(
    value: str | float | None, default: float, low: float, high: float
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _initialize_detection_state(app: FastAPI) -> None:
    app.state.visible_watermark_detection_enabled = _env_bool(
        "ENABLE_VISIBLE_WATERMARK_DETECTION",
        "false",
        "VISIBLE_WATERMARK_DETECTION_ENABLED",
    )
    app.state.visual_match_fallback_enabled = _env_bool(
        "ENABLE_VISUAL_MATCH_FALLBACK", "false", "VISUAL_MATCH_FALLBACK_ENABLED"
    )
    app.state.small_crop_trace_default_enabled = _env_bool(
        "ENABLE_SMALL_CROP_TRACE_REDUNDANCY", "true"
    )
    app.state.aligned_authenticated_detection_enabled = _env_bool(
        "ENABLE_ALIGNED_AUTHENTICATED_DETECTION", "true"
    )
    app.state.dense_watermark_fallback_enabled = _env_bool(
        "ENABLE_DENSE_WATERMARK_FALLBACK", "false"
    )
    try:
        app.state.aligned_candidate_limit = max(
            1, min(32, int(os.getenv("ALIGNED_CANDIDATE_LIMIT", "8")))
        )
    except ValueError:
        app.state.aligned_candidate_limit = 8
    app.state.watermark_detection_budget_seconds = _clamp_float(
        os.getenv("WATERMARK_DETECTION_BUDGET_SECONDS", "5"), 5.0, 0.1, 60.0
    )


def register_static_routes(app: FastAPI, settings: Settings) -> None:
    mimetypes.add_type("font/woff2", ".woff2")
    mimetypes.add_type("font/woff", ".woff")
    mimetypes.add_type("font/ttf", ".ttf")
    app.mount(
        "/assets",
        StaticFiles(directory=str(settings.base_dir / "assets"), check_dir=False),
        name="assets",
    )

    @app.get("/uploads/{media_path:path}", name="uploads")
    def upload_file(
        media_path: str,
        expires: str | None = None,
        signature: str | None = None,
        current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
        repository: Repository = Depends(get_repository),
    ) -> FileResponse:
        media_url = f"/uploads/{media_path}"
        path = resolve_media_path(settings.upload_dir, media_path)
        has_signature = expires is not None or signature is not None
        if has_signature:
            if not verify_media_signature(
                media_url,
                expires=expires,
                signature=signature,
                key=app.state.media_signing_key,
            ):
                raise HTTPException(status_code=403, detail="图片访问链接无效或已过期")
        elif current_user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        elif not user_can_access_media(repository, current_user, media_url):
            raise HTTPException(status_code=404, detail="图片不存在")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(
            path,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(settings.base_dir / "index.html")

    @app.get("/site-logo.png")
    def site_logo() -> FileResponse:
        return FileResponse(settings.base_dir / "site-logo.png")

    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(settings.base_dir / "favicon.ico")

    @app.get("/favico.ico")
    def favico() -> FileResponse:
        return FileResponse(settings.base_dir / "favico.ico")


def create_app(
    *,
    settings: Settings = default_settings,
    initialize_database: bool | None = None,
    auth_service_factory: ServiceFactory | None = None,
    watermark_service_factory: ServiceFactory | None = None,
    management_service_factory: ServiceFactory | None = None,
) -> FastAPI:
    ensure_directories(settings)
    enabled = not running_pytest() if initialize_database is None else initialize_database
    runtime = create_runtime(settings, enabled=enabled)
    repository = Repository(
        runtime.store, ensure_dirs=lambda: ensure_directories(settings)
    )

    app = FastAPI(title=settings.app_name, lifespan=application_lifespan)
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.repository = repository
    app.state.media_signing_key = derive_media_signing_key(
        DEFAULT_WATERMARK_AUTH_KEY,
        runtime.media_signing_key,
    )
    try:
        app.state.media_url_ttl_seconds = max(
            30,
            min(3600, int(os.getenv("MEDIA_URL_TTL_SECONDS", "300"))),
        )
    except ValueError:
        app.state.media_url_ttl_seconds = 300
    app.state.generated_trace_ids = runtime.generated_trace_ids
    app.state.auth_service = AuthService(
        repository,
        sessions=runtime.auth_sessions,
    )
    app.state.watermark_service = WatermarkService(
        settings=settings,
        repository=repository,
        runtime=runtime,
        operations=build_default_operations(
            settings=settings,
            repository=repository,
            runtime=runtime,
            state_value=lambda name: getattr(app.state, name),
            ensure_directories=lambda: ensure_directories(settings),
        ),
    )
    app.state.management_service = ManagementService(
        settings=settings,
        repository=repository,
        runtime=runtime,
        ensure_directories=lambda: ensure_directories(settings),
        database_enabled=enabled,
    )
    if auth_service_factory is not None:
        app.state.auth_service_factory = auth_service_factory
    if watermark_service_factory is not None:
        app.state.watermark_service_factory = watermark_service_factory
    if management_service_factory is not None:
        app.state.management_service_factory = management_service_factory
    _initialize_detection_state(app)
    register_static_routes(app, settings)
    for api_router in (
        auth.router,
        users.router,
        watermark.router,
        images.router,
        dashboard.router,
    ):
        app.include_router(api_router)
    return app
