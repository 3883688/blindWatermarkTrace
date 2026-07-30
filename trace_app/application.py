from __future__ import annotations

import hashlib
import hmac
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trace_app.api import auth, compat_v4, media, users, v4
from trace_app.auth.service import AuthService
from trace_app.config import (
    DEFAULT_WATERMARK_AUTH_KEY,
    Settings,
    settings as default_settings,
)
from trace_app.database.connection import create_runtime
from trace_app.database.repositories import Repository
from trace_app.management.service import ManagementService
from trace_app.media import derive_media_signing_key
from trace_app.runtime import dispose_engine, dispose_runtime
from trace_app.v4.media import V4MediaService
from trace_app.v4.jobs import DeepJobStore
from trace_app.v4.keys import KeyRing
from trace_app.v4.onnx_models import DinoOnnxModels, LightGlueOnnxMatcher
from trace_app.v4.production import create_production_services
from trace_app.v4.repository import V4Repository
from trace_app.v4.security import DatabaseSessionStore, LoginRateLimiter

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


def register_static_routes(app: FastAPI, settings: Settings) -> None:
    mimetypes.add_type("font/woff2", ".woff2")
    mimetypes.add_type("font/woff", ".woff")
    mimetypes.add_type("font/ttf", ".ttf")
    app.mount(
        "/assets",
        StaticFiles(directory=str(settings.base_dir / "assets"), check_dir=False),
        name="assets",
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
    v4_generation_service_factory: ServiceFactory | None = None,
    v4_detection_service_factory: ServiceFactory | None = None,
    v4_record_repository_factory: ServiceFactory | None = None,
    v4_capabilities_factory: ServiceFactory | None = None,
    v4_media_service_factory: ServiceFactory | None = None,
    v4_remote_fetch_factory: ServiceFactory | None = None,
    v4_job_service_factory: ServiceFactory | None = None,
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
    app.state.v4_media_service = None
    app.state.v4_generation_service = None
    app.state.v4_detection_service = None
    app.state.v4_record_repository = None
    app.state.v4_job_service = None
    app.state.v4_capabilities = {"dinov2": False, "lightglue": False}
    if runtime.engine is not None:
        v4_media_key = hmac.new(
            app.state.media_signing_key,
            b"trace-v4-opaque-media-url-v1",
            hashlib.sha256,
        ).digest()
        app.state.v4_record_repository = V4Repository(runtime.engine)
        app.state.v4_job_service = DeepJobStore(runtime.engine)
        app.state.v4_media_service = V4MediaService(
            app.state.v4_record_repository,
            storage_root=settings.upload_dir,
            signing_key=v4_media_key,
            public_base_url=settings.media_public_base_url,
            default_ttl_seconds=app.state.media_url_ttl_seconds,
        )
        model_root = settings.v4_model_manifest_path.parent
        model_paths = (
            model_root / "dinov2-small.onnx",
            model_root / "superpoint_lightglue_pipeline.onnx",
        )
        if settings.environment == "production" or all(path.is_file() for path in model_paths):
            if not DEFAULT_WATERMARK_AUTH_KEY:
                raise RuntimeError("WATERMARK_AUTH_KEY is required for V4 production")
            secret = hashlib.sha256(DEFAULT_WATERMARK_AUTH_KEY.encode("utf-8")).digest()
            key_id = "v4-" + hashlib.sha256(secret).hexdigest()[:16]
            dino_models = DinoOnnxModels(model_paths[0])
            lightglue = LightGlueOnnxMatcher(
                model_paths[1]
            )
            services = create_production_services(
                repository=app.state.v4_record_repository,
                media=app.state.v4_media_service,
                key_ring=KeyRing({key_id: secret}, key_id),
                dino_models=dino_models,
                lightglue_matcher=lightglue,
            )
            app.state.v4_generation_service = services.generation
            app.state.v4_detection_service = services.detection
            app.state.v4_capabilities = {"dinov2": True, "lightglue": True}
    session_store = (
        None if runtime.engine is None else DatabaseSessionStore(runtime.engine)
    )
    rate_limiter = None if runtime.engine is None else LoginRateLimiter(runtime.engine)
    app.state.login_rate_limiter = rate_limiter
    app.state.auth_service = AuthService(
        repository,
        session_store=session_store,
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
    for name, factory in (
        ("v4_generation_service", v4_generation_service_factory),
        ("v4_detection_service", v4_detection_service_factory),
        ("v4_record_repository", v4_record_repository_factory),
        ("v4_capabilities", v4_capabilities_factory),
        ("v4_media_service", v4_media_service_factory),
        ("v4_remote_fetch", v4_remote_fetch_factory),
        ("v4_job_service", v4_job_service_factory),
    ):
        if factory is not None:
            setattr(app.state, f"{name}_factory", factory)
    register_static_routes(app, settings)
    for api_router in (auth.router, users.router, media.router, v4.router, compat_v4.router):
        app.include_router(api_router)
    return app
