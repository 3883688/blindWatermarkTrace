import inspect
from typing import Any, Callable

from fastapi import Request

from trace_app.auth.service import AuthService
from trace_app.database.repositories import Repository
from trace_app.management.service import ManagementService
from trace_app.watermark.service import WatermarkService


def _override(request: Request, dependency: Callable[..., Any]) -> Any | None:
    replacement = request.app.dependency_overrides.get(dependency)
    if replacement is None:
        return None
    if len(inspect.signature(replacement).parameters) == 0:
        return replacement()
    return replacement(request)


def get_repository(request: Request) -> Repository:
    overridden = _override(request, get_repository)
    if overridden is not None:
        return overridden
    return request.app.state.repository


def _service_from_state(request: Request, name: str) -> Any:
    factory = getattr(request.app.state, f"{name}_factory", None)
    if factory is not None:
        return factory()
    return getattr(request.app.state, name)


def get_auth_service(request: Request) -> AuthService:
    overridden = _override(request, get_auth_service)
    if overridden is not None:
        return overridden
    return _service_from_state(request, "auth_service")


def get_watermark_service(request: Request) -> WatermarkService:
    overridden = _override(request, get_watermark_service)
    if overridden is not None:
        return overridden
    return _service_from_state(request, "watermark_service")


def get_management_service(request: Request) -> ManagementService:
    overridden = _override(request, get_management_service)
    if overridden is not None:
        return overridden
    return _service_from_state(request, "management_service")
