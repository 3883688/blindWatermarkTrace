from typing import Any

from fastapi import Request

from trace_app.auth.service import AuthService
from trace_app.database.repositories import Repository
from trace_app.management.service import ManagementService
from trace_app.watermark.service import WatermarkService


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def _service_from_state(request: Request, name: str) -> Any:
    factory = getattr(request.app.state, f"{name}_factory", None)
    if factory is not None:
        return factory()
    return getattr(request.app.state, name)


def get_auth_service(request: Request) -> AuthService:
    return _service_from_state(request, "auth_service")


def get_watermark_service(request: Request) -> WatermarkService:
    return _service_from_state(request, "watermark_service")


def get_management_service(request: Request) -> ManagementService:
    return _service_from_state(request, "management_service")
