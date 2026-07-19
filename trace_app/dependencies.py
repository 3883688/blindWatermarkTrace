from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.auth.service import AuthService
from trace_app.database.repositories import Repository
from trace_app.management.service import ManagementService
from trace_app.watermark.service import WatermarkService


bearer_auth = HTTPBearer(auto_error=False)


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


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_auth),
    service: AuthService = Depends(get_auth_service),
) -> AuthenticatedUser | None:
    if credentials is None:
        return None
    return service.resolve_token(credentials.credentials)


def get_current_user(
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> AuthenticatedUser:
    if current_user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return current_user
