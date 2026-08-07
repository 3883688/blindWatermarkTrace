from typing import Any

from fastapi import Depends, Form, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.auth.service import AuthService
from trace_app.database.repositories import Repository
from trace_app.management.service import ManagementService
from trace_app.watermark.service import WatermarkService
from trace_app.v4.domain import OwnerScope
from trace_app.v4.media import V4MediaService
from trace_app.v4.security import RateLimitExceeded


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


def enforce_login_rate_limit(
    request: Request,
    username: str = Form(...),
) -> None:
    limiter = getattr(request.app.state, "login_rate_limiter", None)
    if limiter is None:
        return
    client_ip = "unknown" if request.client is None else request.client.host
    try:
        limiter.consume(username=username, client_ip=client_ip)
    except RateLimitExceeded as error:
        raise HTTPException(
            status_code=429, detail="登录尝试过多，请稍后再试"
        ) from error


def get_watermark_service(request: Request) -> WatermarkService:
    return _service_from_state(request, "watermark_service")


def get_management_service(request: Request) -> ManagementService:
    return _service_from_state(request, "management_service")


def get_v4_media_service(request: Request) -> V4MediaService:
    service = _service_from_state(request, "v4_media_service")
    if service is None:
        raise HTTPException(status_code=503, detail="媒体服务不可用")
    return service


def _required_v4_service(request: Request, name: str) -> Any:
    service = _service_from_state(request, name)
    if service is None:
        raise HTTPException(status_code=503, detail="V4 服务不可用")
    return service


def get_v4_generation_service(request: Request) -> Any:
    return _required_v4_service(request, "v4_generation_service")


def get_v4_detection_service(request: Request) -> Any:
    return _required_v4_service(request, "v4_detection_service")


def get_v4_record_repository(request: Request) -> Any:
    return _required_v4_service(request, "v4_record_repository")


def get_v4_job_service(request: Request) -> Any:
    return _required_v4_service(request, "v4_job_service")


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


def require_admin(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


def resolve_owner_scope(
    cross_owner: bool = False,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> OwnerScope:
    if cross_owner and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return OwnerScope(current_user.id, cross_owner=cross_owner)
