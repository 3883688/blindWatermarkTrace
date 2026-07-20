from typing import Any

from fastapi import APIRouter, Depends

from trace_app.auth.service import AuthService
from trace_app.dependencies import get_auth_service

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/roles")
def get_roles(service: AuthService = Depends(get_auth_service)) -> dict[str, Any]:
    return service.list_roles()


@router.put("/roles/{role_key}")
def update_role(
    role_key: str,
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    return service.update_role(role_key, payload)


@router.get("/users")
def get_users(service: AuthService = Depends(get_auth_service)) -> dict[str, Any]:
    return service.list_users()


@router.post("/users")
def create_user(
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    return service.create_user(payload)


@router.put("/users/{username}")
def update_user(
    username: str,
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    return service.update_user(username, payload)


@router.delete("/users/{username}")
def delete_user(
    username: str,
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    return service.delete_user(username)
