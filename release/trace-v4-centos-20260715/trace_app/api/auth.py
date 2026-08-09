from typing import Any

from fastapi import APIRouter, Depends, Form

from trace_app.auth.service import AuthService
from trace_app.dependencies import get_auth_service

router = APIRouter(tags=["auth"])


@router.post("/auth/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    return service.login(username, password)
