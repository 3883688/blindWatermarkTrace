"""用户与角色管理接口（``/api/roles``、``/api/users``）。

这一组路由全部是 :class:`AuthService` 的薄委托：接住路径参数和 JSON 体，
原样转交，不做校验也不做分支。所有规则——菜单键白名单、角色是否存在、
用户名唯一性——都在服务层，便于被单元测试直接覆盖。
"""

from typing import Any

from fastapi import APIRouter, Depends

from trace_app.auth.service import AuthService
from trace_app.dependencies import get_auth_service

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/roles")
def get_roles(service: AuthService = Depends(get_auth_service)) -> dict[str, Any]:
    """列出全部角色及其菜单授权。"""
    return service.list_roles()


@router.put("/roles/{role_key}")
def update_role(
    role_key: str,
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """更新某个角色的菜单授权。

    ``payload`` 中的未知菜单键会被服务层静默过滤掉，避免前端版本不一致时
    把不存在的菜单写进权限表。
    """
    return service.update_role(role_key, payload)


@router.get("/users")
def get_users(service: AuthService = Depends(get_auth_service)) -> dict[str, Any]:
    """列出全部用户（服务层已剔除密码哈希等敏感列）。"""
    return service.list_users()


@router.post("/users")
def create_user(
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """新建用户；未指定角色时服务层落到默认角色。"""
    return service.create_user(payload)


@router.put("/users/{username}")
def update_user(
    username: str,
    payload: dict[str, Any],
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """按用户名更新用户资料；``payload`` 中不含的字段保持原值。"""
    return service.update_user(username, payload)


@router.delete("/users/{username}")
def delete_user(
    username: str,
    service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """按用户名删除用户；用户不存在时服务层抛 404。"""
    return service.delete_user(username)
