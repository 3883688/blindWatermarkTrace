"""登录认证接口。

只有一个 ``POST /auth/login``。注意它没有挂 ``/api`` 前缀，是历史遗留的对外契约，
前端与部署层的反向代理规则都依赖这个路径，改动会破坏兼容性。
"""

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
    """账号密码登录，返回令牌与该用户可见的菜单权限。

    用表单（``Form``）而非 JSON 接收凭据，与前端登录页的提交方式保持一致。
    密码校验、失败计数、错误信息统一由 :class:`AuthService` 处理——路由层刻意
    不做任何分支判断，避免鉴权逻辑散落在两处（结构测试也会强制这一点）。
    """
    return service.login(username, password)
