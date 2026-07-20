import uuid
from typing import Any

from fastapi import HTTPException

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import MENU_LABELS
from trace_app.database.repositories import Repository


class AuthService:
    def __init__(
        self,
        repository: Repository | None = None,
        *,
        sessions: dict[str, int] | None = None,
    ) -> None:
        self.repository = repository
        self.sessions = {} if sessions is None else sessions

    def _require_repository(self) -> Repository:
        if self.repository is None:
            raise HTTPException(status_code=503, detail="数据库不可用")
        return self.repository

    def public_users(self, users: dict[str, Any]) -> dict[str, dict[str, str]]:
        return {
            username: {"role": str(info.get("role") or "operator")}
            for username, info in users.items()
        }

    def allowed_menu_keys(self, menus: Any) -> list[str]:
        if not isinstance(menus, list):
            return []
        return [key for key in menus if key in MENU_LABELS]

    def role_for_username(self, username: str) -> str:
        users = self._require_repository().read_users()["users"]
        return str(users.get(username, {}).get("role") or "operator")

    def login(self, username: str, password: str) -> dict[str, Any]:
        repository = self._require_repository()
        identity = repository.authenticate_user(username, password)
        if identity is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        role = str(identity["role"])
        roles = repository.read_roles()["roles"]
        menus = self.allowed_menu_keys(roles.get(role, {}).get("menus", []))
        token = f"local-{uuid.uuid4().hex}"
        self.sessions[token] = int(identity["id"])
        return {
            "token": token,
            "username": str(identity["username"]),
            "role": role,
            "menus": menus,
        }

    def resolve_token(self, token: str) -> AuthenticatedUser:
        user_id = self.sessions.get(token)
        identity = (
            None
            if user_id is None
            else self._require_repository().get_user_by_id(user_id)
        )
        if identity is None:
            self.sessions.pop(token, None)
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        return AuthenticatedUser(
            id=int(identity["id"]),
            username=str(identity["username"]),
            role=str(identity["role"]),
        )

    def list_roles(self) -> dict[str, Any]:
        roles = self._require_repository().read_roles()["roles"]
        return {"menus": MENU_LABELS, "roles": roles}

    def update_role(self, role_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        repository = self._require_repository()
        roles = repository.read_roles()["roles"]
        if role_key not in roles:
            raise HTTPException(status_code=404, detail="角色不存在")
        repository.update_role_menus(
            role_key, self.allowed_menu_keys(payload.get("menus"))
        )
        return self.list_roles()

    def list_users(self) -> dict[str, Any]:
        repository = self._require_repository()
        return {
            "users": self.public_users(repository.read_users()["users"]),
            "roles": repository.read_roles()["roles"],
        }

    def create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        repository = self._require_repository()
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        role = str(payload.get("role") or "operator")
        roles = repository.read_roles()["roles"]
        if not username:
            raise HTTPException(status_code=400, detail="请输入用户名")
        if not password:
            raise HTTPException(status_code=400, detail="请输入密码")
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        if username in repository.list_users():
            raise HTTPException(status_code=409, detail="用户已存在")
        repository.create_user(username, password, role)
        return {"users": repository.list_users(), "roles": roles}

    def update_user(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        repository = self._require_repository()
        users = repository.list_users()
        if username not in users:
            raise HTTPException(status_code=404, detail="用户不存在")
        role = str(payload.get("role") or "")
        roles = repository.read_roles()["roles"]
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        repository.update_user_role(username, role)
        return {"users": repository.list_users(), "roles": roles}

    def delete_user(self, username: str) -> dict[str, Any]:
        repository = self._require_repository()
        if not repository.delete_user(username):
            raise HTTPException(status_code=404, detail="用户不存在")
        return {
            "users": repository.list_users(),
            "roles": repository.read_roles()["roles"],
        }
