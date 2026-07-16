import uuid
from typing import Any

from fastapi import HTTPException

from trace_app.config import MENU_LABELS
from trace_app.database.repositories import Repository


class AuthService:
    def __init__(self, repository: Repository | None = None) -> None:
        self.repository = repository if repository is not None else Repository(None)

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
        users = self.repository.read_users()["users"]
        return str(users.get(username, {}).get("role") or "operator")

    def login(self, username: str, password: str) -> dict[str, Any]:
        role = self.repository.authenticate(username, password)
        if role is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        roles = self.repository.read_roles()["roles"]
        menus = self.allowed_menu_keys(roles.get(role, {}).get("menus", []))
        return {
            "token": f"local-{uuid.uuid4().hex}",
            "username": username,
            "role": role,
            "menus": menus,
        }

    def list_roles(self) -> dict[str, Any]:
        return {"menus": MENU_LABELS, "roles": self.repository.read_roles()["roles"]}

    def update_role(self, role_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        roles = self.repository.read_roles()["roles"]
        if role_key not in roles:
            raise HTTPException(status_code=404, detail="角色不存在")
        self.repository.update_role_menus(
            role_key, self.allowed_menu_keys(payload.get("menus"))
        )
        return self.list_roles()

    def list_users(self) -> dict[str, Any]:
        return {
            "users": self.public_users(self.repository.read_users()["users"]),
            "roles": self.repository.read_roles()["roles"],
        }

    def create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        role = str(payload.get("role") or "operator")
        roles = self.repository.read_roles()["roles"]
        if not username:
            raise HTTPException(status_code=400, detail="请输入用户名")
        if not password:
            raise HTTPException(status_code=400, detail="请输入密码")
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        if username in self.repository.list_users():
            raise HTTPException(status_code=409, detail="用户已存在")
        self.repository.create_user(username, password, role)
        return {"users": self.repository.list_users(), "roles": roles}

    def update_user(self, username: str, payload: dict[str, Any]) -> dict[str, Any]:
        users = self.repository.list_users()
        if username not in users:
            raise HTTPException(status_code=404, detail="用户不存在")
        role = str(payload.get("role") or "")
        roles = self.repository.read_roles()["roles"]
        if role not in roles:
            raise HTTPException(status_code=400, detail="角色不存在")
        self.repository.update_user_role(username, role)
        return {"users": self.repository.list_users(), "roles": roles}

    def delete_user(self, username: str) -> dict[str, Any]:
        if not self.repository.delete_user(username):
            raise HTTPException(status_code=404, detail="用户不存在")
        return {
            "users": self.repository.list_users(),
            "roles": self.repository.read_roles()["roles"],
        }
