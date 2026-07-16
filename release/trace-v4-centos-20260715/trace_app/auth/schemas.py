from typing import Any, TypedDict


class RolePayload(TypedDict, total=False):
    menus: Any


class UserPayload(TypedDict, total=False):
    username: Any
    password: Any
    role: Any
