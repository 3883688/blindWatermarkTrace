from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: int
    username: str
    role: str


class RolePayload(TypedDict, total=False):
    menus: Any


class UserPayload(TypedDict, total=False):
    username: Any
    password: Any
    role: Any
