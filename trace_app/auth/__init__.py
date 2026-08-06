from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_app.auth.service import AuthService

__all__ = ["AuthService"]


def __getattr__(name: str) -> Any:
    if name == "AuthService":
        from trace_app.auth.service import AuthService

        return AuthService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
