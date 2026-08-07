from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trace_app.database.connection import create_runtime, seed_database_defaults
    from trace_app.database.repositories import Repository

__all__ = ["Repository", "create_runtime", "seed_database_defaults"]


def __getattr__(name: str) -> Any:
    if name == "Repository":
        from trace_app.database.repositories import Repository

        return Repository
    if name in {"create_runtime", "seed_database_defaults"}:
        from trace_app.database.connection import create_runtime, seed_database_defaults

        return {
            "create_runtime": create_runtime,
            "seed_database_defaults": seed_database_defaults,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
