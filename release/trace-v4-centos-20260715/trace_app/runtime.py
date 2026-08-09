from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from database_store import DatabaseStore


@dataclass(slots=True)
class Runtime:
    engine: Engine | None = None
    store: DatabaseStore | None = None
    db_error: str = ""
    generated_trace_ids: list[str] = field(default_factory=list)


def dispose_engine(engine: Any | None) -> None:
    if engine is not None:
        engine.dispose()


def dispose_runtime(runtime: Runtime) -> None:
    dispose_engine(runtime.engine)
