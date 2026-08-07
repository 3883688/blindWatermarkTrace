from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from trace_app.database.store import DatabaseStore


@dataclass(slots=True)
class Runtime:
    engine: Engine | None = None
    store: DatabaseStore | None = None
    db_error: str = ""
    generated_trace_ids: list[str] = field(default_factory=list)
    auth_sessions: dict[str, int] = field(default_factory=dict)
    media_signing_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))


def dispose_engine(engine: Any | None) -> None:
    if engine is not None:
        engine.dispose()


def dispose_runtime(runtime: Runtime) -> None:
    dispose_engine(runtime.engine)
