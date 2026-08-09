from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from database_store import DatabaseStore
from trace_app.config import DEFAULT_ROLES, Settings
from trace_app.runtime import Runtime, dispose_engine


def seed_database_defaults(store: DatabaseStore, settings: Settings) -> None:
    if not store.read_roles():
        store.replace_roles(DEFAULT_ROLES)
    if (
        settings.admin_user
        and settings.admin_pass
        and settings.admin_user not in store.list_users()
    ):
        store.create_user(settings.admin_user, settings.admin_pass, "admin")


def create_runtime(settings: Settings, *, enabled: bool = True) -> Runtime:
    runtime = Runtime()
    if not enabled:
        return runtime

    missing = [
        name
        for name, value in (
            ("DB_URL", settings.db_url),
            ("ADMIN_USER", settings.admin_user),
            ("ADMIN_PASS", settings.admin_pass),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable: {missing[0]}")

    try:
        runtime.engine = create_engine(
            settings.db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )
        runtime.store = DatabaseStore(runtime.engine)
        runtime.store.create_schema()
        seed_database_defaults(runtime.store, settings)
    except SQLAlchemyError as exc:
        runtime.db_error = type(exc).__name__
        runtime.store = None
        dispose_engine(runtime.engine)
        error = RuntimeError("Database initialization failed")
        setattr(error, "runtime", runtime)
        raise error from exc
    return runtime
