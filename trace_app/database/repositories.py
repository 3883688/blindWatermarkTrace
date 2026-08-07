from datetime import datetime
from typing import Any, Callable

from fastapi import HTTPException

from trace_app.database.store import DatabaseStore


class Repository:
    def __init__(
        self,
        store: DatabaseStore | None,
        ensure_dirs: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._ensure_dirs = ensure_dirs or (lambda: None)

    @property
    def store(self) -> DatabaseStore:
        if self._store is None:
            raise HTTPException(status_code=503, detail="数据库不可用")
        return self._store

    def db_clear_all(self) -> None:
        self.store.clear_all()

    def read_records(
        self, *, owner_user_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self.store.read_records(owner_user_id=owner_user_id)

    def write_records(self, records: list[dict[str, Any]]) -> None:
        self.replace_records(records)

    def replace_records(self, records: list[dict[str, Any]]) -> None:
        self.store.replace_records(records)

    def add_record(
        self,
        record: dict[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> None:
        self.store.insert_record(record, owner_user_id=owner_user_id)

    def delete_record(
        self, image_id: str, *, owner_user_id: int | None = None
    ) -> dict[str, Any] | None:
        return self.store.delete_record(image_id, owner_user_id=owner_user_id)

    def read_detection_stats(self) -> dict[str, int]:
        stats = self.store.get_stats("detection_stats", {})
        return {
            "attempts": int(stats.get("attempts", 0) or 0),
            "successes": int(stats.get("successes", 0) or 0),
        }

    def write_detection_stats(self, stats: dict[str, int]) -> None:
        self._ensure_dirs()
        normalized = {
            "attempts": int(stats.get("attempts", 0) or 0),
            "successes": int(stats.get("successes", 0) or 0),
        }
        self.store.set_stats("detection_stats", normalized)

    def record_detection_result(self, success: bool) -> None:
        stats = self.read_detection_stats()
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        self.write_detection_stats(stats)

    def read_watermark_stats(self) -> dict[str, dict[str, int]]:
        stats = self.store.get_stats("watermark_stats", {})
        daily = stats.get("daily", {})
        if not isinstance(daily, dict):
            daily = {}
        return {"daily": {str(day): int(count or 0) for day, count in daily.items()}}

    def write_watermark_stats(self, stats: dict[str, Any]) -> None:
        self._ensure_dirs()
        daily = stats.get("daily", {})
        if not isinstance(daily, dict):
            daily = {}
        normalized = {
            "daily": {str(day): int(count or 0) for day, count in daily.items()}
        }
        self.store.set_stats("watermark_stats", normalized)

    def record_watermark_generation(self) -> None:
        stats = self.read_watermark_stats()
        today = datetime.now().strftime("%Y-%m-%d")
        stats["daily"][today] = int(stats["daily"].get(today, 0)) + 1
        self.write_watermark_stats(stats)

    def read_roles(self) -> dict[str, Any]:
        return {"roles": self.store.read_roles()}

    def replace_roles(self, roles: dict[str, dict[str, Any]]) -> None:
        self.store.replace_roles(roles)

    def update_role_menus(self, role_key: str, menus: list[str]) -> bool:
        return self.store.update_role_menus(role_key, menus)

    def read_users(self) -> dict[str, Any]:
        return {"users": self.store.list_users()}

    def list_users(self) -> dict[str, dict[str, str]]:
        return self.store.list_users()

    def create_user(self, username: str, password: str, role_key: str) -> None:
        self.store.create_user(username, password, role_key)

    def update_user_role(self, username: str, role_key: str) -> bool:
        return self.store.update_user_role(username, role_key)

    def delete_user(self, username: str) -> bool:
        return self.store.delete_user(username)

    def authenticate_user(
        self, username: str, password: str
    ) -> dict[str, Any] | None:
        return self.store.authenticate_user(username, password)

    def get_login_identity(self, username: str) -> dict[str, Any] | None:
        return self.store.get_login_identity(username)

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        return self.store.get_user_by_id(user_id)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        return self.store.get_user_by_username(username)

    def authenticate(self, username: str, password: str) -> str | None:
        return self.store.authenticate(username, password)

    def today_watermark_count(
        self,
        records: list[dict[str, Any]],
        stats: dict[str, dict[str, int]] | None = None,
        is_today: Callable[[dict[str, Any]], bool] | None = None,
    ) -> int:
        stats = self.read_watermark_stats() if stats is None else stats
        today = datetime.now().strftime("%Y-%m-%d")
        if today in stats["daily"]:
            return int(stats["daily"][today])
        predicate = self.is_today_record if is_today is None else is_today
        return sum(1 for item in records if predicate(item))

    @staticmethod
    def is_today_record(record: dict[str, Any]) -> bool:
        created_at = str(record.get("created_at") or "")
        return created_at.startswith(datetime.now().strftime("%Y-%m-%d"))
