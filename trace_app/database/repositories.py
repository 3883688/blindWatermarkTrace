from datetime import datetime
from typing import Any, Callable

from fastapi import HTTPException

from database_store import DatabaseStore


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

    def read_records(self) -> list[dict[str, Any]]:
        return self.store.read_records()

    def write_records(self, records: list[dict[str, Any]]) -> None:
        self.replace_records(records)

    def replace_records(self, records: list[dict[str, Any]]) -> None:
        self.store.replace_records(records)

    def add_record(self, record: dict[str, Any]) -> None:
        records = self.read_records()
        records.insert(0, record)
        self.write_records(records)

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

    def read_users(self) -> dict[str, Any]:
        return {"users": self.store.list_users()}

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
