from __future__ import annotations

import re
import shutil
from typing import Any, Callable

from fastapi import HTTPException

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import Settings
from trace_app.database.connection import seed_database_defaults
from trace_app.database.repositories import Repository
from trace_app.runtime import Runtime


class ManagementService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        runtime: Runtime,
        ensure_directories: Callable[[], None],
        database_enabled: bool,
        today_watermark_count: Callable[[list[dict[str, Any]]], int] | None = None,
        read_records: Callable[[], list[dict[str, Any]]] | None = None,
        read_detection_stats: Callable[[], dict[str, int]] | None = None,
        write_records: Callable[[list[dict[str, Any]]], None] | None = None,
        database_ready: Callable[[], bool] | None = None,
        database_error: Callable[[], str] | None = None,
        masked_db_url: Callable[[], str] | None = None,
        clear_database: Callable[[], None] | None = None,
        seed_database: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.runtime = runtime
        self.ensure_directories = ensure_directories
        self.database_enabled = database_enabled
        self._today_watermark_count = today_watermark_count
        self._read_records = read_records or repository.read_records
        self._read_detection_stats = (
            read_detection_stats or repository.read_detection_stats
        )
        self._write_records = write_records or repository.write_records
        self._database_ready = database_ready or (lambda: runtime.store is not None)
        self._database_error = database_error or (lambda: runtime.db_error)
        self._masked_db_url = masked_db_url or (
            lambda: re.sub(
                r"://([^:/@]+):([^@]+)@", r"://\1:****@", settings.db_url
            )
        )
        self._clear_database = clear_database or repository.db_clear_all
        self._seed_database = seed_database or (
            lambda: seed_database_defaults(repository.store, settings)
        )

    def _today_count(self, records: list[dict[str, Any]]) -> int:
        if self._today_watermark_count is not None:
            return self._today_watermark_count(records)
        return self.repository.today_watermark_count(records)

    def dashboard_stats(self) -> dict[str, int | float]:
        records = self._read_records()
        detection_stats = self._read_detection_stats()
        attempts = detection_stats["attempts"]
        successes = detection_stats["successes"]
        success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
        return {
            "today": self._today_count(records),
            "detection_success_rate": success_rate,
        }

    def list_images(self, current_user: AuthenticatedUser) -> dict[str, Any]:
        owner_user_id = None if current_user.role == "admin" else current_user.id
        records = self.repository.read_records(owner_user_id=owner_user_id)
        protected = sum(1 for item in records if item.get("status") == "保护中")
        leaks = sum(1 for item in records if item.get("status") == "泄露预警")
        hits = sum(1 for item in records if item.get("status") == "溯源命中")
        detection_stats = self._read_detection_stats()
        attempts = detection_stats["attempts"]
        successes = detection_stats["successes"]
        success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
        return {
            "items": records,
            "stats": {
                "total": len(records),
                "protected": protected,
                "leaks": leaks,
                "hits": hits,
                "today": self._today_count(records),
                "detection_attempts": attempts,
                "detection_successes": successes,
                "detection_success_rate": success_rate,
            },
            "db_enabled": self.database_enabled,
            "db_ready": self._database_ready(),
            "db_error": self._database_error(),
            "db_url": self._masked_db_url(),
        }

    def delete_image(
        self, image_id: str, current_user: AuthenticatedUser
    ) -> dict[str, bool]:
        owner_user_id = None if current_user.role == "admin" else current_user.id
        target = self.repository.delete_record(
            image_id,
            owner_user_id=owner_user_id,
        )
        if not target:
            raise HTTPException(status_code=404, detail="图片不存在")
        for key in ("original_url", "download_url", "thumbnail_url"):
            value = target.get(key)
            if value and value.startswith("/uploads/"):
                path = self.settings.upload_dir / value.replace("/uploads/", "")
                if path.exists():
                    path.unlink()
        return {"deleted": True}

    def reset_dev_data(self) -> dict[str, bool]:
        if self.settings.upload_dir.exists():
            shutil.rmtree(self.settings.upload_dir)
        if self._database_ready():
            self._clear_database()
            self._seed_database()
        if self.settings.data_dir.exists():
            shutil.rmtree(self.settings.data_dir)
        self.ensure_directories()
        return {"reset": True}
