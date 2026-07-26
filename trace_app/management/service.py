"""图片资产管理与看板统计的服务层。

支撑 ``/api/images``、``/api/dashboard-stats`` 与 ``/api/dev/reset``。

构造函数里那一长串 ``Callable`` 参数都是**可选的依赖注入点**：默认全部回落到
:class:`Repository` 的同名方法，测试时可以逐个替换成假实现，从而在不起数据库、
不碰磁盘的前提下验证统计口径与删除逻辑。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import Settings
from trace_app.database.connection import seed_database_defaults
from trace_app.database.repositories import Repository
from trace_app.media import media_path_from_url, resolve_media_path
from trace_app.runtime import Runtime


class ManagementService:
    """图片资产管理服务：列表、媒体寻址、删除、看板统计与开发期重置。"""

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
        """
        :param database_enabled: 配置层面是否启用数据库（与"当前是否连上"是两回事）。
        :param ensure_directories: 重建上传/数据目录的回调，重置数据后调用。

        其余 ``Callable`` 参数为空时统一回落到仓储的同名方法，见下方 ``or`` 兜底。
        """
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
        # "就绪"以 runtime 里是否真的挂上了 store 为准，而非看配置开关。
        self._database_ready = database_ready or (lambda: runtime.store is not None)
        self._database_error = database_error or (lambda: runtime.db_error)
        # 看板要展示数据库地址便于运维排障，但必须把 URL 里的密码替换成 ****
        # 再下发；正则只匹配 scheme://user:pass@ 这一段，保留主机与库名。
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
        """今日嵌入量：优先用注入的计数器，否则用仓储的默认实现。"""
        if self._today_watermark_count is not None:
            return self._today_watermark_count(records)
        return self.repository.today_watermark_count(records)

    def dashboard_stats(self) -> dict[str, int | float]:
        """首页看板的两个核心指标：今日嵌入量与检测成功率。"""
        records = self._read_records()
        detection_stats = self._read_detection_stats()
        attempts = detection_stats["attempts"]
        successes = detection_stats["successes"]
        # 一次检测都没发生时成功率记 0.0 而不是除零；保留一位小数供前端直出。
        success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
        return {
            "today": self._today_count(records),
            "detection_success_rate": success_rate,
        }

    def list_images(self, current_user: AuthenticatedUser) -> dict[str, Any]:
        """列出当前用户可见的图片记录，并附带统计与数据库健康信息。

        权限在**查询层**收口：管理员传 ``owner_user_id=None`` 拿全量，
        其他角色只带自己的 id，从 SQL 层就过滤掉他人数据，
        而不是查出来再在内存里筛（那样容易漏筛导致越权）。

        统计口径注意：``total/protected/leaks/hits`` 是**当前用户可见范围**内的
        计数，而 ``detection_*`` 是全局累计值，两者并非同一分母。
        """
        owner_user_id = None if current_user.role == "admin" else current_user.id
        records = self.repository.read_records(owner_user_id=owner_user_id)
        # 三种业务状态各自计数，用于列表页顶部的状态卡片。
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
        """删除一条图片记录，并清理它在磁盘上的三个文件。

        归属校验同样下沉到仓储：非管理员带上 ``owner_user_id``，删别人的记录
        会命中 0 行、``target`` 为空，最终返回 404——**不区分"不存在"与"无权限"**，
        避免通过响应码探测他人图片是否存在。

        先删库再删文件：万一文件删除中途失败，留下的是孤儿文件（可后台清理），
        而不是"记录还在但文件没了"的坏链。
        """
        owner_user_id = None if current_user.role == "admin" else current_user.id
        target = self.repository.delete_record(
            image_id,
            owner_user_id=owner_user_id,
        )
        if not target:
            raise HTTPException(status_code=404, detail="图片不存在")
        # 原图、带水印成品图、缩略图三份文件一并清理。
        for key in ("original_url", "download_url", "thumbnail_url"):
            value = target.get(key)
            # 只处理受管的 /uploads/ 前缀，外链或异常值一律跳过，防止误删。
            if value and value.startswith("/uploads/"):
                path = self.settings.upload_dir / value.replace("/uploads/", "")
                if path.exists():
                    path.unlink()
        return {"deleted": True}

    def get_image_media_path(
        self,
        image_id: str,
        variant: str,
    ) -> Path:
        """把 ``(image_id, variant)`` 解析成磁盘上的真实文件路径。

        :param variant: 只接受 ``download``（成品图）与 ``thumbnail``（缩略图）。
        :raises HTTPException: 任何一步不满足都统一抛 404，不泄露失败原因。

        四道校验层层收紧，缺一不可：

        1. ``variant`` 必须在固定映射表里——**不允许**用它拼字段名或路径，
           否则就是任意文件读取；
        2. 记录必须存在，且该字段有值；
        3. 值必须是 ``/uploads/`` 前缀的受管地址；
        4. :func:`resolve_media_path` 再做一次目录逃逸检查（``..`` 之类）。
        """
        field = {
            "download": "download_url",
            "thumbnail": "thumbnail_url",
        }.get(variant)
        if field is None:
            raise HTTPException(status_code=404, detail="图片不存在")
        # 注意这里读全量记录、不带 owner 过滤：本接口靠 URL 签名鉴权，
        # 调用方（get_image_media）已在签名校验通过后才走到这一步。
        record = next(
            (
                item
                for item in self.repository.read_records()
                if str(item.get("id")) == image_id
            ),
            None,
        )
        media_url = record.get(field) if record else None
        if not isinstance(media_url, str) or not media_url.startswith("/uploads/"):
            raise HTTPException(status_code=404, detail="图片不存在")
        path = resolve_media_path(
            self.settings.upload_dir,
            media_path_from_url(media_url),
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return path

    def reset_dev_data(self) -> dict[str, bool]:
        """清空开发环境数据：删除上传目录、重置数据库、重建目录结构。

        **破坏性操作**，仅供开发/演示环境使用。

        顺序是有讲究的：先清空数据库并重新灌入默认账号角色，最后才
        ``ensure_directories()`` 重建空目录——保证方法返回时目录结构完整可用，
        下一次上传不会因为目录不存在而失败。
        """
        if self.settings.upload_dir.exists():
            shutil.rmtree(self.settings.upload_dir)
        # 数据库没连上就跳过清库，只清文件，不因此报错。
        if self._database_ready():
            self._clear_database()
            self._seed_database()
        if self.settings.data_dir.exists():
            shutil.rmtree(self.settings.data_dir)
        self.ensure_directories()
        return {"reset": True}
