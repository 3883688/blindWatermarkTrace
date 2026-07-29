"""HTTP 接口层聚合入口。

在此集中导入各路由模块，供 :mod:`trace_app.application` 逐个 ``include_router``。
这里只做导入，不创建 ``APIRouter``，保证"每个应用实例注册一次路由"的契约。
"""

from trace_app.api import auth, dashboard, images, media, users, watermark

__all__ = ["auth", "dashboard", "images", "media", "users", "watermark"]
