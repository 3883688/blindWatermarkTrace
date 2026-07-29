"""首页看板与开发期数据重置接口。"""

from fastapi import APIRouter, Depends

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service, require_admin
from trace_app.management.service import ManagementService

router = APIRouter(prefix="/api", tags=["dashboard"])
dev_router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard-stats")
def dashboard_stats(
    _current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, int | float]:
    """首页统计：累计嵌入次数、检测次数、检出率等聚合指标。"""
    return service.dashboard_stats()


@dev_router.post("/dev/reset")
def reset_dev_data(
    _admin: AuthenticatedUser = Depends(require_admin),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    """清空开发环境的图片记录与统计计数。

    这是**破坏性**接口，只应在开发/演示环境暴露；生产部署需要在反向代理层
    屏蔽 ``/api/dev/*``。
    """
    return service.reset_dev_data()
