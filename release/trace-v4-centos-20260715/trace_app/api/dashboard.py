from fastapi import APIRouter, Depends

from trace_app.dependencies import get_management_service
from trace_app.management.service import ManagementService

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard-stats")
def dashboard_stats(
    service: ManagementService = Depends(get_management_service),
) -> dict[str, int | float]:
    return service.dashboard_stats()


@router.post("/dev/reset")
def reset_dev_data(
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.reset_dev_data()
