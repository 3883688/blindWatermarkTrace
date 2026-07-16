from typing import Any

from fastapi import APIRouter, Depends

from trace_app.dependencies import get_management_service
from trace_app.management.service import ManagementService

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("")
def list_images(
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    return service.list_images()


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.delete_image(image_id)
