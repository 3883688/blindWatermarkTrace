from typing import Any

from fastapi import APIRouter, Depends

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service
from trace_app.management.service import ManagementService

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("")
def list_images(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    return service.list_images(current_user)


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.delete_image(image_id, current_user)
