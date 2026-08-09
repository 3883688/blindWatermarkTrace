from typing import Any

from fastapi import APIRouter, Depends, Request

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service
from trace_app.management.service import ManagementService
from trace_app.media import with_media_access_urls

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("")
def list_images(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    result = service.list_images(current_user)
    result["items"] = [
        with_media_access_urls(
            record,
            key=request.app.state.media_signing_key,
            ttl_seconds=request.app.state.media_url_ttl_seconds,
        )
        for record in result["items"]
    ]
    return result


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.delete_image(image_id, current_user)
