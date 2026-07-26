from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service
from trace_app.management.service import ManagementService
from trace_app.media import sign_expiring_url, verify_media_signature

router = APIRouter(prefix="/api/images", tags=["images"])

PUBLIC_IMAGE_FIELDS = (
    "id",
    "name",
    "size",
    "user_id",
    "trace_id",
    "mode",
    "mode_label",
    "created_at",
    "time",
    "status",
    "confidence",
    "conf",
)


def _public_image(
    record: dict[str, Any], *, key: bytes, ttl_seconds: int
) -> dict[str, Any]:
    result = {key: record[key] for key in PUBLIC_IMAGE_FIELDS if key in record}
    image_id = quote(str(record["id"]), safe="")
    if record.get("download_url"):
        result["download_access_url"] = sign_expiring_url(
            f"/api/images/{image_id}/download",
            key,
            ttl_seconds=ttl_seconds,
        )
    if record.get("thumbnail_url"):
        result["thumbnail_access_url"] = sign_expiring_url(
            f"/api/images/{image_id}/thumbnail",
            key,
            ttl_seconds=ttl_seconds,
        )
    return result


@router.get("")
def list_images(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    service_result = service.list_images(current_user)
    return {
        "items": [
            _public_image(
                record,
                key=request.app.state.media_signing_key,
                ttl_seconds=request.app.state.media_url_ttl_seconds,
            )
            for record in service_result["items"]
        ],
        "stats": service_result["stats"],
    }


@router.get("/{image_id}/{variant}", response_class=FileResponse)
def get_image_media(
    request: Request,
    image_id: str,
    variant: str,
    expire_time: str | None = None,
    signature: str | None = None,
    service: ManagementService = Depends(get_management_service),
) -> FileResponse:
    access_path = f"/api/images/{quote(image_id, safe='')}/{variant}"
    if not verify_media_signature(
        access_path,
        expires=expire_time,
        signature=signature,
        key=request.app.state.media_signing_key,
    ):
        raise HTTPException(status_code=403, detail="图片访问链接无效或已过期")
    path = service.get_image_media_path(image_id, variant)
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    return service.delete_image(image_id, current_user)
