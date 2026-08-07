"""Opaque V4 media URL issuance and signed transfer endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_v4_media_service
from trace_app.v4.media import V4MediaService


router = APIRouter(prefix="/api/media", tags=["media"])


@router.post("/{media_id}/access")
def issue_media_access(
    media_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: V4MediaService = Depends(get_v4_media_service),
) -> dict[str, str]:
    return {
        "url": service.issue_url(
            media_id,
            requester_user_id=current_user.id,
            requester_is_admin=current_user.role == "admin",
        )
    }


@router.get("/{media_id}")
def transfer_media(
    media_id: str,
    expires: int | None = Query(None),
    signature: str | None = Query(None),
    service: V4MediaService = Depends(get_v4_media_service),
) -> FileResponse:
    media = service.get_media_or_404(media_id)
    if expires is None or signature is None or not service.verify(
        media, expires=expires, signature=signature
    ):
        raise HTTPException(status_code=403, detail="媒体访问链接无效或已过期")
    path = service.resolve_storage_key(media.storage_key)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="媒体不存在")
    return FileResponse(
        path,
        media_type=media.content_type,
        headers={"Cache-Control": "private, no-store"},
    )
