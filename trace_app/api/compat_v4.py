"""Original browser API contract backed exclusively by V4 services."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from trace_app.api.v4 import _upload_bytes
from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import (
    get_current_user,
    get_repository,
    get_v4_detection_service,
    get_v4_generation_service,
    get_v4_media_service,
    get_v4_record_repository,
)
from trace_app.imaging.io import fetch_remote_image_bytes
from trace_app.v4.deadlines import Deadline
from trace_app.v4.detection import DetectionRequest
from trace_app.v4.domain import DetectionOutcome, OwnerScope
from trace_app.v4.generation import GenerationRequest


router = APIRouter(prefix="/api", tags=["original-v4-contract"])


def _scope(user: AuthenticatedUser) -> OwnerScope:
    return OwnerScope(user.id, cross_owner=user.role == "admin")


def _created_at(record: object) -> str:
    value = getattr(record, "created_at", None)
    return value.isoformat() if isinstance(value, datetime) else datetime.now(UTC).isoformat()


def _group(repository: Any, scope: OwnerScope, record: object) -> object | None:
    return repository.get_source_group(scope, getattr(record, "source_group_id"))


def _url(media: Any, media_id: str | None, user: AuthenticatedUser) -> str | None:
    if not media_id:
        return None
    return media.issue_url(
        media_id,
        requester_user_id=user.id,
        requester_is_admin=user.role == "admin",
    )


def _format_byte_size(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "-"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / 1024 / 1024:.1f} MB"


def _owner_username(
    record: object,
    users: Any,
    cache: dict[int, str] | None = None,
) -> str:
    owner_user_id = int(getattr(record, "owner_user_id"))
    if cache is not None and owner_user_id in cache:
        return cache[owner_user_id]
    identity = users.get_user_by_id(owner_user_id)
    username = identity.get("username") if isinstance(identity, dict) else None
    result = str(username).strip() if username else str(owner_user_id)
    if cache is not None:
        cache[owner_user_id] = result
    return result


def _image_payload(
    record: object,
    user: AuthenticatedUser,
    repository: Any,
    media: Any,
    users: Any,
    owner_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    group = _group(repository, _scope(user), record)
    output_media_id = getattr(record, "output_media_id", None)
    output_media = repository.get_media(output_media_id) if output_media_id else None
    result: dict[str, Any] = {
        "id": str(getattr(record, "id")),
        "name": getattr(record, "original_filename", "image"),
        "size": _format_byte_size(getattr(output_media, "byte_size", None)),
        "user_id": _owner_username(record, users, owner_names),
        "trace_id": getattr(record, "trace_id"),
        "evidence_uuid_head": str(getattr(record, "evidence_uuid", ""))[:8],
        "evidence_uuid_tail": str(getattr(record, "evidence_uuid", ""))[-8:],
        "robust_watermark_version": 4,
        "mode": "dct",
        "mode_label": "V4 HMAC64 + RS",
        "created_at": _created_at(record),
        "time": _created_at(record),
        "status": "保护中",
        "confidence": 100,
        "conf": 100,
    }
    for media_id, field in (
        (getattr(group, "original_media_id", None), "original_access_url"),
        (getattr(record, "output_media_id", None), "download_access_url"),
        (getattr(record, "thumbnail_media_id", None), "thumbnail_access_url"),
    ):
        access = _url(media, media_id, user)
        if access:
            result[field] = access
    return result


def _generation_payload(
    record: object,
    user: AuthenticatedUser,
    repository: Any,
    media: Any,
    users: Any,
) -> dict[str, Any]:
    item = _image_payload(record, user, repository, media, users)
    return {
        **item,
        "evidence_uuid_head": str(getattr(record, "evidence_uuid", ""))[:8],
        "evidence_uuid_tail": str(getattr(record, "evidence_uuid", ""))[-8:],
        "robust_watermark_version": 4,
        "original_access_url": item.get("original_access_url"),
        "download_access_url": item.get("download_access_url"),
    }


@router.post("/watermark/embed")
async def embed(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(""),
    copyright_enabled: str = Form(""),
    copyright_text: str = Form(""),
    copyright_opacity: str = Form(""),
    copyright_complexity: str = Form(""),
    copyright_irregular_enabled: str = Form(""),
    copyright_prominent_corner_enabled: str = Form(""),
    protected_region_enhancement: str = Form("false"),
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_generation_service),
    repository: Any = Depends(get_v4_record_repository),
    media: Any = Depends(get_v4_media_service),
    users: Any = Depends(get_repository),
) -> dict[str, Any]:
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "image"
    content = await _upload_bytes(file, request)
    result = await run_in_threadpool(
        service.generate,
        GenerationRequest(
            current_user.id,
            content,
            content_type,
            filename,
            metadata={
                "copyright_enabled": copyright_enabled,
                "copyright_text": copyright_text,
                "copyright_opacity": copyright_opacity,
                "copyright_complexity": copyright_complexity,
                "copyright_irregular_enabled": copyright_irregular_enabled,
                "copyright_prominent_corner_enabled": copyright_prominent_corner_enabled,
                "protected_region_enhancement": protected_region_enhancement,
            },
        ),
        Deadline.synchronous(),
    )
    return _generation_payload(result.record, current_user, repository, media, users)


def _detected_payload(
    result: Any,
    user: AuthenticatedUser,
    repository: Any,
    media: Any,
    users: Any,
) -> dict[str, Any]:
    if result.outcome != DetectionOutcome.SUCCESS or result.record is None:
        raise HTTPException(status_code=404, detail="未检测到可验证的 V4 水印")
    record = result.record
    group = _group(repository, _scope(user), record)
    payload = _image_payload(record, user, repository, media, users)
    payload["extracted_at"] = datetime.now(UTC).isoformat()
    payload["matched_file_access_url"] = _url(
        media, getattr(group, "original_media_id", None), user
    )
    payload["robust_watermark_version"] = 4
    return payload


@router.post("/watermark/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_detection_service),
    repository: Any = Depends(get_v4_record_repository),
    media: Any = Depends(get_v4_media_service),
    users: Any = Depends(get_repository),
) -> dict[str, Any]:
    content = await _upload_bytes(file, request)
    result = await run_in_threadpool(
        service.detect,
        DetectionRequest(_scope(current_user), content),
        Deadline.synchronous(),
    )
    repository.increment_counter(current_user.id, "detection_total")
    if result.outcome == DetectionOutcome.SUCCESS:
        repository.increment_counter(current_user.id, "detection_success")
    return _detected_payload(result, current_user, repository, media, users)


@router.post("/watermark/extract-url")
async def extract_url(
    url: str = Form(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_detection_service),
    repository: Any = Depends(get_v4_record_repository),
    media: Any = Depends(get_v4_media_service),
    users: Any = Depends(get_repository),
) -> dict[str, Any]:
    deadline = Deadline.synchronous()
    content, _content_type = await run_in_threadpool(
        fetch_remote_image_bytes, url, deadline=deadline
    )
    result = await run_in_threadpool(
        service.detect, DetectionRequest(_scope(current_user), content), deadline
    )
    repository.increment_counter(current_user.id, "detection_total")
    if result.outcome == DetectionOutcome.SUCCESS:
        repository.increment_counter(current_user.id, "detection_success")
    return _detected_payload(result, current_user, repository, media, users)


@router.get("/images")
def images(
    current_user: AuthenticatedUser = Depends(get_current_user),
    repository: Any = Depends(get_v4_record_repository),
    media: Any = Depends(get_v4_media_service),
    users: Any = Depends(get_repository),
) -> dict[str, Any]:
    scope = _scope(current_user)
    owner_names: dict[int, str] = {}
    items = [
        _image_payload(record, current_user, repository, media, users, owner_names)
        for record in repository.list_records(scope)
    ]
    return {"items": items, "stats": repository.dashboard_stats(scope)}


@router.delete("/images/{image_id}")
def delete_image(
    image_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    repository: Any = Depends(get_v4_record_repository),
) -> dict[str, bool]:
    if not repository.delete_record(_scope(current_user), image_id):
        raise HTTPException(status_code=404, detail="图片不存在")
    return {"deleted": True}


@router.get("/dashboard-stats")
def dashboard(
    current_user: AuthenticatedUser = Depends(get_current_user),
    repository: Any = Depends(get_v4_record_repository),
) -> dict[str, int | float]:
    return repository.dashboard_stats(_scope(current_user))


__all__ = ("router",)
