"""Authenticated V4-only product API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import (
    get_current_user,
    get_v4_detection_service,
    get_v4_generation_service,
    get_v4_job_service,
    get_v4_media_service,
    get_v4_record_repository,
)
from trace_app.imaging.io import fetch_remote_image_bytes
from trace_app.v4.deadlines import Deadline, DeadlineExceeded
from trace_app.v4.detection import DetectionRequest
from trace_app.v4.domain import DetectionOutcome, OwnerScope
from trace_app.v4.generation import GenerationRequest
from trace_app.v4.uploads import UploadLimitExceeded, stream_upload
from watermark_v4.payload import CODEC_ID


router = APIRouter(prefix="/api/v4", tags=["v4"])
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _scope(user: AuthenticatedUser, cross_owner: bool) -> OwnerScope:
    if cross_owner and user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return OwnerScope(user.id, cross_owner=cross_owner)


def _attr(record: object, *names: str) -> Any:
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _record_payload(record: object, user: AuthenticatedUser, media: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(_attr(record, "id", "record_id")),
        "source_group_id": str(_attr(record, "source_group_id")),
        "owner_user_id": _attr(record, "owner_user_id"),
        "trace_id": _attr(record, "trace_id"),
        "codec": _attr(record, "codec") or CODEC_ID,
        "status": _attr(record, "status") or "active",
    }
    for field, url_field in (
        ("output_media_id", "output_access_url"),
        ("thumbnail_media_id", "thumbnail_access_url"),
    ):
        media_id = _attr(record, field)
        if media_id:
            payload[field] = media_id
            payload[url_field] = media.issue_url(
                media_id,
                requester_user_id=user.id,
                requester_is_admin=user.role == "admin",
            )
    return payload


async def _upload_bytes(upload: UploadFile, request: Request) -> bytes:
    staged = None
    try:
        staged = await stream_upload(
            upload,
            temp_root=Path(request.app.state.settings.data_dir) / "v4-temp",
            max_bytes=MAX_UPLOAD_BYTES,
        )
        if staged.byte_size == 0:
            raise HTTPException(status_code=422, detail="图片不能为空")
        return await run_in_threadpool(staged.path.read_bytes)
    except UploadLimitExceeded as exc:
        raise HTTPException(status_code=413, detail="图片文件过大") from exc
    finally:
        if staged is not None:
            staged.cleanup()
        await upload.close()


def _detection_payload(result: Any, user: AuthenticatedUser, media: Any) -> dict[str, Any]:
    outcome = result.outcome.value if isinstance(result.outcome, DetectionOutcome) else str(result.outcome)
    payload: dict[str, Any] = {"outcome": outcome}
    if outcome == DetectionOutcome.SUCCESS.value and result.record is not None:
        payload["record"] = _record_payload(result.record, user, media)
    return payload


@router.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
    codec: str = Form(...),
    copyright_enabled: str = Form(""),
    copyright_text: str = Form(""),
    copyright_opacity: str = Form(""),
    copyright_complexity: str = Form(""),
    copyright_irregular_enabled: str = Form(""),
    copyright_prominent_corner_enabled: str = Form(""),
    output_quality: str = Form("80"),
    pilot_amplitude: str = Form("0.75"),
    protected_region_enhancement: str = Form("false"),
    user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_generation_service),
    media: Any = Depends(get_v4_media_service),
) -> dict[str, Any]:
    if codec != CODEC_ID:
        raise HTTPException(status_code=422, detail="仅支持 V4 codec")
    content = await _upload_bytes(file, request)
    deadline = Deadline.synchronous()
    result = await run_in_threadpool(
        service.generate,
        GenerationRequest(
            user.id,
            content,
            file.content_type or "application/octet-stream",
            metadata={
                "copyright_enabled": copyright_enabled,
                "copyright_text": copyright_text,
                "copyright_opacity": copyright_opacity,
                "copyright_complexity": copyright_complexity,
                "copyright_irregular_enabled": copyright_irregular_enabled,
                "copyright_prominent_corner_enabled": copyright_prominent_corner_enabled,
                "output_quality": output_quality,
                "pilot_amplitude": pilot_amplitude,
                "protected_region_enhancement": protected_region_enhancement,
            },
        ),
        deadline,
    )
    return {
        "outcome": "success",
        "source_group_created": bool(result.source_group_created),
        **_record_payload(result.record, user, media),
    }


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    cross_owner: bool = Form(False),
    user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_detection_service),
    media: Any = Depends(get_v4_media_service),
) -> dict[str, Any]:
    active_scope = _scope(user, cross_owner)
    content = await _upload_bytes(file, request)
    result = await run_in_threadpool(
        service.detect, DetectionRequest(active_scope, content), Deadline.synchronous()
    )
    return _detection_payload(result, user, media)


@router.post("/detect-url")
async def detect_url(
    request: Request,
    url: str = Form(...),
    cross_owner: bool = Form(False),
    user: AuthenticatedUser = Depends(get_current_user),
    service: Any = Depends(get_v4_detection_service),
    media: Any = Depends(get_v4_media_service),
) -> dict[str, Any]:
    active_scope = _scope(user, cross_owner)
    deadline = Deadline.synchronous()
    fetcher = getattr(request.app.state, "v4_remote_fetch_factory", None)
    if fetcher is None:
        fetcher = lambda remote_url, active_deadline: fetch_remote_image_bytes(
            remote_url, deadline=active_deadline
        )
    try:
        content, _content_type = await run_in_threadpool(fetcher, url, deadline)
        result = await run_in_threadpool(
            service.detect, DetectionRequest(active_scope, content), deadline
        )
    except DeadlineExceeded:
        return {"outcome": DetectionOutcome.TIMEOUT.value}
    return _detection_payload(result, user, media)


@router.get("/records")
def records(
    cross_owner: bool = Query(False),
    user: AuthenticatedUser = Depends(get_current_user),
    repository: Any = Depends(get_v4_record_repository),
    media: Any = Depends(get_v4_media_service),
) -> dict[str, Any]:
    return {
        "items": [
            _record_payload(record, user, media)
            for record in repository.list_records(_scope(user, cross_owner))
        ]
    }


@router.get("/capabilities")
def capabilities(
    request: Request,
    _user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    factory = getattr(request.app.state, "v4_capabilities_factory", None)
    state = getattr(request.app.state, "v4_capabilities", {}) if factory is None else factory()
    return {
        "codec": CODEC_ID,
        "dinov2": state.get("dinov2") is True,
        "lightglue": state.get("lightglue") is True,
    }


@router.delete("/records/{record_id}")
def delete_record(
    record_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    repository: Any = Depends(get_v4_record_repository),
) -> dict[str, bool]:
    deleted = repository.delete_record(OwnerScope(user.id), record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="V4 记录不存在")
    return {"deleted": True}


def _job_payload(job: Any) -> dict[str, Any]:
    payload = {
        "id": str(job.id),
        "status": job.status,
        "progress": int(job.progress),
        "deadline_at": job.deadline_at.isoformat(),
    }
    if job.result is not None:
        payload["result"] = {
            key: value
            for key, value in job.result.items()
            if key in {"outcome", "result_media_id", "evidence_id"}
        }
    return payload


@router.post("/jobs", status_code=202)
def create_deep_job(
    media_id: str = Form(...),
    cross_owner: bool = Form(False),
    user: AuthenticatedUser = Depends(get_current_user),
    jobs: Any = Depends(get_v4_job_service),
    media_service: Any = Depends(get_v4_media_service),
) -> dict[str, Any]:
    scope = _scope(user, cross_owner)
    media = media_service.get_media_or_404(media_id)
    if scope.query_owner_id is not None and media.owner_user_id != scope.query_owner_id:
        raise HTTPException(status_code=404, detail="媒体不存在")
    return _job_payload(jobs.create(user.id, scope, media.id))


@router.get("/jobs/{job_id}")
def get_deep_job(
    job_id: UUID,
    cross_owner: bool = Query(False),
    user: AuthenticatedUser = Depends(get_current_user),
    jobs: Any = Depends(get_v4_job_service),
) -> dict[str, Any]:
    job = jobs.get(_scope(user, cross_owner), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _job_payload(job)


@router.delete("/jobs/{job_id}")
def cancel_deep_job(
    job_id: UUID,
    cross_owner: bool = Query(False),
    user: AuthenticatedUser = Depends(get_current_user),
    jobs: Any = Depends(get_v4_job_service),
) -> dict[str, bool]:
    if not jobs.cancel(_scope(user, cross_owner), job_id, now=datetime.now(UTC)):
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"cancelled": True}
