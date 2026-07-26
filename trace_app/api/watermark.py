from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import (
    DEFAULT_ROBUST_WATERMARK_STRENGTH,
    DEFAULT_ROBUST_WATERMARK_VERSION,
)
from trace_app.dependencies import get_optional_current_user, get_watermark_service
from trace_app.media import sign_media_url
from trace_app.watermark.service import WatermarkService

router = APIRouter(prefix="/api/watermark", tags=["watermark"])


EMBED_PUBLIC_FIELDS = (
    "id",
    "user_id",
    "trace_id",
    "evidence_uuid_head",
    "evidence_uuid_tail",
    "mode",
    "mode_label",
    "created_at",
    "status",
    "confidence",
    "robust_watermark_version",
)
EXTRACT_PUBLIC_FIELDS = (
    "user_id",
    "trace_id",
    "evidence_uuid_head",
    "evidence_uuid_tail",
    "mode",
    "mode_label",
    "created_at",
    "extracted_at",
    "status",
    "confidence",
    "phash_match",
    "robust_watermark_version",
)


def _signed_access(request: Request, media_url: Any) -> str | None:
    if not isinstance(media_url, str) or not media_url.startswith("/uploads/"):
        return None
    return sign_media_url(
        media_url,
        request.app.state.media_signing_key,
        ttl_seconds=request.app.state.media_url_ttl_seconds,
    )


def _public_response(
    payload: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    return {key: payload[key] for key in fields if key in payload}


def _public_embed_response(
    request: Request, payload: dict[str, Any]
) -> dict[str, Any]:
    result = _public_response(payload, EMBED_PUBLIC_FIELDS)
    for source, target in (
        ("original_url", "original_access_url"),
        ("download_url", "download_access_url"),
    ):
        access_url = _signed_access(request, payload.get(source))
        if access_url:
            result[target] = access_url
    return result


def _public_extract_response(
    request: Request, payload: dict[str, Any]
) -> dict[str, Any]:
    result = _public_response(payload, EXTRACT_PUBLIC_FIELDS)
    access_url = _signed_access(request, payload.get("matched_file_url"))
    if access_url:
        result["matched_file_access_url"] = access_url
    return result


@router.post("/embed")
async def embed_watermark(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    mode: str = Form("dct"),
    copyright_enabled: str = Form("false"),
    copyright_text: str = Form("© QQ:757675150"),
    copyright_opacity: str = Form("0.16"),
    copyright_complexity: str = Form("medium"),
    copyright_irregular_enabled: str = Form("true"),
    copyright_prominent_corner_enabled: str = Form("false"),
    fidelity_level: str = Form("0.75"),
    robust_watermark_strength: str = Form(DEFAULT_ROBUST_WATERMARK_STRENGTH),
    robust_watermark_version: str = Form(DEFAULT_ROBUST_WATERMARK_VERSION),
    small_crop_trace_enabled: str = Form(""),
    small_crop_trace_strength: str = Form("1.0"),
    small_crop_trace_density: str = Form("high"),
    dot_matrix_trace_enabled: str = Form("false"),
    dot_matrix_trace_strength: str = Form("0.85"),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    result = await service.embed(
        file=file,
        owner_user_id=None if current_user is None else current_user.id,
        user_id=user_id,
        mode=mode,
        copyright_enabled=copyright_enabled,
        copyright_text=copyright_text,
        copyright_opacity=copyright_opacity,
        copyright_complexity=copyright_complexity,
        copyright_irregular_enabled=copyright_irregular_enabled,
        copyright_prominent_corner_enabled=copyright_prominent_corner_enabled,
        fidelity_level=fidelity_level,
        robust_watermark_strength=robust_watermark_strength,
        robust_watermark_version=robust_watermark_version,
        small_crop_trace_enabled=small_crop_trace_enabled,
        small_crop_trace_strength=small_crop_trace_strength,
        small_crop_trace_density=small_crop_trace_density,
        dot_matrix_trace_enabled=dot_matrix_trace_enabled,
        dot_matrix_trace_strength=dot_matrix_trace_strength,
    )
    return _public_embed_response(request, result)


@router.post("/extract")
async def extract_watermark(
    request: Request,
    file: UploadFile = File(...),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    return _public_extract_response(request, await service.extract_upload(file))


@router.post("/extract-url")
def extract_watermark_url(
    request: Request,
    url: str = Form(...),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    return _public_extract_response(request, service.extract_url(url))
