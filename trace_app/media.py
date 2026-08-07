import hashlib
import hmac
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit

from fastapi import HTTPException

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.database.repositories import Repository


MEDIA_DIRECTORIES = frozenset({"originals", "watermarked", "thumbnails"})
MEDIA_URL_FIELDS = ("original_url", "download_url", "thumbnail_url")
MEDIA_ACCESS_FIELDS = {
    "original_url": "original_access_url",
    "download_url": "download_access_url",
    "thumbnail_url": "thumbnail_access_url",
    "matched_file_url": "matched_file_access_url",
}


def media_path_from_url(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/uploads/"):
        raise HTTPException(status_code=400, detail="图片链接路径无效")
    return unquote(parsed.path.removeprefix("/uploads/"))


def resolve_media_path(upload_dir: Path, media_path: str) -> Path:
    normalized = unquote(str(media_path or ""))
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or "\\" in normalized
        or relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] not in MEDIA_DIRECTORIES
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise HTTPException(status_code=400, detail="图片链接路径无效")

    root = upload_dir.resolve()
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="图片链接路径无效") from exc
    return target


def derive_media_signing_key(configured_key: str, fallback_key: bytes) -> bytes:
    encoded = str(configured_key or "").encode("utf-8")
    if len(encoded) < 32:
        return fallback_key
    return hmac.new(
        encoded,
        b"trace-media-url-signing-v1",
        hashlib.sha256,
    ).digest()


def _signature_payload(media_url: str, expires: int) -> bytes:
    return f"v1\n{expires}\n{media_url}".encode("utf-8")


def sign_expiring_url(
    access_path: str,
    key: bytes,
    *,
    ttl_seconds: int,
    now: int | float | None = None,
) -> str:
    parsed = urlsplit(str(access_path or ""))
    if parsed.scheme or parsed.netloc or parsed.query or not parsed.path.startswith("/"):
        raise HTTPException(status_code=400, detail="图片链接路径无效")
    current_time = time.time() if now is None else now
    expire_time = int(current_time) + max(1, int(ttl_seconds))
    signature = hmac.new(
        key,
        _signature_payload(parsed.path, expire_time),
        hashlib.sha256,
    ).hexdigest()
    return f"{parsed.path}?{urlencode({'expire_time': expire_time, 'signature': signature})}"


def sign_media_url(
    media_url: str,
    key: bytes,
    *,
    ttl_seconds: int,
    now: int | float | None = None,
) -> str:
    media_path = media_path_from_url(media_url)
    canonical_url = f"/uploads/{media_path}"
    return sign_expiring_url(
        canonical_url,
        key,
        ttl_seconds=ttl_seconds,
        now=now,
    )


def verify_media_signature(
    media_url: str,
    *,
    expires: str | int | None,
    signature: str | None,
    key: bytes,
    now: int | float | None = None,
) -> bool:
    try:
        expires_at = int(expires)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    current_time = time.time() if now is None else now
    if current_time > expires_at or not signature:
        return False
    expected = hmac.new(
        key,
        _signature_payload(media_url, expires_at),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, str(signature))


def with_media_access_urls(
    payload: dict[str, Any],
    *,
    key: bytes,
    ttl_seconds: int,
) -> dict[str, Any]:
    result = dict(payload)
    for source_field, access_field in MEDIA_ACCESS_FIELDS.items():
        value = result.get(source_field)
        if isinstance(value, str) and value.startswith("/uploads/"):
            result[access_field] = sign_media_url(
                value,
                key,
                ttl_seconds=ttl_seconds,
            )
    return result


def user_can_access_media(
    repository: Repository,
    current_user: AuthenticatedUser,
    media_url: str,
) -> bool:
    owner_user_id = None if current_user.role == "admin" else current_user.id
    records = repository.read_records(owner_user_id=owner_user_id)
    return any(
        record.get(field) == media_url
        for record in records
        for field in MEDIA_URL_FIELDS
    )
