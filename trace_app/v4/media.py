"""Opaque media mappings and scoped signed access URLs."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from fastapi import HTTPException

from trace_app.v4.repository import StoredMediaObject, V4Repository


_PREFIXES = {
    "original": "originals",
    "watermarked": "watermarked",
    "thumbnail": "thumbnails",
}


def _canonical_fields(*values: str) -> bytes:
    payload = bytearray()
    for value in values:
        encoded = value.encode("utf-8")
        payload.extend(len(encoded).to_bytes(4, "big"))
        payload.extend(encoded)
    return bytes(payload)


class V4MediaService:
    def __init__(
        self,
        repository: V4Repository,
        *,
        storage_root: Path,
        signing_key: bytes,
        public_base_url: str = "",
        default_ttl_seconds: int = 300,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("V4 media signing key must contain at least 32 bytes")
        if default_ttl_seconds <= 0:
            raise ValueError("V4 media URL TTL must be positive")
        self.repository = repository
        self.storage_root = Path(storage_root).resolve(strict=False)
        self.signing_key = bytes(signing_key)
        self.public_base_url = public_base_url.rstrip("/")
        self.default_ttl_seconds = default_ttl_seconds

    def put_bytes(
        self,
        *,
        owner_user_id: int,
        variant: str,
        content_type: str,
        content: bytes,
    ) -> StoredMediaObject:
        staged = self.stage_bytes(
            owner_user_id=owner_user_id,
            variant=variant,
            content_type=content_type,
            content=content,
        )
        try:
            stored = self.repository.insert_media(staged.media_input)
            self.promote(staged)
            return stored
        except Exception:
            self.discard(staged)
            raise

    def stage_bytes(
        self,
        *,
        owner_user_id: int,
        variant: str,
        content_type: str,
        content: bytes,
    ):
        from trace_app.v4.generation import StagedMedia

        prefix = _PREFIXES.get(variant)
        if prefix is None:
            raise ValueError("unsupported V4 media variant")
        object_name = secrets.token_hex(32)
        storage_key = f"{prefix}/{object_name[:2]}/{object_name}.bin"
        target = self.resolve_storage_key(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=".v4-media-", delete=False
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
            temporary_path = None
            return StagedMedia(
                media_id=secrets.token_urlsafe(16),
                owner_user_id=owner_user_id,
                variant=variant,
                storage_key=storage_key,
                content_type=content_type,
                content=bytes(content),
            )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def discard(self, item) -> None:
        self.resolve_storage_key(item.storage_key).unlink(missing_ok=True)

    def promote(self, item) -> None:
        # The opaque mapping becomes visible only when its database transaction commits.
        return None

    def sign(
        self,
        media: StoredMediaObject,
        *,
        ttl_seconds: int | None = None,
        now: int | float | None = None,
    ) -> tuple[int, str]:
        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        if ttl <= 0:
            raise ValueError("V4 media URL TTL must be positive")
        expires = int(time.time() if now is None else now) + ttl
        signature = hmac.new(
            self.signing_key,
            self._signature_payload(media, expires),
            hashlib.sha256,
        ).hexdigest()
        return expires, signature

    def verify(
        self,
        media: StoredMediaObject,
        *,
        expires: int,
        signature: str,
        now: int | float | None = None,
    ) -> bool:
        current = int(time.time() if now is None else now)
        if current > int(expires):
            return False
        expected = hmac.new(
            self.signing_key,
            self._signature_payload(media, int(expires)),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, str(signature))

    def issue_url(
        self,
        media_id: str,
        *,
        requester_user_id: int,
        requester_is_admin: bool,
    ) -> str:
        media = self.get_media_or_404(media_id)
        if not requester_is_admin and media.owner_user_id != requester_user_id:
            raise HTTPException(status_code=404, detail="媒体不存在")
        expires, signature = self.sign(media)
        query = urlencode({"expires": expires, "signature": signature})
        return f"{self.public_base_url}/api/media/{media.id}?{query}"

    def get_media_or_404(self, media_id: str) -> StoredMediaObject:
        media = self.repository.get_media(media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="媒体不存在")
        return media

    def resolve_storage_key(self, storage_key: str) -> Path:
        if not storage_key or "\\" in storage_key:
            raise HTTPException(status_code=404, detail="媒体不存在")
        logical = PurePosixPath(storage_key)
        parts = logical.parts
        if logical.is_absolute() or len(parts) < 3 or any(
            part in {"", ".", ".."} for part in parts
        ):
            raise HTTPException(status_code=404, detail="媒体不存在")
        if parts[0] not in _PREFIXES.values():
            raise HTTPException(status_code=404, detail="媒体不存在")

        root = self.storage_root
        candidate = root.joinpath(*parts)
        paths = [root]
        paths.extend(root.joinpath(*parts[:index]) for index in range(1, len(parts) + 1))
        if any(path.exists() and path.is_symlink() for path in paths):
            raise HTTPException(status_code=404, detail="媒体不存在")
        try:
            candidate.resolve(strict=False).relative_to(root)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="媒体不存在") from error
        return candidate

    @staticmethod
    def _signature_payload(media: StoredMediaObject, expires: int) -> bytes:
        return _canonical_fields(
            "v4-media-url-v1",
            media.id,
            media.variant,
            str(media.owner_user_id),
            str(expires),
        )


__all__ = ("V4MediaService",)
