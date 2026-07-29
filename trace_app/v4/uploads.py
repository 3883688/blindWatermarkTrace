"""Bounded streaming of untrusted V4 uploads to private temporary storage."""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


UPLOAD_CHUNK_BYTES = 1024 * 1024


class AsyncReadable(Protocol):
    async def read(self, size: int) -> bytes: ...


class UploadLimitExceeded(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StagedUpload:
    path: Path
    byte_size: int
    sha256: bytes
    private_dir: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.private_dir, ignore_errors=True)


async def stream_upload(
    upload: AsyncReadable,
    *,
    temp_root: Path,
    max_bytes: int,
    temp_quota_bytes: int | None = None,
) -> StagedUpload:
    if max_bytes <= 0:
        raise ValueError("upload byte limit must be positive")
    quota = max_bytes if temp_quota_bytes is None else min(max_bytes, temp_quota_bytes)
    if quota <= 0:
        raise ValueError("temporary disk quota must be positive")

    root = Path(temp_root)
    root.mkdir(parents=True, exist_ok=True)
    private_dir = Path(tempfile.mkdtemp(prefix="v4-upload-", dir=root))
    try:
        os.chmod(private_dir, 0o700)
    except OSError:
        pass
    path = private_dir / "upload.bin"
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("xb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                byte_size += len(chunk)
                if byte_size > quota:
                    raise UploadLimitExceeded("V4 upload exceeds byte quota")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        return StagedUpload(path, byte_size, digest.digest(), private_dir)
    except Exception:
        shutil.rmtree(private_dir, ignore_errors=True)
        raise


def decode_image_unbounded(content: bytes) -> Image.Image:
    """Decode inside an isolated worker; process resource limits are the boundary."""
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(BytesIO(content))
        image.load()
        return image
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


__all__ = (
    "StagedUpload",
    "UPLOAD_CHUNK_BYTES",
    "UploadLimitExceeded",
    "decode_image_unbounded",
    "stream_upload",
)
