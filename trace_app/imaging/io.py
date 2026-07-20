import ipaddress
from io import BytesIO
from pathlib import Path
import socket
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib import request as urllib_request

from fastapi import HTTPException, UploadFile
from PIL import Image

from trace_app.media import media_path_from_url, resolve_media_path


REMOTE_IMAGE_MAX_BYTES = 20 * 1024 * 1024
REMOTE_IMAGE_MAX_REDIRECTS = 4
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any):
        return None


def _default_open_url(request: urllib_request.Request, *, timeout: float):
    opener = urllib_request.build_opener(
        urllib_request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


def _validate_public_http_url(
    url: str,
    *,
    resolve_host: Callable[..., Any],
) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise HTTPException(status_code=400, detail="仅支持公网 http(s) 图片链接")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = resolve_host(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址") from exc
    if not addresses:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址")
    try:
        resolved = {
            ipaddress.ip_address(str(item[4][0]).split("%", 1)[0])
            for item in addresses
        }
    except (IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址") from exc
    if any(not address.is_global for address in resolved):
        raise HTTPException(status_code=400, detail="不允许访问内网或本机地址")


def fetch_remote_image_bytes(
    url: str,
    *,
    resolve_host: Callable[..., Any] | None = None,
    open_url: Callable[..., Any] | None = None,
) -> tuple[bytes, str]:
    resolver = resolve_host or socket.getaddrinfo
    opener = open_url or _default_open_url
    current_url = str(url or "").strip()

    for redirect_count in range(REMOTE_IMAGE_MAX_REDIRECTS + 1):
        _validate_public_http_url(current_url, resolve_host=resolver)
        request = urllib_request.Request(
            current_url,
            headers={"User-Agent": "WatermarkSystem/1.0"},
        )
        try:
            response = opener(request, timeout=10)
        except HTTPError as exc:
            if exc.code not in REDIRECT_STATUSES:
                raise HTTPException(status_code=400, detail="无法读取图片链接") from exc
            location = exc.headers.get("location", "")
            exc.close()
            if not location or redirect_count >= REMOTE_IMAGE_MAX_REDIRECTS:
                raise HTTPException(status_code=400, detail="图片链接重定向无效")
            current_url = urljoin(current_url, location)
            continue
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="无法读取图片链接") from exc

        with response:
            status = getattr(response, "status", None)
            if status in REDIRECT_STATUSES:
                location = response.headers.get("location", "")
                if not location or redirect_count >= REMOTE_IMAGE_MAX_REDIRECTS:
                    raise HTTPException(status_code=400, detail="图片链接重定向无效")
                current_url = urljoin(current_url, location)
                continue
            if status is not None and not 200 <= int(status) < 300:
                raise HTTPException(status_code=400, detail="无法读取图片链接")

            content_type = str(response.headers.get("content-type", ""))
            content_length = response.headers.get("content-length")
            try:
                if content_length is not None and int(content_length) > REMOTE_IMAGE_MAX_BYTES:
                    raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="图片链接响应无效") from exc
            data = response.read(REMOTE_IMAGE_MAX_BYTES + 1)

        if len(data) > REMOTE_IMAGE_MAX_BYTES:
            raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
        if content_type and not content_type.lower().split(";", 1)[0].strip().startswith(
            "image/"
        ):
            raise HTTPException(status_code=400, detail="链接内容不是图片")
        return data, content_type

    raise HTTPException(status_code=400, detail="图片链接重定向无效")


async def load_upload_image(
    file: UploadFile,
    *,
    load_image_from_bytes_fn: Callable[[bytes], Image.Image] | None = None,
) -> Image.Image:
    content = await file.read()
    loader = load_image_from_bytes_fn or load_image_from_bytes
    return loader(content)


def load_image_from_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片") from exc
    return image


def save_thumbnail(image: Image.Image, path: Path, scale: float = 0.20) -> None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    thumbnail = rgb.resize(thumb_size, Image.Resampling.LANCZOS)
    thumbnail.save(path, format="PNG", optimize=True)


def load_image_from_url(url: str, upload_dir: Path) -> Image.Image:
    text = str(url or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入图片链接")
    if text.startswith("/uploads/"):
        path = resolve_media_path(upload_dir, media_path_from_url(text))
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="图片链接不存在")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
    data, _content_type = fetch_remote_image_bytes(text)
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
