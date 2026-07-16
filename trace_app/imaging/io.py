from io import BytesIO
from pathlib import Path
from urllib import request as urllib_request

from fastapi import HTTPException, UploadFile
from PIL import Image


async def load_upload_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    return load_image_from_bytes(content)


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
        path = upload_dir / text.replace("/uploads/", "")
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="图片链接不存在")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
    if not text.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="仅支持 http(s) 或 /uploads/ 图片链接")
    try:
        request = urllib_request.Request(text, headers={"User-Agent": "WatermarkSystem/1.0"})
        with urllib_request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read(20 * 1024 * 1024 + 1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取图片链接") from exc
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
    if content_type and "image" not in content_type.lower():
        raise HTTPException(status_code=400, detail="链接内容不是图片")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
