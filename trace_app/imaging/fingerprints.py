import hashlib
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from trace_app.imaging.io import load_image_from_bytes


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def path_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def image_content_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest().upper()


def matched_file_fingerprint(
    content: bytes,
    *,
    read_records: Callable[[], list[dict[str, Any]]],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
    watermark_layers: list[str],
    file_sha256_fn: Callable[[bytes], str] = file_sha256,
    image_content_sha256_fn: Callable[[Image.Image], str] = image_content_sha256,
    load_image_from_bytes_fn: Callable[[bytes], Image.Image] = load_image_from_bytes,
) -> dict[str, Any] | None:
    digest = file_sha256_fn(content)
    query_image_digest = None
    for record in read_records():
        for file_type in ("original", "watermarked"):
            stored_file_digest = str(
                record.get(f"{file_type}_file_sha256") or ""
            ).upper()
            stored_image_digest = str(
                record.get(f"{file_type}_image_sha256") or ""
            ).upper()
            if stored_file_digest and stored_file_digest == digest:
                matched_hash_type = "file_bytes"
                matched_hash = digest
            elif stored_image_digest:
                try:
                    if query_image_digest is None:
                        query_image_digest = image_content_sha256_fn(
                            load_image_from_bytes_fn(content)
                        )
                except Exception:
                    return None
                if stored_image_digest != query_image_digest:
                    continue
                matched_hash_type = "image_pixels"
                matched_hash = query_image_digest
            else:
                continue
            return with_evidence_fields({
                "id": record.get("id"),
                "trace_id": record.get("trace_id"),
                "user_id": record.get("user_id"),
                "mode": "file_fingerprint",
                "mode_label": "文件指纹一样",
                "created_at": record.get("created_at"),
                "confidence": 100,
                "phash_match": False,
                "status": "文件指纹一样",
                "extracted_at": now_text(),
                "file_hash": digest,
                "image_hash": query_image_digest,
                "matched_hash": matched_hash,
                "matched_hash_type": matched_hash_type,
                "matched_file_type": file_type,
                "matched_file_url": record.get(
                    "original_url" if file_type == "original" else "download_url"
                ),
                "watermark_layers": record.get("watermark_layers", watermark_layers),
                "layer_scores": {},
            }, record)
    return None
