import hashlib
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from trace_app.imaging.io import load_image_from_bytes


def file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest().upper()


def path_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


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
    file_md5_fn: Callable[[bytes], str] | None = None,
    file_sha256_fn: Callable[[bytes], str] | None = None,
    image_content_sha256_fn: Callable[[Image.Image], str] | None = None,
    load_image_from_bytes_fn: Callable[[bytes], Image.Image] | None = None,
) -> dict[str, Any] | None:
    hash_md5 = file_md5_fn or file_md5
    hash_file = file_sha256_fn or file_sha256
    hash_image = image_content_sha256_fn or image_content_sha256
    load_image = load_image_from_bytes_fn or load_image_from_bytes
    md5_digest = hash_md5(content)
    sha256_digest = hash_file(content)
    records = read_records()

    def match_result(
        record: dict[str, Any],
        file_type: str,
        matched_hash_type: str,
        matched_hash: str,
        image_hash: str | None,
    ) -> dict[str, Any]:
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
            "file_md5": md5_digest,
            "file_hash": sha256_digest,
            "image_hash": image_hash,
            "matched_hash": matched_hash,
            "matched_hash_type": matched_hash_type,
            "matched_file_type": file_type,
            "matched_file_url": record.get(
                "original_url" if file_type == "original" else "download_url"
            ),
            "watermark_layers": record.get("watermark_layers", watermark_layers),
            "layer_scores": {},
        }, record)

    for record in records:
        for file_type in ("original", "watermarked"):
            stored_md5 = str(
                record.get(f"{file_type}_file_md5") or ""
            ).upper()
            stored_sha256 = str(
                record.get(f"{file_type}_file_sha256") or ""
            ).upper()
            if (
                stored_md5 == md5_digest
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                return match_result(
                    record,
                    file_type,
                    "file_md5_sha256",
                    sha256_digest,
                    None,
                )
            if (
                not stored_md5
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                return match_result(
                    record,
                    file_type,
                    "file_sha256",
                    sha256_digest,
                    None,
                )

    query_image_digest = None
    for record in records:
        for file_type in ("original", "watermarked"):
            stored_image_digest = str(
                record.get(f"{file_type}_image_sha256") or ""
            ).upper()
            if not stored_image_digest:
                continue
            try:
                if query_image_digest is None:
                    query_image_digest = hash_image(load_image(content))
            except Exception:
                return None
            if stored_image_digest == query_image_digest:
                return match_result(
                    record,
                    file_type,
                    "image_pixels",
                    query_image_digest,
                    query_image_digest,
                )
    return None
