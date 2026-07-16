import re
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
from trace_app.config import (
    FEATURE_MATCH_MIN_GOOD,
    FEATURE_RECENT_BACKFILL,
    FEATURE_RECENT_RESERVE,
    ROBUST_CHANNEL,
    WATERMARK_LAYERS,
)
from watermark_v4.features import (
    extract_feature_index as extract_v4_feature_index,
    save_feature_index as save_v4_feature_index,
)

cv2.setNumThreads(1)


def save_record_feature_index(image: Image.Image, record_id: str, data_dir: Path) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index") / f"{safe_id}.npz"
    descriptors = extract_feature_descriptors(image)
    save_feature_descriptors(data_dir / relative, descriptors)
    return relative.as_posix()


def save_record_feature_index_v4(image: Image.Image, record_id: str, data_dir: Path) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index_v4") / f"{safe_id}.npz"
    index = extract_v4_feature_index(image)
    save_v4_feature_index(data_dir / relative, index)
    return relative.as_posix()


def record_feature_index_path(record: dict[str, Any], data_dir: Path) -> Path | None:
    raw = str(record.get("feature_index_path") or "").strip()
    if raw:
        relative = Path(raw.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        return data_dir / relative
    record_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record.get("id") or ""))
    if not record_id:
        return None
    return data_dir / "feature_index" / f"{record_id}.npz"


def rank_aligned_candidates(
    image: Image.Image,
    records: list[dict[str, Any]],
    *,
    upload_dir: Path,
    data_dir: Path,
    generated_trace_ids: list[str],
) -> list[dict[str, Any]]:
    query_ratio = image.width / max(1, image.height)
    recent_trace_ids = list(generated_trace_ids[:FEATURE_RECENT_BACKFILL])
    for record in records:
        trace_id = record.get("trace_id")
        if len(recent_trace_ids) >= FEATURE_RECENT_BACKFILL:
            break
        if trace_id and record.get("created_at") and trace_id not in recent_trace_ids:
            recent_trace_ids.append(trace_id)
    recent_order = {
        trace_id: index
        for index, trace_id in enumerate(recent_trace_ids[:FEATURE_RECENT_RESERVE])
    }
    backfill_trace_ids = set(recent_trace_ids)

    for record in records:
        if record.get("trace_id") not in backfill_trace_ids:
            continue
        path = record_feature_index_path(record, data_dir)
        if path and path.exists():
            continue
        url = record.get("download_url")
        record_id = record.get("id")
        if not record_id or not url or not url.startswith("/uploads/"):
            continue
        image_path = upload_dir / url.replace("/uploads/", "")
        try:
            with Image.open(image_path) as target:
                save_record_feature_index(target.convert("RGB"), str(record_id), data_dir)
        except (OSError, ValueError):
            continue

    query_descriptors = extract_feature_descriptors(image)
    feature_ranked = []
    remaining = []
    for record in records:
        path = record_feature_index_path(record, data_dir)
        descriptors = (
            load_feature_descriptors(path)
            if path is not None and path.exists()
            else np.empty((0, 32), dtype=np.uint8)
        )
        match_count, match_quality = descriptor_match_score(query_descriptors, descriptors)
        if match_count >= FEATURE_MATCH_MIN_GOOD:
            feature_ranked.append({
                **record,
                "_feature_match_count": match_count,
                "_feature_match_quality": match_quality,
            })
        else:
            remaining.append(record)

    feature_ranked.sort(
        key=lambda record: (
            -int(record.get("_feature_match_count", 0)),
            -float(record.get("_feature_match_quality", 0.0)),
        )
    )

    def ratio_distance(record: dict[str, Any]) -> float:
        recorded_width = record.get("image_width")
        recorded_height = record.get("image_height")
        if recorded_width and recorded_height:
            try:
                target_ratio = float(recorded_width) / max(1.0, float(recorded_height))
                return abs(target_ratio - query_ratio)
            except (TypeError, ValueError):
                pass
        url = record.get("download_url")
        if not url or not url.startswith("/uploads/"):
            return float("inf")
        path = upload_dir / url.replace("/uploads/", "")
        try:
            with Image.open(path) as target:
                target_ratio = target.width / max(1, target.height)
        except Exception:
            return float("inf")
        return abs(target_ratio - query_ratio)

    feature_trace_ids = {record.get("trace_id") for record in feature_ranked}
    recent_ranked = sorted(
        [
            record
            for record in remaining
            if record.get("trace_id") in recent_order
            and record.get("trace_id") not in feature_trace_ids
        ],
        key=lambda record: recent_order[record.get("trace_id")],
    )[:FEATURE_RECENT_RESERVE]
    reserved_ids = {id(record) for record in recent_ranked}
    aspect_ranked = sorted(
        [record for record in remaining if id(record) not in reserved_ids],
        key=ratio_distance,
    )
    return feature_ranked + recent_ranked + aspect_ranked


def image_to_cv_gray(image: Image.Image, max_side: int = 1200):
    rgb = image.convert("RGB")
    arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    height, width = arr.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        arr = cv2.resize(arr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return arr


def record_visual_consistency(
    image: Image.Image,
    record: dict[str, Any],
    upload_dir: Path,
) -> tuple[bool, int, float, float]:
    url = record.get("download_url")
    original_url = record.get("original_url")
    if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
        return False, 0, 0.0, 0.0
    path = upload_dir / url.replace("/uploads/", "")
    original_path = upload_dir / original_url.replace("/uploads/", "")
    if not path.exists() or not original_path.exists():
        return False, 0, 0.0, 0.0
    try:
        query = image_to_cv_gray(image)
        target = image_to_cv_gray(Image.open(path))
    except Exception:
        return False, 0, 0.0, 0.0
    inliers, ratio = feature_match_score(query, target)
    residual_score = robust_residual_score(image, original_path, path, min_inliers=18, min_ratio=0.32)
    standard_match = inliers >= 18 and ratio >= 0.32 and residual_score >= 0.08
    strong_visual_small_crop_match = inliers >= 30 and ratio >= 0.65 and residual_score >= 0.06
    return (standard_match or strong_visual_small_crop_match), inliers, ratio, residual_score


def residual_candidate_evidence(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    record_visual_consistency_fn: Callable[
        [Image.Image, dict[str, Any]], tuple[bool, int, float, float]
    ],
) -> dict[str, Any] | None:
    records = [record for record in records if record.get("robust_watermark")]
    if not records:
        return None

    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    best_residual = 0.0
    for record in records:
        consistent, inliers, ratio, residual_score = record_visual_consistency_fn(image, record)
        if not consistent:
            continue
        if residual_score > best_residual or (
            residual_score == best_residual and (inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio))
        ):
            best_record = record
            best_inliers = inliers
            best_ratio = ratio
            best_residual = residual_score

    if not best_record or best_residual < 0.12:
        return None

    return {
        "candidate_id": best_record.get("id"),
        "candidate_trace_id": best_record.get("trace_id"),
        "visual_inliers": best_inliers,
        "visual_ratio": round(best_ratio, 3),
        "residual_score": round(best_residual, 4),
    }


def detect_by_residual_match(image: Image.Image) -> dict[str, Any] | None:
    # Visual and residual similarity can rank candidates but cannot prove that the
    # query contains a watermark. Code-backed detectors perform final attribution.
    return None


def feature_match_score(query_gray, target_gray) -> tuple[int, float]:
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good.append(first)

    if len(good) < 10:
        return len(good), 0.0

    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None:
        return len(good), 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    return inliers, ratio


def feature_match_homography(query_gray, target_gray):
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return None, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good.append(first)

    if len(good) < 10:
        return None, len(good), 0.0

    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    query_to_target, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None or query_to_target is None:
        return None, len(good), 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    try:
        target_to_query = np.linalg.inv(query_to_target)
    except np.linalg.LinAlgError:
        return None, inliers, ratio
    return target_to_query, inliers, ratio


def align_query_to_record(
    image: Image.Image,
    record: dict[str, Any],
    upload_dir: Path,
) -> dict[str, Any] | None:
    url = record.get("download_url")
    if not url or not url.startswith("/uploads/"):
        return None
    target_path = upload_dir / url.replace("/uploads/", "")
    if not target_path.exists():
        return None
    try:
        query_image = resize_for_residual(image)
        original_target = Image.open(target_path).convert("RGB")
        target_image = resize_for_residual(original_target)
    except Exception:
        return None

    query = np.asarray(query_image, dtype=np.uint8)
    target = np.asarray(target_image, dtype=np.uint8)
    query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    target_to_query, inliers, ratio = feature_match_homography(query_gray, target_gray)
    if target_to_query is None or inliers < 18 or ratio < 0.32:
        return None
    try:
        query_to_target = np.linalg.inv(target_to_query)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(query_to_target).all() or abs(float(np.linalg.det(query_to_target))) < 1e-9:
        return None

    target_height, target_width = target.shape[:2]
    aligned = cv2.warpPerspective(
        query,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid_mask = cv2.warpPerspective(
        np.ones(query.shape[:2], dtype=np.uint8) * 255,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    coverage = float(valid_mask.mean())
    if coverage < 0.05 or coverage > 1.0:
        return None
    target_scale = target_width / max(1, original_target.width)
    return {
        "image": aligned,
        "valid_mask": valid_mask,
        "inliers": inliers,
        "ratio": round(ratio, 4),
        "coverage": round(coverage, 4),
        "target_scale": target_scale,
        "target_size": (target_width, target_height),
        "homography": query_to_target,
    }


def resize_for_residual(image: Image.Image, max_side: int = 1200) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    return rgb


def robust_residual_score(
    query_image: Image.Image,
    original_path: Path,
    watermarked_path: Path,
    min_inliers: int = 80,
    min_ratio: float = 0.80,
) -> float:
    query = np.array(resize_for_residual(query_image), dtype=np.float32)
    watermarked = np.array(resize_for_residual(Image.open(watermarked_path)), dtype=np.float32)
    original = np.array(resize_for_residual(Image.open(original_path)), dtype=np.float32)
    query_gray = cv2.cvtColor(query.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(watermarked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    homography, inliers, ratio = feature_match_homography(query_gray, target_gray)
    if homography is None or inliers < min_inliers or ratio < min_ratio:
        return 0.0

    query_height, query_width = query.shape[:2]
    warped_watermarked = cv2.warpPerspective(watermarked, homography, (query_width, query_height))
    warped_original = cv2.warpPerspective(original, homography, (query_width, query_height))
    valid = cv2.warpPerspective(
        np.ones(watermarked.shape[:2], dtype=np.uint8) * 255,
        homography,
        (query_width, query_height),
    ) > 0
    if int(valid.sum()) < query_width * query_height * 0.30:
        return 0.0

    expected = (warped_watermarked[:, :, ROBUST_CHANNEL] - warped_original[:, :, ROBUST_CHANNEL])[valid]
    observed = (query[:, :, ROBUST_CHANNEL] - warped_original[:, :, ROBUST_CHANNEL])[valid]
    expected = expected - expected.mean()
    observed = observed - observed.mean()
    expected_norm = float(np.linalg.norm(expected))
    observed_norm = float(np.linalg.norm(observed))
    if expected_norm < 1e-6 or observed_norm < 1e-6:
        return 0.0
    return float(np.dot(expected, observed) / (expected_norm * observed_norm))


def detect_by_visual_match(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    upload_dir: Path,
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
) -> dict[str, Any] | None:
    if not records:
        return None

    query = image_to_cv_gray(image)
    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    for record in records:
        if not record.get("robust_watermark"):
            continue
        url = record.get("download_url")
        original_url = record.get("original_url")
        if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
            continue
        path = upload_dir / url.replace("/uploads/", "")
        original_path = upload_dir / original_url.replace("/uploads/", "")
        if not path.exists() or not original_path.exists():
            continue
        try:
            target = image_to_cv_gray(Image.open(path))
        except Exception:
            continue
        inliers, ratio = feature_match_score(query, target)
        if inliers >= 80 and ratio >= 0.80:
            residual_score = robust_residual_score(image, original_path, path)
            if residual_score < 0.18:
                continue
        else:
            residual_score = 0.0
        if inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio):
            best_record = {**record, "_residual_score": residual_score}
            best_inliers = inliers
            best_ratio = ratio

    if not best_record or best_inliers < 80 or best_ratio < 0.80:
        return None

    confidence = min(96, max(75, int(75 + best_record.get("_residual_score", 0) * 25)))
    return with_evidence_fields({
        "id": best_record.get("id"),
        "trace_id": best_record.get("trace_id"),
        "user_id": best_record.get("user_id"),
        "mode": "robust_dct",
        "mode_label": "30% 局部截图匹配",
        "created_at": best_record.get("created_at"),
        "confidence": confidence,
        "phash_match": True,
        "status": "局部截图命中",
        "extracted_at": now_text(),
        "match_inliers": best_inliers,
        "match_ratio": round(best_ratio, 3),
        "watermark_layers": best_record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(float(best_record.get("_residual_score", 0.0)), 4),
            "dwt": round(float(best_record.get("_residual_score", 0.0)), 4),
            "fft": round(float(best_record.get("_residual_score", 0.0)), 4),
        },
    }, best_record)


def is_registered_original_image(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    upload_dir: Path,
) -> bool:
    query = np.array(image.convert("RGB"), dtype=np.int16)
    query_height, query_width = query.shape[:2]
    for record in records:
        original_url = record.get("original_url")
        if not original_url or not original_url.startswith("/uploads/"):
            continue
        original_path = upload_dir / original_url.replace("/uploads/", "")
        if not original_path.exists():
            continue
        try:
            with Image.open(original_path) as original:
                if original.size != (query_width, query_height):
                    continue
                original_arr = np.array(original.convert("RGB"), dtype=np.int16)
        except Exception:
            continue
        diff = np.abs(query - original_arr)
        if float(diff.mean()) <= 0.05 and int(diff.max()) <= 1:
            return True
    return False
