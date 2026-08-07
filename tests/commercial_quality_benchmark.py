import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/quality_benchmark/runtime/uploads"
os.environ["DATA_DIR"] = "test_output/quality_benchmark/runtime/data"

from tests.commercial_quality_metrics import metric_distribution, quality_gate, quality_metrics
from tests.commercial_benchmark_config import build_embedding_form
from tests.commercial_report_contract import build_report_metadata, validate_report


OUTPUT_DIR = ROOT / "test_output" / "commercial_quality_benchmark"
RUNTIME_DIR = ROOT / "test_output" / "quality_benchmark" / "runtime"
JSON_PATH = OUTPUT_DIR / "commercial_quality_results.json"
CSV_PATH = OUTPUT_DIR / "commercial_quality_results.csv"
REPORT_PATH = OUTPUT_DIR / "commercial_quality_test_report.md"

FIDELITY_LEVELS = [
    float(item.strip())
    for item in os.getenv("FIDELITY_LEVELS", "0.70,0.75,0.80,0.85,0.90").split(",")
    if item.strip()
]
QUALITY_MIN_PSNR = float(os.getenv("QUALITY_MIN_PSNR", "38.0"))
QUALITY_MIN_SSIM = float(os.getenv("QUALITY_MIN_SSIM", "0.95"))
PROBE_MIN_RECALL = float(os.getenv("PROBE_MIN_RECALL", "0.95"))
SMALL_CROP_TRACE_STRENGTH = os.getenv("SMALL_CROP_TRACE_STRENGTH", "0.35")
SMALL_CROP_TRACE_DENSITY = os.getenv("SMALL_CROP_TRACE_DENSITY", "medium")
ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260707"))


def build_report(
    summary: dict,
    cases: list,
    seed: int,
    algorithm_version: str,
    *,
    settings: dict,
    verdict: str,
    failed_gates: list,
    **extra,
) -> dict:
    return {
        **extra,
        "metadata": build_report_metadata("quality", seed, algorithm_version),
        "summary": summary,
        "cases": cases,
        "settings": settings,
        "verdict": verdict,
        "failed_gates": failed_gates,
    }


def select_recommended_config(
    configs: list[dict[str, Any]],
    min_probe_recall: float = PROBE_MIN_RECALL,
) -> dict[str, Any] | None:
    accepted = [
        config
        for config in configs
        if config.get("quality_pass")
        and int(config.get("wrong", 0)) == 0
        and int(config.get("false_positive", 0)) == 0
        and float(config.get("probe_recall", 0.0)) >= min_probe_recall
    ]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda item: (
            float(item["min_ssim"]),
            float(item["min_psnr"]),
            float(item.get("probe_recall", 0.0)),
            float(item.get("fidelity", 0.0)),
        ),
    )


def rejected_reasons(config: dict[str, Any]) -> list[str]:
    reasons = []
    if not config["quality_pass"]:
        reasons.append("quality_gate")
    if config["wrong"]:
        reasons.append("wrong_trace")
    if config["false_positive"]:
        reasons.append("false_positive")
    if config["probe_recall"] < PROBE_MIN_RECALL:
        reasons.append("probe_recall")
    return reasons


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def center_crop(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * ratio)))
    crop_height = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def scaled_crop(image: Image.Image, scale: float, crop_ratio: float) -> Image.Image:
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return center_crop(image.resize(size, Image.Resampling.BICUBIC), crop_ratio)


def wechat_sim(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    scale = min(1.0, 1440 / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
            Image.Resampling.BICUBIC,
        )
    return jpeg_roundtrip(rgb, 78).filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=2)
    )


def screen_photo_sim(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dx, dy = width * 0.035, height * 0.028
    target = np.float32([
        [dx, dy * 1.5],
        [width - 1 - dx * 0.6, dy],
        [width - 1 - dx, height - 1 - dy],
        [dx * 0.7, height - 1 - dy * 0.6],
    ])
    warped = cv2.warpPerspective(
        array,
        cv2.getPerspectiveTransform(source, target),
        (width, height),
        borderValue=(245, 245, 245),
    )
    return jpeg_roundtrip(Image.fromarray(warped).filter(ImageFilter.GaussianBlur(0.45)), 86)


PROBES: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
    ("intact", lambda image: image.convert("RGB")),
    ("scale_0.5_crop_0.5", lambda image: scaled_crop(image, 0.5, 0.5)),
    ("scale_1.5_crop_0.3", lambda image: scaled_crop(image, 1.5, 0.3)),
    ("jpeg_q30", lambda image: jpeg_roundtrip(image, 30)),
    ("wechat_sim", wechat_sim),
    ("screen_photo_sim", screen_photo_sim),
]
PROBE_FILTER = {
    item.strip()
    for item in os.getenv("QUALITY_PROBE_FILTER", "").split(",")
    if item.strip()
}
if PROBE_FILTER:
    PROBES = [probe for probe in PROBES if probe[0] in PROBE_FILTER]
if not PROBES:
    raise ValueError("QUALITY_PROBE_FILTER did not match any quality probes")


def reset_runtime(main_module) -> None:
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    main_module.app.state.generated_trace_ids = []
    main_module.ensure_dirs()


def upload_path(main_module, url: str) -> Path:
    return main_module.UPLOAD_DIR / url.replace("/uploads/", "")


def run_detection(main_module, image: Image.Image) -> tuple[bool, dict[str, Any], float]:
    started = time.perf_counter()
    try:
        result = main_module.extract_watermark_from_image(image)
        success = True
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
        result = {"detail": getattr(exc, "detail", str(exc))}
        success = False
    return success, result, round((time.perf_counter() - started) * 1000, 3)


def embed_sources(main_module, client, sources: list[Path], fidelity: float) -> list[dict[str, Any]]:
    records = []
    for source in sources:
        with source.open("rb") as file_handle:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, file_handle, "image/png")},
                data=build_embedding_form("commercial-quality-benchmark", fidelity),
            )
        response.raise_for_status()
        records.append(response.json())
    return records


def run_config(main_module, client, sources: list[Path], fidelity: float) -> dict[str, Any]:
    reset_runtime(main_module)
    config_dir = OUTPUT_DIR / f"fidelity-{fidelity:.2f}"
    config_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = embed_sources(main_module, client, sources, fidelity)
    quality_rows = []
    probe_rows = []

    for source, record in zip(sources, records):
        original = Image.open(source).convert("RGB")
        watermarked_path = upload_path(main_module, record["download_url"])
        watermarked = Image.open(watermarked_path).convert("RGB")
        local_watermarked = config_dir / f"{source.stem}-{record['trace_id']}.png"
        shutil.copy2(watermarked_path, local_watermarked)
        metrics = quality_metrics(original, watermarked)
        quality_rows.append({
            "source": source.name,
            "trace_id": record["trace_id"],
            "output": str(local_watermarked.relative_to(ROOT)),
            "size_delta_bytes": local_watermarked.stat().st_size - source.stat().st_size,
            "pass": quality_gate(metrics, QUALITY_MIN_PSNR, QUALITY_MIN_SSIM),
            **metrics,
        })

        for probe_name, probe in PROBES:
            for case_type, image in (("watermarked", watermarked), ("unwatermarked", original)):
                success, detected, detection_ms = run_detection(main_module, probe(image))
                expected_trace = record["trace_id"] if case_type == "watermarked" else ""
                detected_trace = detected.get("trace_id", "") if success else ""
                probe_rows.append({
                    "source": source.name,
                    "probe": probe_name,
                    "case_type": case_type,
                    "expected_trace": expected_trace,
                    "detected_trace": detected_trace,
                    "success": success,
                    "correct_trace": success and detected_trace == expected_trace if expected_trace else not success,
                    "mode": detected.get("mode", "") if success else "",
                    "confidence": detected.get("confidence", "") if success else "",
                    "detection_ms": detection_ms,
                })

    positives = [row for row in probe_rows if row["case_type"] == "watermarked"]
    negatives = [row for row in probe_rows if row["case_type"] == "unwatermarked"]
    correct = sum(1 for row in positives if row["correct_trace"])
    wrong = sum(1 for row in positives if row["success"] and not row["correct_trace"])
    false_positive = sum(1 for row in negatives if row["success"])
    config = {
        "fidelity": fidelity,
        "quality_pass": all(row["pass"] for row in quality_rows),
        "min_psnr": min(row["psnr"] for row in quality_rows),
        "min_ssim": min(row["ssim"] for row in quality_rows),
        "psnr_distribution": metric_distribution(quality_rows, "psnr"),
        "ssim_distribution": metric_distribution(quality_rows, "ssim"),
        "probe_total": len(positives),
        "probe_correct": correct,
        "probe_recall": round(correct / len(positives), 6) if positives else 0.0,
        "wrong": wrong,
        "negative_total": len(negatives),
        "false_positive": false_positive,
        "quality_rows": quality_rows,
        "probe_rows": probe_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    config["rejected_reasons"] = rejected_reasons(config)
    return config


def write_outputs(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = JSON_PATH.with_suffix(".json.tmp")
    errors = validate_report(payload)
    if errors:
        raise ValueError("invalid commercial report: " + ", ".join(errors))
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(JSON_PATH)

    fields = [
        "fidelity", "source", "trace_id", "pass", "psnr", "ssim", "mae", "rmse",
        "max_abs_diff", "size_delta_bytes", "output",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for config in payload["configs"]:
            for row in config["quality_rows"]:
                writer.writerow({"fidelity": config["fidelity"], **{field: row.get(field, "") for field in fields[1:]}})

    lines = [
        "# 商用水印画质与追溯联合扫描报告",
        "",
        f"- 测试时间：{payload['created_at']}",
        f"- 图片：{len(payload['sources'])} 张，来自 `img/`",
        f"- 质量门槛：PSNR >= {QUALITY_MIN_PSNR} dB，SSIM >= {QUALITY_MIN_SSIM}",
        f"- 代表性探针召回门槛：{PROBE_MIN_RECALL:.2%}",
        f"- 推荐保真度：{payload['recommended_fidelity'] if payload['recommended_fidelity'] is not None else '无'}",
        f"- 结论：{payload['verdict']}",
        "",
        "| 保真度 | 最低 PSNR | 最低 SSIM | 探针召回 | 错 trace | 负样本误报 | 质量通过 | 淘汰原因 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for config in payload["configs"]:
        lines.append(
            f"| {config['fidelity']:.2f} | {config['min_psnr']:.3f} | {config['min_ssim']:.6f} | "
            f"{config['probe_recall']:.2%} | {config['wrong']} | {config['false_positive']} | "
            f"{'是' if config['quality_pass'] else '否'} | {', '.join(config['rejected_reasons']) or '-'} |"
        )
    lines.extend([
        "",
        "推荐配置是在零错误 trace、零负样本误报、质量和探针召回全部达标的配置中，按最低 SSIM、最低 PSNR 选择图像损伤最小的一档。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_run() -> int:
    import main
    from fastapi.testclient import TestClient

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "img").glob("*.png"))
    if not sources:
        raise RuntimeError("no PNG sources found in img/")

    started = time.perf_counter()
    client = TestClient(main.app)
    configs = [run_config(main, client, sources, fidelity) for fidelity in FIDELITY_LEVELS]
    selected = select_recommended_config(configs)
    verdict = "PASS" if selected else "FAIL"
    summary = {
        "recommended_fidelity": selected["fidelity"] if selected else None,
        "verdict": verdict,
        "config_count": len(configs),
    }
    payload = build_report(
        summary,
        configs,
        seed=RANDOM_SEED,
        algorithm_version=ROBUST_WATERMARK_VERSION,
        settings={
            "fidelity_levels": FIDELITY_LEVELS,
            "quality_min_psnr": QUALITY_MIN_PSNR,
            "quality_min_ssim": QUALITY_MIN_SSIM,
            "probe_min_recall": PROBE_MIN_RECALL,
            "small_crop_trace_strength": SMALL_CROP_TRACE_STRENGTH,
            "small_crop_trace_density": SMALL_CROP_TRACE_DENSITY,
            "robust_watermark_strength": ROBUST_WATERMARK_STRENGTH,
            "robust_watermark_version": ROBUST_WATERMARK_VERSION,
            "probes": [name for name, _ in PROBES],
        },
        verdict=verdict,
        failed_gates=[] if selected else ["no_approved_configuration"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sources=[source.name for source in sources],
        recommended_fidelity=selected["fidelity"] if selected else None,
        configs=configs,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    write_outputs(payload)
    print(json.dumps({
        "verdict": payload["verdict"],
        "recommended_fidelity": payload["recommended_fidelity"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main_run())
