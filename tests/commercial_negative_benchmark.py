import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/negative_benchmark/uploads"
os.environ["DATA_DIR"] = "test_output/negative_benchmark/data"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

import main
from tests.commercial_benchmark_config import build_embedding_form
from tests.commercial_report_contract import build_report_metadata, validate_report
from tests.commercial_attack_benchmark import (
    ATTACK_BY_NAME,
    upload_url_to_path,
)


OUTPUT_DIR = ROOT / "test_output" / "commercial_negative_benchmark"
FALSE_POSITIVE_DIR = OUTPUT_DIR / "false_positive"
REPORT_PATH = OUTPUT_DIR / "commercial_negative_test_report.md"
JSON_PATH = OUTPUT_DIR / "commercial_negative_results.json"
CSV_PATH = OUTPUT_DIR / "commercial_negative_results.csv"

NEGATIVE_ATTACKS = [
    item.strip()
    for item in os.getenv(
        "NEGATIVE_ATTACKS",
        "jpeg_q90,jpeg_q50,jpeg_q30,rotate_3deg,rotate_10deg,browser_screenshot_sim,wechat_screenshot_sim,screen_photo_sim,gaussian_blur_1_2,median_denoise",
    ).split(",")
    if item.strip()
]
SYNTHETIC_VARIANTS = int(os.getenv("SYNTHETIC_VARIANTS", "1000"))
BENCHMARK_WORKERS = int(os.getenv("BENCHMARK_WORKERS", str(min(12, max(1, (os.cpu_count() or 2) - 2)))))
FIDELITY_LEVEL = os.getenv("FIDELITY_LEVEL", "0.90")
SMALL_CROP_TRACE_STRENGTH = os.getenv("SMALL_CROP_TRACE_STRENGTH", "0.35")
SMALL_CROP_TRACE_DENSITY = os.getenv("SMALL_CROP_TRACE_DENSITY", "medium")
ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260707"))


def reset_state() -> None:
    for path in (main.UPLOAD_DIR, main.DATA_DIR, OUTPUT_DIR):
        if path.exists() and ROOT in path.resolve().parents:
            shutil.rmtree(path)
    main.app.state.generated_trace_ids = []
    FALSE_POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    main.ensure_dirs()


def seed_watermark_records(client: TestClient, sources: list[Path]) -> list[dict]:
    records = []
    for source in sources:
        with source.open("rb") as fp:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, fp, "image/png")},
                data=build_embedding_form("commercial-negative-benchmark", FIDELITY_LEVEL),
            )
        response.raise_for_status()
        records.append(response.json())
    return records


def save_seed_records(records: list[dict]) -> None:
    seed_dir = OUTPUT_DIR / "seed_watermarked"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = upload_url_to_path(record["download_url"])
        shutil.copy2(path, seed_dir / f"{record['name']}-{record['trace_id']}.png")


SYNTHETIC_FAMILIES = (
    "solid",
    "gradient",
    "correlated_noise",
    "grid",
    "ui_blocks",
    "periodic",
    "text_edges",
    "radial",
    "checker",
    "low_contrast",
)


def synthetic_family(index: int) -> str:
    return SYNTHETIC_FAMILIES[index % len(SYNTHETIC_FAMILIES)]


def synthetic_image(index: int, size: tuple[int, int] = (1280, 860)) -> Image.Image:
    width, height = size
    rng = np.random.default_rng(20260709 + index)
    kind = index % len(SYNTHETIC_FAMILIES)
    if kind == 0:
        color = tuple(int(x) for x in rng.integers(20, 235, size=3))
        return Image.new("RGB", size, color)
    if kind == 1:
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        xx = np.broadcast_to(x[None, :], (height, width))
        yy = np.broadcast_to(y, (height, width))
        arr = np.stack([
            40 + 160 * xx,
            50 + 120 * yy,
            180 - 90 * xx + 20 * yy,
        ], axis=2)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if kind == 2:
        arr = rng.normal(128, 28, (height, width, 3))
        arr = cv2.GaussianBlur(np.clip(arr, 0, 255).astype(np.uint8), (0, 0), sigmaX=1.1)
        return Image.fromarray(arr, "RGB")
    if kind == 3:
        arr = np.full((height, width, 3), 245, dtype=np.uint8)
        step = 80
        for y in range(0, height, step):
            arr[y : y + 2, :, :] = 210
        for x in range(0, width, step):
            arr[:, x : x + 2, :] = 210
        return Image.fromarray(arr, "RGB")
    if kind == 4:
        image = Image.new("RGB", size, (250, 250, 250))
        draw = ImageDraw.Draw(image)
        for _ in range(60):
            x1 = int(rng.integers(0, width))
            y1 = int(rng.integers(0, height))
            x2 = int(min(width, x1 + rng.integers(20, 180)))
            y2 = int(min(height, y1 + rng.integers(10, 90)))
            color = tuple(int(x) for x in rng.integers(80, 230, size=3))
            draw.rectangle((x1, y1, x2, y2), fill=color)
        return image
    yy, xx = np.mgrid[0:height, 0:width]
    if kind == 5:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = ((np.sin(xx / 28.0) + 1) * 90 + 40).astype(np.uint8)
        arr[:, :, 1] = ((np.cos(yy / 34.0) + 1) * 70 + 60).astype(np.uint8)
        arr[:, :, 2] = ((np.sin((xx + yy) / 45.0) + 1) * 60 + 80).astype(np.uint8)
        return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.35))
    if kind == 6:
        image = Image.new("RGB", size, (248, 248, 248))
        draw = ImageDraw.Draw(image)
        for row in range(18, height, 34):
            length = int(rng.integers(width // 6, width * 4 // 5))
            tone = int(rng.integers(55, 205))
            draw.rectangle((24, row, 24 + length, row + int(rng.integers(2, 5))), fill=(tone, tone, tone))
        return image
    if kind == 7:
        radius = np.sqrt(((xx - width / 2) / max(1, width)) ** 2 + ((yy - height / 2) / max(1, height)) ** 2)
        arr = np.stack([210 - 180 * radius, 80 + 130 * radius, 150 + 80 * np.cos(radius * 18)], axis=2)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if kind == 8:
        cell = int(rng.integers(18, 70))
        board = ((xx // cell + yy // cell) % 2).astype(np.uint8)
        colors = rng.integers(35, 225, size=(2, 3), dtype=np.uint8)
        return Image.fromarray(colors[board], "RGB")
    base = rng.integers(90, 170, size=3)
    arr = np.broadcast_to(base, (height, width, 3)).astype(np.float32).copy()
    arr += rng.normal(0, 2.5, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(radius=1.8)
    )


def create_negative_cases(sources: list[Path]) -> list[dict]:
    cases = []
    case_dir = OUTPUT_DIR / "negative_inputs"
    case_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        image = Image.open(source).convert("RGB")
        base_path = case_dir / f"{source.stem}_original.png"
        image.save(base_path)
        cases.append({"name": f"{source.stem}_original", "category": "original", "path": str(base_path)})
        for attack_name in NEGATIVE_ATTACKS:
            attack = ATTACK_BY_NAME[attack_name]
            attacked = attack(image)
            path = case_dir / f"{source.stem}_{attack_name}.png"
            attacked.save(path)
            cases.append({"name": f"{source.stem}_{attack_name}", "category": attack_name, "path": str(path)})

    for index in range(SYNTHETIC_VARIANTS):
        image = synthetic_image(index)
        path = case_dir / f"synthetic_{index:03d}.png"
        image.save(path)
        cases.append({
            "name": f"synthetic_{index:04d}",
            "category": f"synthetic_{synthetic_family(index)}",
            "path": str(path),
        })
    return cases


def init_worker(generated_trace_ids: list[str]) -> None:
    cv2.setNumThreads(1)
    main.app.state.generated_trace_ids = generated_trace_ids
    main.record_detection_result = lambda success: None


def detect_negative_case(case: dict) -> dict:
    image = Image.open(case["path"]).convert("RGB")
    started = time.perf_counter()
    try:
        detected = main.extract_watermark_from_image(image)
        success = True
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        detected = {"detail": getattr(exc, "detail", str(exc))}
        success = status_code != 404
    detection_ms = round((time.perf_counter() - started) * 1000, 3)
    artifact_path = ""
    if success:
        artifact = FALSE_POSITIVE_DIR / f"{case['name']}.png"
        image.save(artifact)
        artifact_path = str(artifact.relative_to(ROOT))
    return {
        "name": case["name"],
        "category": case["category"],
        "success": success,
        "detected_trace_id": detected.get("trace_id", "") if success else "",
        "confidence": detected.get("confidence", "") if success else "",
        "mode_label": detected.get("mode_label", "") if success else "",
        "mode": detected.get("mode", "") if success else "",
        "status": detected.get("status", "") if success else "",
        "code_recovery": detected.get("code_recovery", {}) if success else {},
        "layer_scores": detected.get("layer_scores", {}) if success else {},
        "match_inliers": detected.get("match_inliers", "") if success else "",
        "match_ratio": detected.get("match_ratio", "") if success else "",
        "detection_ms": detection_ms,
        "detail": detected.get("detail", "") if not success else "",
        "artifact_path": artifact_path,
    }


def summarize(results: list[dict]) -> dict:
    by_category: dict[str, dict] = {}
    false_positive = sum(1 for item in results if item["success"])
    for item in results:
        stats = by_category.setdefault(item["category"], {"total": 0, "false_positive": 0})
        stats["total"] += 1
        stats["false_positive"] += int(item["success"])
    for stats in by_category.values():
        stats["false_positive_rate"] = round(stats["false_positive"] / stats["total"], 4) if stats["total"] else 0
    return {
        "total": len(results),
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / len(results), 4) if results else 0,
        "source_false_positive": sum(
            1
            for item in results
            if item["success"] and not item["category"].startswith("synthetic_")
        ),
        "by_category": by_category,
    }


def negative_verdict(summary: dict) -> dict[str, object]:
    failed = []
    if int(summary.get("source_false_positive", 0)) > 0:
        failed.append("source_false_positive")
    total = int(summary.get("total", 0))
    false_positive = int(summary.get("false_positive", 0))
    raw_false_positive_rate = false_positive / total if total else 0.0
    if raw_false_positive_rate >= 0.001:
        failed.append("false_positive_rate")
    return {"verdict": "FAIL" if failed else "PASS", "failed_gates": failed}


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
        "metadata": build_report_metadata("negative", seed, algorithm_version),
        "summary": summary,
        "cases": cases,
        "settings": settings,
        "verdict": verdict,
        "failed_gates": failed_gates,
    }


def write_outputs(payload: dict) -> None:
    errors = validate_report(payload)
    if errors:
        raise ValueError("invalid commercial report: " + ", ".join(errors))
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "name", "category", "success", "detected_trace_id", "confidence", "mode", "mode_label",
        "status", "code_recovery", "layer_scores", "match_inliers", "match_ratio", "detection_ms",
        "detail", "artifact_path",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in payload["cases"]:
            writer.writerow({field: item.get(field, "") for field in fields})

    summary = payload["summary"]
    lines = [
        "# 商用未加水印误报压力测试报告",
        "",
        f"- 测试时间：{payload['created_at']}",
        f"- 候选水印 trace：{len(payload['seed_records'])} 个",
        f"- 负样本总数：{summary['total']}",
        f"- 误报数：{summary['false_positive']}",
        f"- 误报率：{summary['false_positive_rate']:.2%}",
        f"- 原图及攻击变体误报：{summary['source_false_positive']}",
        f"- 并行进程数：{payload['workers']}",
        f"- 商用门禁：{payload['verdict']}",
        f"- 未通过门槛：{', '.join(payload['failed_gates']) or '无'}",
        "",
        "## 按类别",
        "",
        "| 类别 | 用例数 | 误报数 | 误报率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, stats in sorted(summary["by_category"].items()):
        lines.append(f"| {category} | {stats['total']} | {stats['false_positive']} | {stats['false_positive_rate']:.2%} |")
    false_positives = [item for item in payload["cases"] if item["success"]]
    lines.extend(["", "## 误报明细", ""])
    if false_positives:
        lines.extend(["| 名称 | 类别 | trace | 状态 | 置信度 |", "| --- | --- | --- | --- | ---: |"])
        for item in false_positives:
            lines.append(
                f"| {item['name']} | {item['category']} | {item['detected_trace_id']} | {item['status']} | {item['confidence']} |"
            )
    else:
        lines.append("本轮未发现误报。")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_run() -> int:
    benchmark_started = time.perf_counter()
    reset_state()
    sources = sorted((ROOT / "img").glob("*.png"))
    client = TestClient(main.app)
    seed_records = seed_watermark_records(client, sources)
    save_seed_records(seed_records)
    cases = create_negative_cases(sources)
    workers = max(1, min(BENCHMARK_WORKERS, len(cases)))
    generated_trace_ids = list(getattr(main.app.state, "generated_trace_ids", []))
    if workers == 1:
        init_worker(generated_trace_ids)
        results = [detect_negative_case(case) for case in cases]
    else:
        ordered: list[dict | None] = [None] * len(cases)
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(generated_trace_ids,)) as executor:
            future_to_index = {executor.submit(detect_negative_case, case): index for index, case in enumerate(cases)}
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
        results = [item for item in ordered if item is not None]

    summary = summarize(results)
    gate = negative_verdict(summary)
    payload = build_report(
        summary,
        results,
        seed=RANDOM_SEED,
        algorithm_version=ROBUST_WATERMARK_VERSION,
        settings={
            "synthetic_variants": SYNTHETIC_VARIANTS,
            "negative_attacks": NEGATIVE_ATTACKS,
            "fidelity_level": FIDELITY_LEVEL,
            "small_crop_trace_strength": SMALL_CROP_TRACE_STRENGTH,
            "small_crop_trace_density": SMALL_CROP_TRACE_DENSITY,
            "robust_watermark_strength": ROBUST_WATERMARK_STRENGTH,
            "robust_watermark_version": ROBUST_WATERMARK_VERSION,
        },
        verdict=gate["verdict"],
        failed_gates=gate["failed_gates"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        workers=workers,
        seed_records=[{"name": item["name"], "trace_id": item["trace_id"]} for item in seed_records],
        elapsed_seconds=round(time.perf_counter() - benchmark_started, 3),
    )
    write_outputs(payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")
    print(f"json={JSON_PATH}")
    print(f"csv={CSV_PATH}")
    return 0 if gate["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main_run())
