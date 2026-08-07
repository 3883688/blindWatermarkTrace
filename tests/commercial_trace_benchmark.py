import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/uploads"
os.environ["DATA_DIR"] = "test_output/data"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image
import cv2

import main
from tests.commercial_benchmark_config import build_embedding_form
from tests.commercial_report_contract import build_report_metadata, validate_report


def parse_float_list(name: str, default: str) -> list[float]:
    return [float(item.strip()) for item in os.getenv(name, default).split(",") if item.strip()]


RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260707"))
SCALE_FACTORS = parse_float_list("SCALE_FACTORS", "0.5,0.75,1.0,1.25,1.5,2.0")
CROP_RATIOS = parse_float_list("CROP_RATIOS", "0.3,0.5,0.8,1.0")
NEGATIVE_SCALE_FACTORS = parse_float_list("NEGATIVE_SCALE_FACTORS", ",".join(map(str, SCALE_FACTORS)))
NEGATIVE_CROP_RATIOS = parse_float_list("NEGATIVE_CROP_RATIOS", ",".join(map(str, CROP_RATIOS)))
CROPS_PER_RATIO = int(os.getenv("CROPS_PER_RATIO", "3"))
FIDELITY_LEVEL = os.getenv("FIDELITY_LEVEL", "0.75")
SMALL_CROP_TRACE_STRENGTH = os.getenv("SMALL_CROP_TRACE_STRENGTH", "0.35")
SMALL_CROP_TRACE_DENSITY = os.getenv("SMALL_CROP_TRACE_DENSITY", "medium")
ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
BENCHMARK_WORKERS = int(os.getenv("BENCHMARK_WORKERS", str(min(20, max(1, (os.cpu_count() or 2) - 2)))))

OUTPUT_DIR = ROOT / "test_output" / "commercial_trace_benchmark"
WATERMARKED_DIR = OUTPUT_DIR / "watermarked"
ATTACKED_DIR = OUTPUT_DIR / "attacked"
FALSE_POSITIVE_DIR = OUTPUT_DIR / "false_positive"
REPORT_PATH = OUTPUT_DIR / "commercial_trace_test_report.md"
JSON_PATH = OUTPUT_DIR / "commercial_trace_results.json"
CSV_PATH = OUTPUT_DIR / "commercial_trace_results.csv"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def post_extract(client: TestClient, image: Image.Image, filename: str) -> tuple[bool, dict, float]:
    started = time.perf_counter()
    response = client.post(
        "/api/watermark/extract",
        files={"file": (filename, image_to_png_bytes(image), "image/png")},
    )
    detection_ms = round((time.perf_counter() - started) * 1000, 3)
    if response.status_code == 200:
        return True, response.json(), detection_ms
    try:
        return False, response.json(), detection_ms
    except json.JSONDecodeError:
        return False, {"detail": response.text}, detection_ms


def upload_url_to_path(url: str) -> Path:
    if not url.startswith("/uploads/"):
        raise ValueError(f"unexpected upload url: {url}")
    return main.UPLOAD_DIR / url.replace("/uploads/", "")


def crop_random_region(image: Image.Image, ratio: float, rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * ratio)))
    crop_height = max(1, int(round(height * ratio)))
    left = rng.randint(0, max(0, width - crop_width))
    top = rng.randint(0, max(0, height - crop_height))
    return image.crop((left, top, left + crop_width, top + crop_height))


def case_rng(source: str, scale: float, crop_ratio: float, crop_index: int) -> random.Random:
    key = f"{RANDOM_SEED}|{source}|{scale}|{crop_ratio}|{crop_index}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def init_worker(generated_trace_ids: list[str]) -> None:
    cv2.setNumThreads(1)
    main.app.state.generated_trace_ids = generated_trace_ids
    main.record_detection_result = lambda success: None


def detection_diagnostics(detected: dict) -> dict[str, object]:
    recovery = detected.get("code_recovery") or {}
    phase_counts = recovery.get("phase_tile_counts") or []
    return {
        "bit_errors": recovery.get("bit_errors", ""),
        "corrected_symbols": recovery.get("corrected_symbols", ""),
        "erasure_count": recovery.get("erasure_count", ""),
        "recovery_method": recovery.get("recovery_method", ""),
        "phase_tile_counts": "/".join(str(value) for value in phase_counts),
        "authenticated_tiles": recovery.get("authenticated_tiles", ""),
        "mean_signed_agreement": recovery.get("mean_signed_agreement", ""),
    }


def run_crop_case(case: dict) -> dict:
    image = Image.open(case["image_path"]).convert("RGB")
    scaled_size = (
        max(1, int(round(image.width * case["scale_factor"]))),
        max(1, int(round(image.height * case["scale_factor"]))),
    )
    scaled = image.resize(scaled_size, Image.Resampling.BICUBIC)
    attacked = crop_random_region(
        scaled,
        case["crop_ratio"],
        case_rng(case["source"], case["scale_factor"], case["crop_ratio"], case["crop_index"]),
    )
    started = time.perf_counter()
    try:
        detected = main.extract_watermark_from_image(attacked)
        success = True
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
        detected = {"detail": getattr(exc, "detail", str(exc))}
        success = False
    detection_ms = round((time.perf_counter() - started) * 1000, 3)
    detected_trace = detected.get("trace_id", "") if success else ""
    expected_trace = case.get("trace_id", "")
    correct_trace = success and detected_trace == expected_trace if expected_trace else not success
    artifact_path = ""
    if not correct_trace:
        artifact_dir = ATTACKED_DIR if expected_trace else FALSE_POSITIVE_DIR
        artifact = artifact_dir / case["attacked_name"]
        attacked.save(artifact)
        artifact_path = str(artifact.relative_to(ROOT))
    return {
        "source": case["source"],
        "case_type": case["case_type"],
        "trace_id": expected_trace,
        "scale_factor": case["scale_factor"],
        "crop_ratio": case["crop_ratio"],
        "crop_index": case["crop_index"],
        "attacked_size": f"{attacked.width}x{attacked.height}",
        "success": success,
        "correct_trace": correct_trace,
        "detected_trace_id": detected_trace,
        "confidence": detected.get("confidence", "") if success else "",
        "mode_label": detected.get("mode_label", "") if success else "",
        "status": detected.get("status", "") if success else "",
        "detail": detected.get("detail", "") if not success else "",
        "attacked_path": artifact_path,
        "detection_ms": detection_ms,
        **detection_diagnostics(detected if success else {}),
    }


def crop_verdict(summary: dict) -> dict[str, object]:
    failed = []
    if summary.get("wrong", 0):
        failed.append("wrong_trace")
    if summary.get("false_positive", 0):
        failed.append("false_positive")
    if float(summary.get("recall", 0.0)) < 0.95:
        failed.append("overall_recall")
    for ratio, stats in sorted(summary.get("by_crop_ratio", {}).items(), key=lambda item: float(item[0])):
        minimum = 0.80 if float(ratio) <= 0.3 else 0.95
        if float(stats.get("recall", 0.0)) < minimum:
            failed.append(f"crop_{ratio}_recall")
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
        "metadata": build_report_metadata("trace", seed, algorithm_version),
        "summary": summary,
        "cases": cases,
        "settings": settings,
        "verdict": verdict,
        "failed_gates": failed_gates,
    }


def build_summary(results: list[dict]) -> dict:
    positive_results = [item for item in results if item["case_type"] == "watermarked"]
    negative_results = [item for item in results if item["case_type"] == "unwatermarked"]
    total = len(positive_results)
    success = sum(1 for item in positive_results if item["success"])
    correct = sum(1 for item in positive_results if item["correct_trace"])
    wrong = sum(1 for item in positive_results if item["success"] and not item["correct_trace"])
    negative_total = len(negative_results)
    false_positive = sum(1 for item in negative_results if item["success"])
    by_scale: dict[str, dict] = {}
    by_crop_ratio: dict[str, dict] = {}
    for item in positive_results:
        for bucket, key in ((by_scale, str(item["scale_factor"])), (by_crop_ratio, str(item["crop_ratio"]))):
            stats = bucket.setdefault(key, {"total": 0, "success": 0, "correct": 0, "wrong": 0})
            stats["total"] += 1
            stats["success"] += int(item["success"])
            stats["correct"] += int(item["correct_trace"])
            stats["wrong"] += int(item["success"] and not item["correct_trace"])
    for bucket in (by_scale, by_crop_ratio):
        for stats in bucket.values():
            stats["recall"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0
            stats["wrong_rate"] = round(stats["wrong"] / stats["total"], 4) if stats["total"] else 0
    return {
        "total": total,
        "success": success,
        "correct": correct,
        "wrong": wrong,
        "recall": round(correct / total, 4) if total else 0,
        "wrong_rate": round(wrong / total, 4) if total else 0,
        "by_scale": by_scale,
        "by_crop_ratio": by_crop_ratio,
        "negative_total": negative_total,
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / negative_total, 4) if negative_total else 0,
    }


def write_csv(results: list[dict]) -> None:
    fields = [
        "source",
        "case_type",
        "trace_id",
        "scale_factor",
        "crop_ratio",
        "crop_index",
        "attacked_size",
        "success",
        "correct_trace",
        "detected_trace_id",
        "confidence",
        "mode_label",
        "status",
        "detail",
        "attacked_path",
        "detection_ms",
        "bit_errors",
        "corrected_symbols",
        "erasure_count",
        "recovery_method",
        "phase_tile_counts",
        "authenticated_tiles",
        "mean_signed_agreement",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item.get(field, "") for field in fields})


def table_for_bucket(bucket: dict[str, dict], key_label: str) -> str:
    lines = [f"| {key_label} | 用例数 | 正确追溯 | 错误命中 | 召回率 | 错误命中率 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in sorted(bucket, key=lambda value: float(value)):
        stats = bucket[key]
        lines.append(
            f"| {key} | {stats['total']} | {stats['correct']} | {stats['wrong']} | "
            f"{stats['recall']:.2%} | {stats['wrong_rate']:.2%} |"
        )
    return "\n".join(lines)


def write_report(payload: dict) -> None:
    summary = payload["summary"]
    failed = [item for item in payload["cases"] if item["case_type"] == "watermarked" and not item["correct_trace"]]
    false_positives = [item for item in payload["cases"] if item["case_type"] == "unwatermarked" and item["success"]]
    lines = [
        "# 商用水印缩放裁剪追溯测试报告",
        "",
        f"- 测试时间：{payload['created_at']}",
        f"- 随机种子：{payload['metadata']['seed']}",
        f"- 样本图片：{len(payload['sources'])} 张，来自 `img/`",
        f"- 生成水印配置：fidelity_level={FIDELITY_LEVEL}, small_crop_trace_enabled=true, "
        f"small_crop_trace_strength={SMALL_CROP_TRACE_STRENGTH}, small_crop_trace_density={SMALL_CROP_TRACE_DENSITY}, "
        f"robust_watermark_strength={ROBUST_WATERMARK_STRENGTH}, "
        f"robust_watermark_version={ROBUST_WATERMARK_VERSION}",
        f"- 攻击组合：缩放 {SCALE_FACTORS}；随机裁剪比例 {CROP_RATIOS}；每个比例 {CROPS_PER_RATIO} 次",
        f"- 误报抽样：未加水印原图缩放 {NEGATIVE_SCALE_FACTORS}；随机裁剪比例 {NEGATIVE_CROP_RATIOS}；任何识别结果均计为误报",
        "",
        "## 总览",
        "",
        f"- 测试用例数：{summary['total']}",
        f"- 正确追溯：{summary['correct']}",
        f"- 错误命中：{summary['wrong']}",
        f"- 未检出：{summary['total'] - summary['success']}",
        f"- 正确召回率：{summary['recall']:.2%}",
        f"- 错误命中率：{summary['wrong_rate']:.2%}",
        f"- 未加水印测试用例数：{summary['negative_total']}",
        f"- 未加水印误报：{summary['false_positive']}",
        f"- 未加水印误报率：{summary['false_positive_rate']:.2%}",
        f"- 商用门禁：{payload['verdict']}",
        f"- 未通过门槛：{', '.join(payload['failed_gates']) or '无'}",
        f"- 总耗时：{payload['elapsed_seconds']:.2f} 秒",
        "",
        "## 按缩放比例",
        "",
        table_for_bucket(summary["by_scale"], "缩放比例"),
        "",
        "## 按裁剪比例",
        "",
        table_for_bucket(summary["by_crop_ratio"], "裁剪比例"),
        "",
        "## 失败明细",
        "",
    ]
    if failed:
        lines.extend([
            "| 原图 | 缩放 | 裁剪 | 次数 | 期望 trace | 实际 trace | 状态 | 置信度 |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
        ])
        for item in failed[:100]:
            lines.append(
                f"| {item['source']} | {item['scale_factor']} | {item['crop_ratio']} | {item['crop_index']} | "
                f"{item['trace_id']} | {item['detected_trace_id']} | {item['status'] or item['detail']} | {item['confidence']} |"
            )
    else:
        lines.append("本轮没有失败用例。")
    lines.extend([
        "",
        "## 未加水印误报明细",
        "",
    ])
    if false_positives:
        lines.extend([
            "| 原图 | 缩放 | 裁剪 | 次数 | 误报 trace | 状态 | 置信度 |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ])
        for item in false_positives[:100]:
            lines.append(
                f"| {item['source']} | {item['scale_factor']} | {item['crop_ratio']} | {item['crop_index']} | "
                f"{item['detected_trace_id']} | {item['status']} | {item['confidence']} |"
            )
    else:
        lines.append("本轮未加水印原图攻击样本没有误报。")
    lines.extend([
        "",
        "## 结论",
        "",
        "本报告覆盖“生成水印图后缩放并随机裁剪局部区域再追溯”和“未加水印原图同样攻击后的误报检查”。它不能替代完整商用验收；完整验收还需要 JPEG/截图链路/旋转/模糊/锐化/降噪/二次压缩/屏幕拍照，以及更大规模未加水印图片误报测试。",
        "",
        f"机器可读明细见 `{JSON_PATH.relative_to(ROOT)}` 和 `{CSV_PATH.relative_to(ROOT)}`。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_run() -> int:
    benchmark_started = time.perf_counter()
    for path in (main.UPLOAD_DIR, main.DATA_DIR):
        if path.exists() and ROOT in path.resolve().parents and path.name in {"uploads", "data"}:
            shutil.rmtree(path)
    main.app.state.generated_trace_ids = []
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)
    ATTACKED_DIR.mkdir(parents=True, exist_ok=True)
    FALSE_POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    main.ensure_dirs()
    client = TestClient(main.app)
    sources = sorted((ROOT / "img").glob("*.png"))
    generated = []
    cases = []

    for source in sources:
        with source.open("rb") as fp:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, fp, "image/png")},
                data=build_embedding_form("commercial-benchmark", FIDELITY_LEVEL),
            )
        response.raise_for_status()
        record = response.json()
        source_watermarked = upload_url_to_path(record["download_url"])
        local_watermarked = WATERMARKED_DIR / f"{source.stem}-{record['trace_id']}.png"
        shutil.copy2(source_watermarked, local_watermarked)
        generated.append({"source": source.name, "trace_id": record["trace_id"], "path": str(local_watermarked.relative_to(ROOT))})

        for scale_factor in SCALE_FACTORS:
            for crop_ratio in CROP_RATIOS:
                for crop_index in range(1, CROPS_PER_RATIO + 1):
                    attacked_name = f"{source.stem}_scale-{scale_factor}_crop-{crop_ratio}_{crop_index}.png"
                    cases.append({
                        "source": source.name,
                        "case_type": "watermarked",
                        "trace_id": record["trace_id"],
                        "scale_factor": scale_factor,
                        "crop_ratio": crop_ratio,
                        "crop_index": crop_index,
                        "image_path": str(local_watermarked),
                        "attacked_name": attacked_name,
                    })

        for scale_factor in NEGATIVE_SCALE_FACTORS:
            for crop_ratio in NEGATIVE_CROP_RATIOS:
                for crop_index in range(1, CROPS_PER_RATIO + 1):
                    attacked_name = f"negative_{source.stem}_scale-{scale_factor}_crop-{crop_ratio}_{crop_index}.png"
                    cases.append({
                        "source": source.name,
                        "case_type": "unwatermarked",
                        "trace_id": "",
                        "scale_factor": scale_factor,
                        "crop_ratio": crop_ratio,
                        "crop_index": crop_index,
                        "image_path": str(source),
                        "attacked_name": attacked_name,
                    })

    workers = max(1, min(BENCHMARK_WORKERS, len(cases)))
    generated_trace_ids = list(getattr(main.app.state, "generated_trace_ids", []))
    if workers == 1:
        init_worker(generated_trace_ids)
        results = [run_crop_case(case) for case in cases]
    else:
        ordered: list[dict | None] = [None] * len(cases)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(generated_trace_ids,),
        ) as executor:
            future_to_index = {
                executor.submit(run_crop_case, case): index
                for index, case in enumerate(cases)
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
        results = [item for item in ordered if item is not None]

    summary = build_summary(results)
    gate = crop_verdict(summary)
    payload = build_report(
        summary,
        results,
        seed=RANDOM_SEED,
        algorithm_version=ROBUST_WATERMARK_VERSION,
        settings={
            "fidelity_level": FIDELITY_LEVEL,
            "scale_factors": SCALE_FACTORS,
            "crop_ratios": CROP_RATIOS,
            "negative_scale_factors": NEGATIVE_SCALE_FACTORS,
            "negative_crop_ratios": NEGATIVE_CROP_RATIOS,
            "crops_per_ratio": CROPS_PER_RATIO,
            "workers": workers,
            "small_crop_trace_strength": SMALL_CROP_TRACE_STRENGTH,
            "small_crop_trace_density": SMALL_CROP_TRACE_DENSITY,
            "robust_watermark_strength": ROBUST_WATERMARK_STRENGTH,
            "robust_watermark_version": ROBUST_WATERMARK_VERSION,
        },
        verdict=gate["verdict"],
        failed_gates=gate["failed_gates"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sources=generated,
        elapsed_seconds=round(time.perf_counter() - benchmark_started, 3),
    )
    errors = validate_report(payload)
    if errors:
        raise ValueError("invalid commercial report: " + ", ".join(errors))
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results)
    write_report(payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")
    print(f"json={JSON_PATH}")
    print(f"csv={CSV_PATH}")
    return 0 if gate["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main_run())
