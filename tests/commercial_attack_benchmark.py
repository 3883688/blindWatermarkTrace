import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

os.environ["DB_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = "test_output/attack_benchmark/uploads"
os.environ["DATA_DIR"] = "test_output/attack_benchmark/data"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
import cv2
import numpy as np

import main
from tests.commercial_benchmark_config import build_embedding_form
from tests.commercial_report_contract import build_report_metadata, validate_report


TRACE_ROUNDS = int(os.getenv("TRACE_ROUNDS", "1"))
FIDELITY_LEVEL = os.getenv("FIDELITY_LEVEL", "0.75")
SMALL_CROP_TRACE_STRENGTH = os.getenv("SMALL_CROP_TRACE_STRENGTH", "0.35")
SMALL_CROP_TRACE_DENSITY = os.getenv("SMALL_CROP_TRACE_DENSITY", "medium")
ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260707"))
BENCHMARK_LABEL = os.getenv(
    "BENCHMARK_LABEL",
    "commercial_stability_benchmark" if TRACE_ROUNDS > 1 else "commercial_attack_benchmark",
)
OUTPUT_DIR = ROOT / "test_output" / BENCHMARK_LABEL
FAILED_DIR = OUTPUT_DIR / "failed"
FALSE_POSITIVE_DIR = OUTPUT_DIR / "false_positive"
REPORT_PATH = OUTPUT_DIR / "commercial_attack_test_report.md"
JSON_PATH = OUTPUT_DIR / "commercial_attack_results.json"
CSV_PATH = OUTPUT_DIR / "commercial_attack_results.csv"


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def double_jpeg(image: Image.Image, first_quality: int, second_quality: int) -> Image.Image:
    return jpeg_roundtrip(jpeg_roundtrip(image, first_quality), second_quality)


def rotate_small_angle(image: Image.Image, angle: float) -> Image.Image:
    return image.convert("RGB").rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def attack_jpeg_q90(image: Image.Image) -> Image.Image:
    return jpeg_roundtrip(image, 90)


def attack_jpeg_q70(image: Image.Image) -> Image.Image:
    return jpeg_roundtrip(image, 70)


def attack_jpeg_q50(image: Image.Image) -> Image.Image:
    return jpeg_roundtrip(image, 50)


def attack_jpeg_q30(image: Image.Image) -> Image.Image:
    return jpeg_roundtrip(image, 30)


def attack_double_jpeg_70_50(image: Image.Image) -> Image.Image:
    return double_jpeg(image, 70, 50)


def attack_rotate_3deg(image: Image.Image) -> Image.Image:
    return rotate_small_angle(image, 3.0)


def attack_rotate_1deg(image: Image.Image) -> Image.Image:
    return rotate_small_angle(image, 1.0)


def attack_rotate_5deg(image: Image.Image) -> Image.Image:
    return rotate_small_angle(image, 5.0)


def attack_rotate_10deg(image: Image.Image) -> Image.Image:
    return rotate_small_angle(image, 10.0)


def attack_gaussian_blur_1_2(image: Image.Image) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.2))


def attack_unsharp_mask(image: Image.Image) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=3))


def attack_median_denoise(image: Image.Image) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.MedianFilter(size=3))


def attack_browser_screenshot_sim(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    downscaled = rgb.resize((max(1, int(width * 0.85)), max(1, int(height * 0.85))), Image.Resampling.BICUBIC)
    compressed = jpeg_roundtrip(downscaled, 82)
    return compressed.resize((width, height), Image.Resampling.BICUBIC)


def attack_wechat_screenshot_sim(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    max_side = 1440
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
    compressed = jpeg_roundtrip(rgb, 78)
    return compressed.filter(ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=2))


def attack_additive_noise(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.int16)
    rng = np.random.default_rng(20260709)
    noise = rng.normal(0, 3.5, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB")


def attack_screen_photo_sim(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    height, width = arr.shape[:2]
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dx = width * 0.035
    dy = height * 0.028
    dst = np.float32([[dx, dy * 1.5], [width - 1 - dx * 0.6, dy], [width - 1 - dx, height - 1 - dy], [dx * 0.7, height - 1 - dy * 0.6]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (width, height), borderValue=(245, 245, 245))
    yy, xx = np.mgrid[0:height, 0:width]
    vignette = 1.0 - 0.18 * (((xx - width / 2) / max(1, width / 2)) ** 2 + ((yy - height / 2) / max(1, height / 2)) ** 2)
    glare = np.exp(-(((xx - width * 0.72) / max(1, width * 0.18)) ** 2 + ((yy - height * 0.22) / max(1, height * 0.16)) ** 2)) * 18
    rng = np.random.default_rng(20260710)
    noise = rng.normal(0, 2.2, warped.shape)
    simulated = np.clip(warped.astype(np.float32) * vignette[:, :, None] + glare[:, :, None] + noise, 0, 255).astype(np.uint8)
    return jpeg_roundtrip(Image.fromarray(simulated, "RGB").filter(ImageFilter.GaussianBlur(radius=0.45)), 86)


ATTACKS: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
    ("jpeg_q90", attack_jpeg_q90),
    ("jpeg_q70", attack_jpeg_q70),
    ("jpeg_q50", attack_jpeg_q50),
    ("jpeg_q30", attack_jpeg_q30),
    ("double_jpeg_70_50", attack_double_jpeg_70_50),
    ("rotate_1deg", attack_rotate_1deg),
    ("rotate_3deg", attack_rotate_3deg),
    ("rotate_5deg", attack_rotate_5deg),
    ("rotate_10deg", attack_rotate_10deg),
    ("gaussian_blur_1_2", attack_gaussian_blur_1_2),
    ("unsharp_mask", attack_unsharp_mask),
    ("median_denoise", attack_median_denoise),
    ("browser_screenshot_sim", attack_browser_screenshot_sim),
    ("wechat_screenshot_sim", attack_wechat_screenshot_sim),
    ("additive_noise", attack_additive_noise),
    ("screen_photo_sim", attack_screen_photo_sim),
]

ATTACK_FILTER = {
    item.strip()
    for item in os.getenv("ATTACK_FILTER", "").split(",")
    if item.strip()
}
if ATTACK_FILTER:
    ATTACKS = [item for item in ATTACKS if item[0] in ATTACK_FILTER]

ATTACK_BY_NAME = dict(ATTACKS)
NEGATIVE_SOURCE_FILTER = {item.strip() for item in os.getenv("NEGATIVE_SOURCES", "").split(",") if item.strip()}
NEGATIVE_SOURCES = NEGATIVE_SOURCE_FILTER or {"1.png", "2.png", "3.png", "4.png", "5.png"}
BENCHMARK_WORKERS = int(os.getenv("BENCHMARK_WORKERS", str(min(12, max(1, (os.cpu_count() or 2) - 2)))))


def upload_url_to_path(url: str) -> Path:
    if not url.startswith("/uploads/"):
        raise ValueError(f"unexpected upload url: {url}")
    return main.UPLOAD_DIR / url.replace("/uploads/", "")


def post_extract(client: TestClient, image: Image.Image, filename: str) -> tuple[bool, dict]:
    response = client.post(
        "/api/watermark/extract",
        files={"file": (filename, png_bytes(image), "image/png")},
    )
    if response.status_code == 200:
        return True, response.json()
    try:
        return False, response.json()
    except json.JSONDecodeError:
        return False, {"detail": response.text}


def extract_direct(image: Image.Image) -> tuple[bool, dict]:
    try:
        return True, main.extract_watermark_from_image(image)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        detail = getattr(exc, "detail", str(exc))
        if status_code == 404:
            return False, {"detail": detail}
        return False, {"detail": detail or str(exc)}


def reset_state(clear_output: bool = True) -> None:
    paths = [main.UPLOAD_DIR, main.DATA_DIR]
    if clear_output:
        paths.append(OUTPUT_DIR)
    for path in paths:
        if path.exists() and ROOT in path.resolve().parents:
            shutil.rmtree(path)
    main.app.state.generated_trace_ids = []
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    FALSE_POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    main.ensure_dirs()


def seed_watermarks(client: TestClient, sources: list[Path]) -> dict[str, dict]:
    records = {}
    for source in sources:
        with source.open("rb") as fp:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, fp, "image/png")},
                data=build_embedding_form("commercial-attack-benchmark", FIDELITY_LEVEL),
            )
        response.raise_for_status()
        records[source.name] = response.json()
    return records


def init_worker(generated_trace_ids: list[str]) -> None:
    cv2.setNumThreads(1)
    main.app.state.generated_trace_ids = generated_trace_ids
    main.record_detection_result = lambda success: None


def run_attack_case(case: dict) -> dict:
    started = time.perf_counter()
    attack_name = case["attack"]
    attack = ATTACK_BY_NAME[attack_name]
    image = Image.open(case["image_path"]).convert("RGB")
    attacked = attack(image)
    success, detected = extract_direct(attacked)
    detected_trace = detected.get("trace_id", "") if success else ""
    correct = success and detected_trace == case.get("trace_id", "")
    expected_success = case["case_type"] == "watermarked"
    artifact_path = ""
    if (expected_success and not correct) or (not expected_success and success):
        artifact_dir = FAILED_DIR if expected_success else FALSE_POSITIVE_DIR
        artifact = artifact_dir / f"round-{case['round']}_{case['case_type']}_{case['source_stem']}_{attack_name}.png"
        attacked.save(artifact)
        artifact_path = str(artifact.relative_to(ROOT))
    return {
        "round": case["round"],
        "case_type": case["case_type"],
        "source": case["source"],
        "attack": attack_name,
        "trace_id": case.get("trace_id", ""),
        "success": success,
        "correct_trace": correct if expected_success else not success,
        "detected_trace_id": detected_trace,
        "confidence": detected.get("confidence", "") if success else "",
        "mode_label": detected.get("mode_label", "") if success else "",
        "status": detected.get("status", "") if success else "",
        "detail": detected.get("detail", "") if not success else "",
        "artifact_path": artifact_path,
        "detection_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def summarize(results: list[dict]) -> dict:
    positives = [item for item in results if item["case_type"] == "watermarked"]
    negatives = [item for item in results if item["case_type"] == "unwatermarked"]
    correct = sum(1 for item in positives if item["correct_trace"])
    wrong = sum(1 for item in positives if item["success"] and not item["correct_trace"])
    false_positive = sum(1 for item in negatives if item["success"])
    by_attack: dict[str, dict] = {}
    by_round: dict[str, dict] = {}
    for item in positives:
        stats = by_attack.setdefault(item["attack"], {"total": 0, "correct": 0, "wrong": 0})
        stats["total"] += 1
        stats["correct"] += int(item["correct_trace"])
        stats["wrong"] += int(item["success"] and not item["correct_trace"])
    for stats in by_attack.values():
        stats["recall"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0
        stats["wrong_rate"] = round(stats["wrong"] / stats["total"], 4) if stats["total"] else 0
    for item in results:
        key = str(item["round"])
        stats = by_round.setdefault(key, {
            "positive_total": 0,
            "correct": 0,
            "wrong": 0,
            "negative_total": 0,
            "false_positive": 0,
        })
        if item["case_type"] == "watermarked":
            stats["positive_total"] += 1
            stats["correct"] += int(item["correct_trace"])
            stats["wrong"] += int(item["success"] and not item["correct_trace"])
        else:
            stats["negative_total"] += 1
            stats["false_positive"] += int(item["success"])
    for stats in by_round.values():
        stats["recall"] = round(stats["correct"] / stats["positive_total"], 4) if stats["positive_total"] else 0
    return {
        "positive_total": len(positives),
        "positive_correct": correct,
        "positive_wrong": wrong,
        "wrong": wrong,
        "positive_missed": len(positives) - sum(1 for item in positives if item["success"]),
        "recall": round(correct / len(positives), 4) if positives else 0,
        "wrong_rate": round(wrong / len(positives), 4) if positives else 0,
        "negative_total": len(negatives),
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / len(negatives), 4) if negatives else 0,
        "by_attack": by_attack,
        "by_round": by_round,
    }


def attack_verdict(summary: dict) -> dict[str, object]:
    failed = []
    if int(summary.get("wrong", summary.get("positive_wrong", 0))) > 0:
        failed.append("wrong_trace")
    if int(summary.get("false_positive", 0)) > 0:
        failed.append("false_positive")
    for round_key, stats in sorted(summary.get("by_round", {}).items(), key=lambda item: int(item[0])):
        if float(stats.get("recall", 0.0)) < 0.95:
            failed.append(f"round_{round_key}_recall")
        if int(stats.get("wrong", 0)) > 0:
            failed.append(f"round_{round_key}_wrong_trace")
        if int(stats.get("false_positive", 0)) > 0:
            failed.append(f"round_{round_key}_false_positive")
    return {"verdict": "FAIL" if failed else "PASS", "failed_gates": failed}


def build_report_settings(
    *,
    trace_rounds: int,
    workers: int,
    fidelity_level: str,
    robust_watermark_strength: str,
    robust_watermark_version: str,
    attack_filter: list[str],
    attacks: list[str],
    negative_sources: list[str],
) -> dict:
    return {
        "trace_rounds": trace_rounds,
        "workers": workers,
        "fidelity_level": fidelity_level,
        "robust_watermark_strength": robust_watermark_strength,
        "robust_watermark_version": robust_watermark_version,
        "attack_filter": list(attack_filter),
        "attacks": list(attacks),
        "negative_sources": sorted(negative_sources),
    }


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
        "metadata": build_report_metadata("attack", seed, algorithm_version),
        "summary": summary,
        "cases": cases,
        "settings": settings,
        "verdict": verdict,
        "failed_gates": failed_gates,
    }


def write_csv(results: list[dict]) -> None:
    fields = [
        "case_type",
        "source",
        "attack",
        "trace_id",
        "success",
        "correct_trace",
        "detected_trace_id",
        "confidence",
        "mode_label",
        "status",
        "detail",
        "artifact_path",
        "detection_ms",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item.get(field, "") for field in fields})


def write_report(payload: dict) -> None:
    summary = payload["summary"]
    unique_sources = sorted({item["source"] for item in payload["sources"]})
    lines = [
        "# 商用水印攻击矩阵测试报告",
        "",
        f"- 测试时间：{payload['created_at']}",
        f"- 样本图片：{len(unique_sources)} 张，来自 `img/`",
        f"- trace 记录：{len(payload['sources'])} 个",
        f"- 攻击类型：{', '.join(name for name, _ in ATTACKS)}",
        f"- 未加水印负样本：{', '.join(sorted(NEGATIVE_SOURCES))}",
        f"- 并行进程数：{payload['settings']['workers']}",
        f"- 保真度：{payload['settings']['fidelity_level']}",
        f"- 鲁棒层强度：{payload['settings']['robust_watermark_strength']}",
        f"- 鲁棒层版本：{payload['settings']['robust_watermark_version']}",
        f"- 商用门禁：{payload['verdict']}",
        f"- 未通过门槛：{', '.join(payload['failed_gates']) or '无'}",
        f"- 总耗时：{payload['elapsed_seconds']:.2f} 秒",
        f"- 检测吞吐：{payload['cases_per_second']:.2f} 用例/秒",
        "",
        "## 总览",
        "",
        f"- 随机 trace 轮数：{payload['settings']['trace_rounds']}",
        f"- 带水印攻击用例数：{summary['positive_total']}",
        f"- 正确追溯：{summary['positive_correct']}",
        f"- 错误命中：{summary['positive_wrong']}",
        f"- 未检出：{summary['positive_missed']}",
        f"- 正确召回率：{summary['recall']:.2%}",
        f"- 错误命中率：{summary['wrong_rate']:.2%}",
        f"- 未加水印测试用例数：{summary['negative_total']}",
        f"- 未加水印误报：{summary['false_positive']}",
        f"- 未加水印误报率：{summary['false_positive_rate']:.2%}",
        "",
        "## 按攻击类型",
        "",
        "| 攻击 | 用例数 | 正确追溯 | 错误命中 | 召回率 | 错误命中率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for attack, stats in summary["by_attack"].items():
        lines.append(
            f"| {attack} | {stats['total']} | {stats['correct']} | {stats['wrong']} | "
            f"{stats['recall']:.2%} | {stats['wrong_rate']:.2%} |"
        )
    lines.extend([
        "",
        "## 按 trace 轮次",
        "",
        "| 轮次 | 带水印用例 | 正确追溯 | 错误 trace | 召回率 | 负样本 | 误报 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for round_key, stats in summary["by_round"].items():
        lines.append(
            f"| {round_key} | {stats['positive_total']} | {stats['correct']} | {stats['wrong']} | "
            f"{stats['recall']:.2%} | {stats['negative_total']} | {stats['false_positive']} |"
        )
    failures = [item for item in payload["cases"] if item["case_type"] == "watermarked" and not item["correct_trace"]]
    false_positives = [item for item in payload["cases"] if item["case_type"] == "unwatermarked" and item["success"]]
    lines.extend(["", "## 失败明细", ""])
    if failures:
        lines.extend([
            "| 原图 | 攻击 | 期望 trace | 实际 trace | 状态 | 置信度 |",
            "| --- | --- | --- | --- | --- | ---: |",
        ])
        for item in failures:
            lines.append(
                f"| R{item['round']} {item['source']} | {item['attack']} | {item['trace_id']} | {item['detected_trace_id']} | "
                f"{item['status'] or item['detail']} | {item['confidence']} |"
            )
    else:
        lines.append("本轮带水印攻击样本没有失败用例。")
    lines.extend(["", "## 未加水印误报明细", ""])
    if false_positives:
        lines.extend([
            "| 原图 | 攻击 | 误报 trace | 状态 | 置信度 |",
            "| --- | --- | --- | --- | ---: |",
        ])
        for item in false_positives:
            lines.append(
                f"| R{item['round']} {item['source']} | {item['attack']} | {item['detected_trace_id']} | "
                f"{item['status']} | {item['confidence']} |"
            )
    else:
        lines.append("本轮未加水印攻击样本没有误报。")
    lines.extend([
        "",
        f"机器可读明细见 `{JSON_PATH.relative_to(ROOT)}` 和 `{CSV_PATH.relative_to(ROOT)}`。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_run() -> int:
    benchmark_started = time.perf_counter()
    reset_state(clear_output=True)
    sources = sorted((ROOT / "img").glob("*.png"))
    all_results = []
    all_sources = []
    for round_index in range(1, TRACE_ROUNDS + 1):
        reset_state(clear_output=False)
        client = TestClient(main.app)
        records = seed_watermarks(client, sources)
        all_sources.extend(
            {"round": round_index, "source": name, "trace_id": record["trace_id"]}
            for name, record in records.items()
        )
        cases = []
        for source in sources:
            record = records[source.name]
            for attack_name, attack in ATTACKS:
                cases.append({
                    "round": round_index,
                    "case_type": "watermarked",
                    "source": source.name,
                    "source_stem": source.stem,
                    "attack": attack_name,
                    "trace_id": record["trace_id"],
                    "image_path": str(upload_url_to_path(record["download_url"])),
                })

            if source.name in NEGATIVE_SOURCES:
                for attack_name, attack in ATTACKS:
                    cases.append({
                        "round": round_index,
                        "case_type": "unwatermarked",
                        "source": source.name,
                        "source_stem": source.stem,
                        "attack": attack_name,
                        "trace_id": "",
                        "image_path": str(source),
                    })
        workers = max(1, min(BENCHMARK_WORKERS, len(cases)))
        generated_trace_ids = list(getattr(main.app.state, "generated_trace_ids", []))
        if workers == 1:
            init_worker(generated_trace_ids)
            round_results = [run_attack_case(case) for case in cases]
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(generated_trace_ids,)) as executor:
                future_to_index = {executor.submit(run_attack_case, case): index for index, case in enumerate(cases)}
                ordered: list[dict | None] = [None] * len(cases)
                for future in as_completed(future_to_index):
                    ordered[future_to_index[future]] = future.result()
                round_results = [item for item in ordered if item is not None]
        all_results.extend(round_results)

    summary = summarize(all_results)
    gate = attack_verdict(summary)
    elapsed_seconds = time.perf_counter() - benchmark_started
    payload = build_report(
        summary,
        all_results,
        seed=RANDOM_SEED,
        algorithm_version=ROBUST_WATERMARK_VERSION,
        settings=build_report_settings(
            trace_rounds=TRACE_ROUNDS,
            workers=workers,
            fidelity_level=FIDELITY_LEVEL,
            robust_watermark_strength=ROBUST_WATERMARK_STRENGTH,
            robust_watermark_version=ROBUST_WATERMARK_VERSION,
            attack_filter=sorted(ATTACK_FILTER),
            attacks=[name for name, _ in ATTACKS],
            negative_sources=sorted(NEGATIVE_SOURCES),
        ),
        verdict=gate["verdict"],
        failed_gates=gate["failed_gates"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sources=all_sources,
        elapsed_seconds=round(elapsed_seconds, 3),
        cases_per_second=round(len(all_results) / elapsed_seconds, 3) if elapsed_seconds else 0.0,
    )
    errors = validate_report(payload)
    if errors:
        raise ValueError("invalid commercial report: " + ", ".join(errors))
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(all_results)
    write_report(payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")
    print(f"json={JSON_PATH}")
    print(f"csv={CSV_PATH}")
    return 0 if gate["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main_run())
