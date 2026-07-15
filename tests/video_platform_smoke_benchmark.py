import json
import os
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

os.environ["DB_ENABLED"] = "false"
os.environ["ROBUST_WATERMARK_VERSION"] = "3"
os.environ.setdefault("WATERMARK_AUTH_KEY", "video-platform-smoke-test-key-2026-07-12")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test_output" / "video_platform_smoke"
os.environ["UPLOAD_DIR"] = str(OUTPUT / "uploads")
os.environ["DATA_DIR"] = str(OUTPUT / "data")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image

import main


PROFILES = (
    ("landscape_1080_crf23", 1920, 1080, 23),
    ("landscape_720_crf28", 1280, 720, 28),
    ("portrait_720x1280_crf30", 720, 1280, 30),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def encode_and_extract_frame(
    ffmpeg: str, source: Path, profile: tuple[str, int, int, int], stem: str
) -> tuple[Path, Path]:
    name, width, height, crf = profile
    video_dir = OUTPUT / "videos"
    frame_dir = OUTPUT / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / f"{stem}-{name}.mp4"
    frame = frame_dir / f"{stem}-{name}.png"
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(source),
            "-t",
            "1",
            "-r",
            "10",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(video),
        ]
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "0.5",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(frame),
        ]
    )
    return video, frame


def embed_sources(client: TestClient) -> dict[str, dict]:
    records = {}
    for source in sorted((ROOT / "img").glob("[1-5].png")):
        with source.open("rb") as stream:
            response = client.post(
                "/api/watermark/embed",
                files={"file": (source.name, stream, "image/png")},
                data={
                    "user_id": "video-platform-smoke",
                    "mode": "dct",
                    "fidelity_level": "0.75",
                    "robust_watermark_version": "3",
                    "small_crop_trace_enabled": "false",
                    "dot_matrix_trace_enabled": "false",
                    "copyright_enabled": "false",
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"embed failed for {source.name}: {response.text}")
        records[source.name] = response.json()
    return records


def detect_frame(client: TestClient, frame: Path) -> tuple[int, dict, float]:
    started = time.perf_counter()
    with frame.open("rb") as stream:
        response = client.post(
            "/api/watermark/extract",
            files={"file": (frame.name, stream, "image/png")},
        )
    elapsed = time.perf_counter() - started
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    return response.status_code, payload, elapsed


def write_report(results: list[dict], elapsed: float) -> None:
    positives = [item for item in results if item["expected_watermarked"]]
    negatives = [item for item in results if not item["expected_watermarked"]]
    correct = sum(item["correct"] for item in positives)
    wrong = sum(item["wrong_trace"] for item in positives)
    false_positives = sum(item["detected"] for item in negatives)
    by_profile = {}
    for profile, *_ in PROFILES:
        cases = [item for item in positives if item["profile"] == profile]
        by_profile[profile] = {
            "correct": sum(item["correct"] for item in cases),
            "total": len(cases),
        }
    summary = {
        "robust_watermark_version": 3,
        "encoder": "FFmpeg libx264, yuv420p",
        "positive_correct": correct,
        "positive_total": len(positives),
        "positive_recall": correct / len(positives) if positives else 0,
        "wrong_trace": wrong,
        "negative_false_positives": false_positives,
        "negative_total": len(negatives),
        "elapsed_seconds": round(elapsed, 3),
        "profiles": by_profile,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# V3 video platform simulation smoke report",
        "",
        "## Summary",
        "",
        f"- Correct recall: {correct}/{len(positives)} ({summary['positive_recall']:.2%})",
        f"- Wrong trace: {wrong}",
        f"- Unwatermarked false positives: {false_positives}/{len(negatives)}",
        f"- Total elapsed: {elapsed:.1f}s",
        "",
        "## Profiles",
        "",
        "| Profile | Correct | Recall |",
        "|---|---:|---:|",
    ]
    for profile, counts in by_profile.items():
        recall = counts["correct"] / counts["total"] if counts["total"] else 0
        lines.append(f"| {profile} | {counts['correct']}/{counts['total']} | {recall:.2%} |")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Static one-second videos were encoded with real H.264/libx264 and yuv420p, then a decoded midpoint frame was submitted to the image trace API. This is a local platform-transcode approximation, not evidence of behavior after an actual Douyin or Kuaishou upload.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main_benchmark(ffmpeg: str) -> None:
    started = time.perf_counter()
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    main.app.state.generated_trace_ids = []
    main.ensure_dirs()
    client = TestClient(main.app)
    records = embed_sources(client)
    results = []
    for source_name, record in records.items():
        marked = main.UPLOAD_DIR / record["download_url"].replace("/uploads/", "")
        original = ROOT / "img" / source_name
        for expected_watermarked, source, kind in (
            (True, marked, "watermarked"),
            (False, original, "unwatermarked"),
        ):
            for profile in PROFILES:
                video, frame = encode_and_extract_frame(
                    ffmpeg, source, profile, f"{Path(source_name).stem}-{kind}"
                )
                status, payload, detection_seconds = detect_frame(client, frame)
                detected_trace = str(payload.get("trace_id") or "")
                expected_trace = record["trace_id"] if expected_watermarked else ""
                results.append(
                    {
                        "source": source_name,
                        "profile": profile[0],
                        "expected_watermarked": expected_watermarked,
                        "expected_trace_id": expected_trace,
                        "detected_trace_id": detected_trace,
                        "detected": status == 200,
                        "correct": status == 200 and detected_trace == expected_trace,
                        "wrong_trace": status == 200 and detected_trace != expected_trace,
                        "http_status": status,
                        "mode": payload.get("mode", ""),
                        "bit_errors": payload.get("bit_errors"),
                        "detection_seconds": round(detection_seconds, 3),
                        "video_bytes": video.stat().st_size,
                    }
                )
                print(
                    f"{source_name} {kind} {profile[0]}: "
                    f"HTTP {status}, trace={detected_trace or '-'}, {detection_seconds:.2f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    write_report(results, elapsed)
    print(f"Report: {OUTPUT / 'report.md'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: video_platform_smoke_benchmark.py <ffmpeg.exe>")
    main_benchmark(sys.argv[1])
