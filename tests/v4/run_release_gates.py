from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tests.v4.benchmark_manifest import evaluate_release_report, sign_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    evaluate_release_report(report)
    key = os.environ.get("V4_RELEASE_REPORT_KEY", "").encode()
    signed = sign_report(report, key)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(signed, sort_keys=True, indent=2), encoding="utf-8")
    print("release gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
