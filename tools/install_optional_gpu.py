import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GPU_SMOKE_SOURCE = """\
from watermark_v4.compute import _prepare_windows_cuda_packages

_prepare_windows_cuda_packages()
import cupy

assert cupy.cuda.runtime.getDeviceCount() > 0
values = cupy.asarray([1, 2, 3])
squared = values * values
matrix = cupy.eye(4)
product = matrix @ matrix
spectrum = cupy.fft.fft2(matrix)
cupy.cuda.get_current_stream().synchronize()
assert cupy.asnumpy(squared).tolist() == [1, 4, 9]
assert cupy.asnumpy(product).tolist() == cupy.asnumpy(matrix).tolist()
assert spectrum.shape == (4, 4)
"""


@dataclass(frozen=True, slots=True)
class InstallResult:
    installed: bool
    reason: str


def _has_valid_nvidia_devices(output: str) -> bool:
    try:
        rows = csv.reader(output.splitlines(), strict=True)
        parsed_rows = list(rows)
    except (AttributeError, csv.Error):
        return False

    found_device = False
    for row in parsed_rows:
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 2:
            return False
        name, driver_version = (value.strip() for value in row)
        if (
            not name
            or not name.isprintable()
            or name.casefold() in {"name", "n/a", "unknown"}
            or re.fullmatch(r"\d+(?:\.\d+)+", driver_version) is None
        ):
            return False
        found_device = True
    return found_device


def detect_nvidia(*, which=shutil.which, run=subprocess.run) -> InstallResult:
    executable = which("nvidia-smi")
    if executable is None:
        return InstallResult(False, "nvidia-smi-not-found")

    try:
        completed = run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "nvidia-smi-timeout")
    except (OSError, subprocess.SubprocessError):
        return InstallResult(False, "nvidia-smi-failed")

    if completed.returncode != 0:
        return InstallResult(False, "nvidia-smi-failed")

    if not _has_valid_nvidia_devices(completed.stdout):
        return InstallResult(False, "no-nvidia-device")
    return InstallResult(True, "nvidia-gpu-detected")


def install_optional_gpu(
    *,
    python_executable: str,
    requirements_path: Path,
    which=shutil.which,
    run=subprocess.run,
) -> InstallResult:
    detection = detect_nvidia(which=which, run=run)
    if not detection.installed:
        return detection

    if not requirements_path.is_file():
        return InstallResult(False, "gpu-requirements-missing")

    try:
        install = run(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_path),
            ],
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return InstallResult(False, "gpu-install-failed")
    if install.returncode != 0:
        return InstallResult(False, "gpu-install-failed")

    try:
        smoke = run(
            [python_executable, "-c", GPU_SMOKE_SOURCE],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return InstallResult(False, "gpu-smoke-test-failed")
    if smoke.returncode != 0:
        return InstallResult(False, "gpu-smoke-test-failed")
    return InstallResult(True, "gpu-ready")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("requirements-gpu.txt"),
    )
    args = parser.parse_args(argv)
    result = install_optional_gpu(
        python_executable=args.python,
        requirements_path=args.requirements,
    )
    print(f"GPU optional install: {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
