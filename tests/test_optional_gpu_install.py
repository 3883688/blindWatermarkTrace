import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tools.install_optional_gpu as installer
from tools.install_optional_gpu import (
    GPU_SMOKE_SOURCE,
    InstallResult,
    detect_nvidia,
    install_optional_gpu,
)


class ForbiddenRequirementsPath:
    def __getattribute__(self, name):
        raise AssertionError(f"CPU fallback accessed requirements path via {name}")


class NvidiaScenario:
    def __init__(
        self,
        outcome: str,
        *,
        pip_outcome: str = "valid",
        smoke_outcome: str = "valid",
    ) -> None:
        self.outcome = outcome
        self.pip_outcome = pip_outcome
        self.smoke_outcome = smoke_outcome
        self.commands: list[tuple[str, ...]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, command, **kwargs):
        command = tuple(str(part) for part in command)
        self.commands.append(command)
        self.kwargs.append(kwargs)
        if command[0].endswith("nvidia-smi"):
            if self.outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 5)
            if self.outcome == "exception":
                raise OSError("private host detail")
            if self.outcome == "failure":
                return subprocess.CompletedProcess(command, 1, "", "private error")
            stdout = {
                "malformed": "not a csv device row\n",
                "malformed-two-fields": "not a device, not-a-driver\n",
                "header": "name, driver_version\n",
                "invalid-driver": "NVIDIA GeForce RTX 4070, 566.x\n",
                "extra-field": "NVIDIA GeForce RTX 4070, 566.24, unexpected\n",
                "invalid-csv": '"NVIDIA GeForce RTX 4070, 566.24\n',
                "mixed-invalid": (
                    "NVIDIA GeForce RTX 4070, 566.24\n"
                    "name, driver_version\n"
                ),
                "missing-name": ", 566.24\n",
                "missing-driver": "NVIDIA GeForce RTX 4070, \n",
                "zero": "\n",
                "valid": "NVIDIA GeForce RTX 4070 Laptop GPU, 566.24\n",
            }[self.outcome]
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if "pip" in command:
            if self.pip_outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 30)
            if self.pip_outcome == "exception":
                raise OSError("private pip detail")
            return subprocess.CompletedProcess(
                command,
                0 if self.pip_outcome == "valid" else 1,
                "",
                "private pip detail",
            )
        if self.smoke_outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if self.smoke_outcome == "exception":
            raise OSError("private smoke detail")
        return subprocess.CompletedProcess(
            command,
            0 if self.smoke_outcome == "valid" else 1,
            "",
            "private smoke detail",
        )


def test_gpu_requirement_is_separate_and_pinned() -> None:
    requirements = Path(__file__).resolve().parents[1] / "requirements-gpu.txt"

    assert requirements.read_text(encoding="utf-8") == (
        "cupy-cuda12x==13.6.0\n"
        "nvidia-cuda-runtime-cu12==12.6.77\n"
        "nvidia-cuda-nvrtc-cu12==12.6.85\n"
    )


def test_missing_nvidia_smi_never_accesses_gpu_requirements_or_runs_commands() -> None:
    calls = []

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=ForbiddenRequirementsPath(),
        which=lambda _: None,
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result == InstallResult(False, "nvidia-smi-not-found")
    assert calls == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("failure", "nvidia-smi-failed"),
        ("timeout", "nvidia-smi-timeout"),
        ("exception", "nvidia-smi-failed"),
        ("malformed", "no-nvidia-device"),
        ("malformed-two-fields", "no-nvidia-device"),
        ("header", "no-nvidia-device"),
        ("invalid-driver", "no-nvidia-device"),
        ("extra-field", "no-nvidia-device"),
        ("invalid-csv", "no-nvidia-device"),
        ("mixed-invalid", "no-nvidia-device"),
        ("missing-name", "no-nvidia-device"),
        ("missing-driver", "no-nvidia-device"),
        ("zero", "no-nvidia-device"),
    ],
)
def test_unusable_nvidia_query_never_accesses_requirements_or_runs_pip(
    outcome: str,
    reason: str,
) -> None:
    runner = NvidiaScenario(outcome)

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=ForbiddenRequirementsPath(),
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(False, reason)
    assert len(runner.commands) == 1
    assert all("pip" not in command for command in runner.commands)


def test_nvidia_detection_uses_bounded_name_and_driver_query() -> None:
    runner = NvidiaScenario("valid")

    result = detect_nvidia(which=lambda _: "C:/bin/nvidia-smi", run=runner)

    assert result == InstallResult(True, "nvidia-gpu-detected")
    assert runner.commands == [
        (
            "C:/bin/nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        )
    ]
    assert runner.kwargs == [
        {
            "capture_output": True,
            "text": True,
            "timeout": 5,
            "check": False,
        }
    ]


def test_csv_parser_error_never_accesses_requirements_or_runs_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = NvidiaScenario("valid")

    def raise_csv_error(*args, **kwargs):
        raise csv.Error("private parser detail")

    monkeypatch.setattr(installer.csv, "reader", raise_csv_error)

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=ForbiddenRequirementsPath(),
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(False, "no-nvidia-device")
    assert len(runner.commands) == 1
    assert all("pip" not in command for command in runner.commands)


def test_visible_nvidia_gpu_installs_requirements_then_runs_bounded_smoke(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements-gpu.txt"
    requirements.write_text("cupy-cuda12x==13.6.0\n", encoding="utf-8")
    runner = NvidiaScenario("valid")

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=requirements,
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(True, "gpu-ready")
    assert runner.commands[1] == (
        "python",
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
    )
    assert runner.commands[2] == ("python", "-c", GPU_SMOKE_SOURCE)
    assert runner.kwargs[1] == {"capture_output": True, "check": False}
    assert runner.kwargs[2] == {
        "capture_output": True,
        "check": False,
        "timeout": 30,
    }


def test_smoke_source_requires_device_squares_values_and_synchronizes() -> None:
    assert "import cupy" in GPU_SMOKE_SOURCE
    assert "getDeviceCount()" in GPU_SMOKE_SOURCE
    assert "[1, 2, 3]" in GPU_SMOKE_SOURCE
    assert "values * values" in GPU_SMOKE_SOURCE
    assert "synchronize()" in GPU_SMOKE_SOURCE
    assert "[1, 4, 9]" in GPU_SMOKE_SOURCE


def test_missing_gpu_requirements_falls_back_without_running_pip(tmp_path: Path) -> None:
    runner = NvidiaScenario("valid")

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=tmp_path / "missing.txt",
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(False, "gpu-requirements-missing")
    assert len(runner.commands) == 1


@pytest.mark.parametrize("pip_outcome", ["failure", "timeout", "exception"])
def test_failed_gpu_install_returns_sanitized_fallback(
    tmp_path: Path,
    pip_outcome: str,
) -> None:
    requirements = tmp_path / "requirements-gpu.txt"
    requirements.write_text("cupy-cuda12x==13.6.0\n", encoding="utf-8")
    runner = NvidiaScenario("valid", pip_outcome=pip_outcome)

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=requirements,
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(False, "gpu-install-failed")
    assert "private" not in result.reason
    assert len(runner.commands) == 2


@pytest.mark.parametrize("smoke_outcome", ["failure", "timeout", "exception"])
def test_failed_gpu_smoke_returns_sanitized_fallback(
    tmp_path: Path,
    smoke_outcome: str,
) -> None:
    requirements = tmp_path / "requirements-gpu.txt"
    requirements.write_text("cupy-cuda12x==13.6.0\n", encoding="utf-8")
    runner = NvidiaScenario("valid", smoke_outcome=smoke_outcome)

    result = install_optional_gpu(
        python_executable="python",
        requirements_path=requirements,
        which=lambda _: "nvidia-smi",
        run=runner,
    )

    assert result == InstallResult(False, "gpu-smoke-test-failed")
    assert "private" not in result.reason


def test_cli_prints_only_sanitized_cpu_fallback_and_exits_zero(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PATH"] = ""

    completed = subprocess.run(
        [
            sys.executable,
            "tools/install_optional_gpu.py",
            "--python",
            "ignored-python",
            "--requirements",
            str(tmp_path / "must-not-be-accessed.txt"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == "GPU optional install: nvidia-smi-not-found\n"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "result",
    [
        InstallResult(True, "gpu-ready"),
        InstallResult(False, "gpu-install-failed"),
        InstallResult(False, "gpu-smoke-test-failed"),
    ],
)
def test_cli_prints_only_one_sanitized_line_for_nvidia_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result: InstallResult,
) -> None:
    monkeypatch.setattr(installer, "install_optional_gpu", lambda **kwargs: result)

    exit_code = installer.main(
        ["--python", "python", "--requirements", "requirements-gpu.txt"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"GPU optional install: {result.reason}\n"
    assert captured.err == ""
