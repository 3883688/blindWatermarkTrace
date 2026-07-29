import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from trace_app.v4.domain import DetectionOutcome
from trace_app.v4.deadlines import DeadlineExceeded
from trace_app.v4.workers import IsolatedWorkerPool, WorkerLimits


def _return_value(value: int) -> int:
    return value * 2


def _sleep_for(seconds: float) -> None:
    time.sleep(seconds)


def _spawn_descendant(pid_file: str) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=(
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        ),
    )
    Path(pid_file).write_text(str(child.pid), encoding="ascii")
    time.sleep(30)


def _exceed_temp_quota() -> None:
    Path(tempfile.gettempdir(), "large.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    time.sleep(5)


def _raise_deadline() -> None:
    raise DeadlineExceeded("decode")


def _force_cleanup_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_isolated_worker_returns_typed_success() -> None:
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    result = pool.run(_return_value, 21, timeout_seconds=5)
    assert result.outcome is DetectionOutcome.SUCCESS
    assert result.value == 42


def test_timeout_terminates_child_and_never_maps_to_not_found() -> None:
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    result = pool.run(_sleep_for, 2.0, timeout_seconds=0.1)
    assert result.outcome is DetectionOutcome.TIMEOUT
    assert result.outcome is not DetectionOutcome.NOT_FOUND
    assert result.worker_pid is not None
    assert not pool.is_process_alive(result.worker_pid)


def test_concurrency_limit_maps_to_resource_exhausted() -> None:
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    assert pool._slots.acquire(blocking=False)
    try:
        result = pool.run(_return_value, 1, timeout_seconds=1)
    finally:
        pool._slots.release()
    assert result.outcome is DetectionOutcome.RESOURCE_EXHAUSTED


def test_worker_start_failure_returns_typed_service_error() -> None:
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    result = pool.run(lambda: None, timeout_seconds=1)
    assert result.outcome is DetectionOutcome.SERVICE_UNAVAILABLE
    assert result.error_type is not None


def test_timeout_terminates_the_worker_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    result = pool.run(_spawn_descendant, str(pid_file), timeout_seconds=0.5)
    assert result.outcome is DetectionOutcome.TIMEOUT
    child_pid = int(pid_file.read_text(encoding="ascii"))
    try:
        assert not pool.is_os_process_alive(child_pid)
    finally:
        _force_cleanup_pid(child_pid)


def test_temporary_directory_total_quota_maps_to_resource_exhausted() -> None:
    pool = IsolatedWorkerPool(
        WorkerLimits(concurrency=1, temp_bytes=1024 * 1024)
    )
    result = pool.run(_exceed_temp_quota, timeout_seconds=3)
    assert result.outcome is DetectionOutcome.RESOURCE_EXHAUSTED


def test_deadline_exception_from_worker_maps_to_timeout() -> None:
    pool = IsolatedWorkerPool(WorkerLimits(concurrency=1))
    result = pool.run(_raise_deadline, timeout_seconds=3)
    assert result.outcome is DetectionOutcome.TIMEOUT
