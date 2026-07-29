"""Isolated V4 worker execution with typed resource and timeout outcomes."""

from __future__ import annotations

import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from trace_app.v4.domain import DetectionOutcome


@dataclass(frozen=True, slots=True)
class WorkerLimits:
    concurrency: int
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    cpu_seconds: int = 300
    temp_bytes: int = 2 * 1024 * 1024 * 1024
    hard_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if min(
            self.concurrency,
            self.memory_bytes,
            self.cpu_seconds,
            self.temp_bytes,
            self.hard_timeout_seconds,
        ) <= 0:
            raise ValueError("worker limits must be positive")
        if self.hard_timeout_seconds > 1000:
            raise ValueError("worker timeout exceeds approved deep limit")


@dataclass(frozen=True, slots=True)
class WorkerResult:
    outcome: DetectionOutcome
    value: Any = None
    error_type: str | None = None
    worker_pid: int | None = None


def _apply_posix_limits(limits: WorkerLimits) -> None:
    if os.name != "posix":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.temp_bytes, limits.temp_bytes))
    os.setsid()


def _worker_entry(
    output: Any,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    limits: WorkerLimits,
    temp_dir: str,
) -> None:
    try:
        _apply_posix_limits(limits)
        os.environ["TMPDIR"] = temp_dir
        os.environ["TEMP"] = temp_dir
        os.environ["TMP"] = temp_dir
        value = function(*args)
        output.put(("success", value, None, os.getpid()))
    except (MemoryError, OSError) as error:
        output.put(("resource_exhausted", None, type(error).__name__, os.getpid()))
    except BaseException as error:
        output.put(("service_unavailable", None, type(error).__name__, os.getpid()))


class IsolatedWorkerPool:
    def __init__(self, limits: WorkerLimits) -> None:
        self.limits = limits
        self._slots = threading.BoundedSemaphore(limits.concurrency)
        self._active_pids: set[int] = set()
        self._pid_lock = threading.Lock()
        self._context = multiprocessing.get_context("spawn")

    def run(
        self,
        function: Callable[..., Any],
        *args: Any,
        timeout_seconds: float,
    ) -> WorkerResult:
        if timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")
        try:
            multiprocessing.reduction.ForkingPickler.dumps((function, args))
        except Exception as error:
            return WorkerResult(
                DetectionOutcome.SERVICE_UNAVAILABLE,
                error_type=type(error).__name__,
            )
        if not self._slots.acquire(blocking=False):
            return WorkerResult(DetectionOutcome.RESOURCE_EXHAUSTED)
        output = self._context.Queue(maxsize=1)
        temp_dir = tempfile.mkdtemp(prefix="trace-v4-worker-")
        process = self._context.Process(
            target=_worker_entry,
            args=(output, function, args, self.limits, temp_dir),
            daemon=False,
        )
        pid: int | None = None
        try:
            process.start()
            pid = process.pid
            if pid is not None:
                with self._pid_lock:
                    self._active_pids.add(pid)
            wait_seconds = min(float(timeout_seconds), self.limits.hard_timeout_seconds)
            process.join(wait_seconds)
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(5)
                return WorkerResult(DetectionOutcome.TIMEOUT, worker_pid=pid)
            try:
                status, value, error_type, reported_pid = output.get(timeout=1)
            except queue.Empty:
                return WorkerResult(
                    DetectionOutcome.RESOURCE_EXHAUSTED
                    if process.exitcode and process.exitcode < 0
                    else DetectionOutcome.SERVICE_UNAVAILABLE,
                    error_type="WorkerExitedWithoutResult",
                    worker_pid=pid,
                )
            return WorkerResult(
                DetectionOutcome(status), value, error_type, reported_pid
            )
        except Exception as error:
            return WorkerResult(
                DetectionOutcome.SERVICE_UNAVAILABLE,
                error_type=type(error).__name__,
                worker_pid=pid,
            )
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if pid is not None:
                with self._pid_lock:
                    self._active_pids.discard(pid)
            output.close()
            output.join_thread()
            shutil.rmtree(Path(temp_dir), ignore_errors=True)
            self._slots.release()

    def is_process_alive(self, pid: int) -> bool:
        with self._pid_lock:
            return pid in self._active_pids


__all__ = ("IsolatedWorkerPool", "WorkerLimits", "WorkerResult")
