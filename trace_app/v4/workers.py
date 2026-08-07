"""Isolated V4 worker execution with typed resource and timeout outcomes."""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from trace_app.v4.domain import DetectionOutcome
from trace_app.v4.deadlines import DeadlineExceeded


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
    start_event: Any,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    limits: WorkerLimits,
    temp_dir: str,
) -> None:
    try:
        if not start_event.wait(10):
            raise RuntimeError("worker isolation setup timed out")
        _apply_posix_limits(limits)
        os.environ["TMPDIR"] = temp_dir
        os.environ["TEMP"] = temp_dir
        os.environ["TMP"] = temp_dir
        value = function(*args)
        output.put(("success", value, None, os.getpid()))
    except DeadlineExceeded as error:
        output.put(("timeout", None, type(error).__name__, os.getpid()))
    except (MemoryError, OSError) as error:
        output.put(("resource_exhausted", None, type(error).__name__, os.getpid()))
    except BaseException as error:
        output.put(("service_unavailable", None, type(error).__name__, os.getpid()))


class _WindowsJob:
    def __init__(self, limits: WorkerLimits) -> None:
        self.handle: int | None = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMITS(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMITS),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED_LIMITS()
        info.BasicLimitInformation.PerJobUserTimeLimit = limits.cpu_seconds * 10_000_000
        info.BasicLimitInformation.LimitFlags = 0x00000004 | 0x00000100 | 0x00000200 | 0x00002000
        info.ProcessMemoryLimit = limits.memory_bytes
        info.JobMemoryLimit = limits.memory_bytes
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            raise error
        self.handle = int(handle)

    def assign(self, process: multiprocessing.Process) -> None:
        if self.handle is None:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        if not kernel32.AssignProcessToJobObject(self.handle, process.sentinel):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle is None:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle(self.handle)
        self.handle = None


def _directory_size(path: str) -> int:
    total = 0
    try:
        for entry in Path(path).rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (FileNotFoundError, OSError):
        pass
    return total


def _terminate_process_tree(process: multiprocessing.Process) -> None:
    pid = process.pid
    if pid is None:
        return
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
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process.join(5)
    if process.is_alive():
        if os.name == "posix":
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.join(5)


def _os_process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        start_event = self._context.Event()
        temp_dir = tempfile.mkdtemp(prefix="trace-v4-worker-")
        process = self._context.Process(
            target=_worker_entry,
            args=(output, start_event, function, args, self.limits, temp_dir),
            daemon=False,
        )
        pid: int | None = None
        windows_job: _WindowsJob | None = None
        try:
            process.start()
            pid = process.pid
            if pid is not None:
                with self._pid_lock:
                    self._active_pids.add(pid)
            windows_job = _WindowsJob(self.limits)
            windows_job.assign(process)
            start_event.set()
            wait_seconds = min(float(timeout_seconds), self.limits.hard_timeout_seconds)
            expires_at = time.monotonic() + wait_seconds
            resource_exhausted = False
            while process.is_alive() and time.monotonic() < expires_at:
                process.join(min(0.05, max(0.0, expires_at - time.monotonic())))
                if _directory_size(temp_dir) > self.limits.temp_bytes:
                    resource_exhausted = True
                    _terminate_process_tree(process)
                    break
            if resource_exhausted:
                return WorkerResult(
                    DetectionOutcome.RESOURCE_EXHAUSTED,
                    error_type="TemporaryDiskQuotaExceeded",
                    worker_pid=pid,
                )
            if process.is_alive():
                _terminate_process_tree(process)
                return WorkerResult(DetectionOutcome.TIMEOUT, worker_pid=pid)
            try:
                status, value, error_type, reported_pid = output.get(timeout=1)
            except queue.Empty:
                return WorkerResult(
                    DetectionOutcome.RESOURCE_EXHAUSTED
                    if process.exitcode and (process.exitcode < 0 or windows_job is not None)
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
                _terminate_process_tree(process)
            if pid is not None:
                with self._pid_lock:
                    self._active_pids.discard(pid)
            output.close()
            output.join_thread()
            if windows_job is not None:
                windows_job.close()
            shutil.rmtree(Path(temp_dir), ignore_errors=True)
            self._slots.release()

    def is_process_alive(self, pid: int) -> bool:
        with self._pid_lock:
            return pid in self._active_pids

    @staticmethod
    def is_os_process_alive(pid: int) -> bool:
        for _attempt in range(20):
            if not _os_process_alive(pid):
                return False
            time.sleep(0.01)
        return True


__all__ = ("IsolatedWorkerPool", "WorkerLimits", "WorkerResult")
