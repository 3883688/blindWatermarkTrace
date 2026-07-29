import time

from trace_app.v4.domain import DetectionOutcome
from trace_app.v4.workers import IsolatedWorkerPool, WorkerLimits


def _return_value(value: int) -> int:
    return value * 2


def _sleep_for(seconds: float) -> None:
    time.sleep(seconds)


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
