from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import watermark_v4.compute as compute_module
from watermark_v4 import dct as dct_module
from watermark_v4.compute import AdaptiveComputeBackend, get_compute_backend


class StepClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeRuntime:
    class CUDARuntimeError(RuntimeError):
        pass

    def __init__(self, *, device_count: int = 1) -> None:
        self.device_count = device_count

    def getDeviceCount(self) -> int:
        return self.device_count

    @staticmethod
    def deviceSynchronize() -> None:
        return None

    @staticmethod
    def getDeviceProperties(index: int) -> dict[str, bytes]:
        assert index == 0
        return {"name": b"Fake NVIDIA GPU"}


class FakeFFT:
    def __init__(
        self,
        *,
        invalid: bool = False,
        fail_on_call: int | None = None,
        error_type: type[BaseException] = FakeRuntime.CUDARuntimeError,
    ) -> None:
        self.invalid = invalid
        self.fail_on_call = fail_on_call
        self.error_type = error_type
        self.calls = 0

    def fft2(
        self,
        values: np.ndarray,
        axes: tuple[int, int] = (-2, -1),
    ) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise self.error_type("sensitive CUDA failure details")
        result = np.fft.fft2(values, axes=axes)
        return result + 1 if self.invalid else result


class FakeCupy:
    def __init__(
        self,
        *,
        invalid: bool = False,
        fail_on_call: int | None = None,
        error_type: type[BaseException] = FakeRuntime.CUDARuntimeError,
        device_count: int = 1,
    ) -> None:
        self.fft = FakeFFT(
            invalid=invalid,
            fail_on_call=fail_on_call,
            error_type=error_type,
        )
        self.cuda = SimpleNamespace(
            runtime=FakeRuntime(device_count=device_count),
            memory=SimpleNamespace(OutOfMemoryError=MemoryError),
        )

    asarray = staticmethod(np.asarray)
    asnumpy = staticmethod(np.asarray)


def test_windows_pip_cuda_packages_prepare_dll_search_paths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "cuda_runtime"
    nvrtc = tmp_path / "cuda_nvrtc"
    cufft = tmp_path / "cufft"
    cublas = tmp_path / "cublas"
    (runtime / "bin").mkdir(parents=True)
    (nvrtc / "bin").mkdir(parents=True)
    (cufft / "bin").mkdir(parents=True)
    (cublas / "bin").mkdir(parents=True)
    handles = []

    monkeypatch.setattr(
        compute_module,
        "_windows_cuda_package_roots",
        lambda: (
            ("cuda_runtime", runtime),
            ("cuda_nvrtc", nvrtc),
            ("cufft", cufft),
            ("cublas", cublas),
        ),
    )
    monkeypatch.setattr(compute_module, "_CUDA_ENV_PREPARED", False)
    monkeypatch.setattr(compute_module, "_CUDA_DLL_HANDLES", [])
    monkeypatch.setattr(
        compute_module.os,
        "add_dll_directory",
        lambda path: handles.append(path) or object(),
        raising=False,
    )
    monkeypatch.setenv("PATH", "C:\\Windows")
    monkeypatch.delenv("CUDA_PATH", raising=False)

    compute_module._prepare_windows_cuda_packages()

    assert os.environ["CUDA_PATH"] == str(nvrtc)
    assert os.environ["PATH"].split(os.pathsep)[:4] == [
        str(runtime / "bin"),
        str(nvrtc / "bin"),
        str(cufft / "bin"),
        str(cublas / "bin"),
    ]
    assert handles == [
        str(runtime / "bin"),
        str(nvrtc / "bin"),
        str(cufft / "bin"),
        str(cublas / "bin"),
    ]


def test_cpu_mode_never_loads_cupy() -> None:
    loaded: list[str] = []
    backend = AdaptiveComputeBackend(
        requested="cpu",
        module_loader=lambda name: loaded.append(name),
    )

    result = backend.fft2(np.eye(256))

    assert isinstance(result, np.ndarray)
    assert loaded == []
    assert backend.status().selected == "cpu"


def test_ineligible_operation_does_not_load_cupy() -> None:
    loaded: list[str] = []
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda name: loaded.append(name),
    )

    result = backend.fft2(np.eye(4))

    assert isinstance(result, np.ndarray)
    assert loaded == []
    assert backend.status().operations == {}


def test_absent_cupy_records_sanitized_cpu_fallback() -> None:
    def missing_cupy(_name: str) -> object:
        raise ImportError("C:\\private\\cupy.pyd contains a secret")

    backend = AdaptiveComputeBackend(requested="auto", module_loader=missing_cupy)

    backend.fft2(np.eye(256))
    status = backend.status()

    assert status.selected == "cpu"
    assert status.device_name is None
    assert status.fallback_reason == "ImportError"
    assert "private" not in status.fallback_reason
    assert status.operations == {"fft2": "cpu"}


def test_no_visible_device_records_sanitized_cpu_fallback() -> None:
    backend = AdaptiveComputeBackend(
        requested="cuda",
        module_loader=lambda _name: FakeCupy(device_count=0),
    )

    backend.fft2(np.eye(256))

    status = backend.status()
    assert status.selected == "cpu"
    assert status.device_name is None
    assert status.fallback_reason == "no-cuda-device"


def test_auto_uses_gpu_only_after_parity_and_speed_calibration() -> None:
    fake = FakeCupy()
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: fake,
        clock=StepClock([0.0, 1.0, 1.0, 1.5]),
    )

    result = backend.fft2(np.eye(256))

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.fft.fft2(np.eye(256)))
    status = backend.status()
    assert status.selected == "cuda"
    assert status.device_name == "Fake NVIDIA GPU"
    assert status.operations == {"fft2": "cuda"}


@pytest.mark.parametrize(
    ("fake", "times"),
    (
        (FakeCupy(), [0.0, 1.0, 1.0, 1.91]),
        (FakeCupy(invalid=True), [0.0, 1.0, 1.0, 1.5]),
    ),
)
def test_slow_or_invalid_gpu_candidate_stays_on_cpu(
    fake: FakeCupy,
    times: list[float],
) -> None:
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: fake,
        clock=StepClock(times),
    )

    result = backend.fft2(np.eye(256))

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.fft.fft2(np.eye(256)))
    assert backend.status().operations == {"fft2": "cpu"}


def test_forward_and_inverse_dct_gpu_results_match_numpy() -> None:
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: FakeCupy(),
        clock=StepClock([0.0, 1.0, 1.0, 1.5] * 2),
    )
    rng = np.random.default_rng(7)
    blocks = rng.normal(size=(256, 16, 16))
    basis = dct_module._dct_basis(16)

    transformed = backend.forward_dct(blocks, basis)
    restored = backend.inverse_dct(transformed, basis)

    assert isinstance(transformed, np.ndarray)
    assert isinstance(restored, np.ndarray)
    assert np.allclose(restored, blocks, rtol=1e-9, atol=1e-9)
    assert backend.status().operations == {
        "forward_dct": "cuda",
        "inverse_dct": "cuda",
    }


def test_cuda_runtime_failure_retries_cpu_and_permanently_degrades() -> None:
    fake = FakeCupy(fail_on_call=3)
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: fake,
        clock=StepClock([0.0, 1.0, 1.0, 1.5]),
    )
    values = np.eye(256)
    expected = np.fft.fft2(values)

    backend.fft2(values)
    failed_result = backend.fft2(values)
    later_result = backend.fft2(values)

    assert isinstance(failed_result, np.ndarray)
    assert np.allclose(failed_result, expected)
    assert np.allclose(later_result, expected)
    assert fake.fft.calls == 3
    status = backend.status()
    assert status.degraded
    assert status.selected == "cpu"
    assert status.fallback_reason == "CUDARuntimeError"
    assert status.operations == {"fft2": "cpu"}


def test_missing_cuda_library_retries_cpu_and_permanently_degrades() -> None:
    fake = FakeCupy(fail_on_call=1, error_type=ImportError)
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: fake,
        clock=StepClock([0.0, 1.0]),
    )

    result = backend.fft2(np.eye(256))

    assert np.allclose(result, np.fft.fft2(np.eye(256)))
    status = backend.status()
    assert status.degraded
    assert status.fallback_reason == "ImportError"
    assert status.operations == {"fft2": "cpu"}


def test_non_cuda_gpu_error_is_not_swallowed() -> None:
    backend = AdaptiveComputeBackend(
        requested="auto",
        module_loader=lambda _name: FakeCupy(
            fail_on_call=1,
            error_type=ValueError,
        ),
        clock=StepClock([0.0, 1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="sensitive CUDA failure details"):
        backend.fft2(np.eye(256))


@pytest.mark.parametrize("requested", ["gpu", "", "cuda:0"])
def test_invalid_mode_is_rejected(requested: str) -> None:
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        AdaptiveComputeBackend(requested=requested)


def test_get_compute_backend_is_cached_from_environment(monkeypatch) -> None:
    get_compute_backend.cache_clear()
    monkeypatch.setenv("TRACE_COMPUTE_DEVICE", "cpu")

    first = get_compute_backend()
    monkeypatch.setenv("TRACE_COMPUTE_DEVICE", "cuda")

    assert get_compute_backend() is first
    assert first.status().requested == "cpu"
    get_compute_backend.cache_clear()
