from __future__ import annotations

import importlib
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ComputeStatus:
    requested: str
    selected: str
    device_name: str | None
    degraded: bool
    fallback_reason: str | None
    operations: dict[str, str]


class AdaptiveComputeBackend:
    def __init__(
        self,
        requested: str | None = None,
        *,
        module_loader: Callable[[str], object] = importlib.import_module,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        mode = "auto" if requested is None else requested.strip().lower()
        if mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("TRACE_COMPUTE_DEVICE must be auto, cpu, or cuda")
        self._requested = mode
        self._module_loader = module_loader
        self._clock = clock
        self._cupy: object | None = None
        self._load_attempted = False
        self._device_name: str | None = None
        self._degraded = False
        self._fallback_reason: str | None = None
        self._operations: dict[str, str] = {}

    @staticmethod
    def _gpu_errors(cupy: object) -> tuple[type[BaseException], ...]:
        candidates: list[type[BaseException]] = [MemoryError]
        for owner, name in (
            (cupy.cuda.runtime, "CUDARuntimeError"),  # type: ignore[attr-defined]
            (cupy.cuda.memory, "OutOfMemoryError"),  # type: ignore[attr-defined]
        ):
            error_type = getattr(owner, name, None)
            if isinstance(error_type, type) and issubclass(error_type, BaseException):
                candidates.append(error_type)
        return tuple(dict.fromkeys(candidates))

    @staticmethod
    def _clean_device_name(raw_name: object) -> str:
        if isinstance(raw_name, bytes):
            value = raw_name.decode("utf-8", errors="replace")
        else:
            value = str(raw_name or "NVIDIA GPU")
        cleaned = "".join(character if character.isprintable() else "?" for character in value)
        return cleaned[:128] or "NVIDIA GPU"

    def _load_cupy(self) -> object | None:
        if self._requested == "cpu" or self._degraded:
            return None
        if self._load_attempted:
            return self._cupy
        self._load_attempted = True
        try:
            cupy = self._module_loader("cupy")
        except (ImportError, OSError) as exc:
            self._fallback_reason = type(exc).__name__
            LOGGER.warning("V4 compute using CPU: %s", self._fallback_reason)
            return None

        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:  # type: ignore[attr-defined]
                self._fallback_reason = "no-cuda-device"
                LOGGER.warning("V4 compute using CPU: %s", self._fallback_reason)
                return None
            properties = cupy.cuda.runtime.getDeviceProperties(0)  # type: ignore[attr-defined]
        except self._gpu_errors(cupy) as exc:
            self._fallback_reason = type(exc).__name__
            LOGGER.warning("V4 compute using CPU: %s", self._fallback_reason)
            return None

        raw_name = properties.get("name", properties.get(b"name"))
        self._device_name = self._clean_device_name(raw_name)
        self._cupy = cupy
        LOGGER.info("V4 compute device available: %s", self._device_name)
        return self._cupy

    def _degrade(self, operation: str, reason: str) -> None:
        self._degraded = True
        self._fallback_reason = reason
        self._operations = {name: "cpu" for name in self._operations}
        self._operations[operation] = "cpu"
        LOGGER.warning("V4 CUDA backend degraded to CPU: %s", reason)

    def _execute(
        self,
        name: str,
        *,
        eligible: bool,
        cpu_call: Callable[[], np.ndarray],
        gpu_call: Callable[[object], object],
    ) -> np.ndarray:
        if not eligible or self._requested == "cpu" or self._degraded:
            return np.asarray(cpu_call())

        cupy = self._load_cupy()
        if cupy is None:
            self._operations.setdefault(name, "cpu")
            return np.asarray(cpu_call())
        if self._operations.get(name) == "cpu":
            return np.asarray(cpu_call())
        if self._operations.get(name) == "cuda":
            try:
                result = gpu_call(cupy)
                cupy.cuda.runtime.deviceSynchronize()  # type: ignore[attr-defined]
                return np.asarray(cupy.asnumpy(result))  # type: ignore[attr-defined]
            except self._gpu_errors(cupy) as exc:
                self._degrade(name, type(exc).__name__)
                return np.asarray(cpu_call())

        cpu_start = self._clock()
        cpu_result = np.asarray(cpu_call())
        cpu_elapsed = self._clock() - cpu_start
        try:
            cupy.cuda.runtime.deviceSynchronize()  # type: ignore[attr-defined]
            gpu_start = self._clock()
            gpu_value = gpu_call(cupy)
            cupy.cuda.runtime.deviceSynchronize()  # type: ignore[attr-defined]
            gpu_elapsed = self._clock() - gpu_start
            gpu_result = np.asarray(cupy.asnumpy(gpu_value))  # type: ignore[attr-defined]
        except self._gpu_errors(cupy) as exc:
            self._degrade(name, type(exc).__name__)
            return cpu_result

        if (
            np.allclose(cpu_result, gpu_result, rtol=1e-9, atol=1e-9)
            and gpu_elapsed <= cpu_elapsed * 0.90
        ):
            self._operations[name] = "cuda"
            LOGGER.info("V4 compute operation selected CUDA: %s", name)
            return gpu_result

        self._operations[name] = "cpu"
        LOGGER.info("V4 compute operation selected CPU: %s", name)
        return cpu_result

    def forward_dct(self, blocks: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return self._execute(
            "forward_dct",
            eligible=blocks.shape[0] >= 256,
            cpu_call=lambda: basis @ blocks @ basis.T,
            gpu_call=lambda cp: (
                cp.asarray(basis) @ cp.asarray(blocks) @ cp.asarray(basis).T
            ),
        )

    def inverse_dct(self, blocks: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return self._execute(
            "inverse_dct",
            eligible=blocks.shape[0] >= 256,
            cpu_call=lambda: basis.T @ blocks @ basis,
            gpu_call=lambda cp: (
                cp.asarray(basis).T @ cp.asarray(blocks) @ cp.asarray(basis)
            ),
        )

    def fft2(
        self,
        values: np.ndarray,
        axes: tuple[int, int] = (-2, -1),
    ) -> np.ndarray:
        return self._execute(
            "fft2",
            eligible=values.size >= 65_536,
            cpu_call=lambda: np.fft.fft2(values, axes=axes),
            gpu_call=lambda cp: cp.fft.fft2(cp.asarray(values), axes=axes),
        )

    def status(self) -> ComputeStatus:
        selected = (
            "cuda"
            if not self._degraded and "cuda" in self._operations.values()
            else "cpu"
        )
        return ComputeStatus(
            requested=self._requested,
            selected=selected,
            device_name=self._device_name,
            degraded=self._degraded,
            fallback_reason=self._fallback_reason,
            operations=dict(self._operations),
        )


@lru_cache(maxsize=1)
def get_compute_backend() -> AdaptiveComputeBackend:
    return AdaptiveComputeBackend(os.getenv("TRACE_COMPUTE_DEVICE", "auto"))
