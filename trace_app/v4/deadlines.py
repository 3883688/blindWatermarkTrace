"""Monotonic, non-extendable deadlines shared across V4 stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


Clock = Callable[[], float]


class DeadlineExceeded(TimeoutError):
    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"V4 deadline exceeded during {stage}")


@dataclass(frozen=True, slots=True)
class Deadline:
    expires_at: float
    _clock: Clock = field(default=time.monotonic, repr=False, compare=False)

    @classmethod
    def after(
        cls,
        seconds: float,
        *,
        clock: Clock = time.monotonic,
        maximum_seconds: float | None = None,
    ) -> "Deadline":
        duration = float(seconds)
        if duration <= 0:
            raise ValueError("deadline duration must be positive")
        if maximum_seconds is not None and duration > maximum_seconds:
            raise ValueError("deadline exceeds approved maximum")
        return cls(clock() + duration, clock)

    @classmethod
    def synchronous(cls, *, clock: Clock = time.monotonic) -> "Deadline":
        return cls.after(300, clock=clock, maximum_seconds=300)

    @classmethod
    def deep(cls, *, clock: Clock = time.monotonic) -> "Deadline":
        return cls.after(1000, clock=clock, maximum_seconds=1000)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - self._clock())

    def check(self, stage: str) -> None:
        if self.remaining() <= 0:
            raise DeadlineExceeded(stage)

    def child(self, seconds: float) -> "Deadline":
        duration = float(seconds)
        if duration <= 0:
            raise ValueError("child deadline duration must be positive")
        return Deadline(min(self.expires_at, self._clock() + duration), self._clock)


__all__ = ("Deadline", "DeadlineExceeded")
