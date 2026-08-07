"""Immutable A/B carrier observation aggregated once per aligned source group."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, Sequence

import numpy as np

from watermark_v4.dct import TileScores
from watermark_v4.payload import carrier_class_for_tile


@dataclass(frozen=True, slots=True)
class CarrierEvidence:
    carrier_class: int
    tile_count: int
    phase_count: int
    minimum_coverage: float
    signal_score: float


@dataclass(frozen=True, slots=True)
class V4Observation:
    observed_codeword: bytes
    byte_confidences: tuple[float, ...]
    class_evidence: tuple[CarrierEvidence, CarrierEvidence]
    elapsed_seconds: float

    @property
    def tile_counts(self) -> tuple[int, int]:
        return tuple(item.tile_count for item in self.class_evidence)  # type: ignore[return-value]

    @property
    def phase_counts(self) -> tuple[int, int]:
        return tuple(item.phase_count for item in self.class_evidence)  # type: ignore[return-value]

    @property
    def coverage(self) -> tuple[float, float]:
        return tuple(item.minimum_coverage for item in self.class_evidence)  # type: ignore[return-value]


def _scores_to_bytes(scores: np.ndarray) -> bytes:
    bits = scores > 0.0
    return bytes(
        sum(int(bits[start + offset]) << (7 - offset) for offset in range(8))
        for start in range(0, 64, 8)
    )


def extract_observation(
    tiles: Sequence[TileScores],
    *,
    coverages: Sequence[float] | None = None,
    minimum_tiles_per_class: int = 1,
    minimum_phases: int = 2,
    minimum_coverage: float = 0.0,
    clock: Callable[[], float] = monotonic,
) -> V4Observation | None:
    started = clock()
    if minimum_tiles_per_class <= 0 or minimum_phases <= 0:
        raise ValueError("observation gates must be positive")
    if (
        type(minimum_coverage) is not float
        or not np.isfinite(minimum_coverage)
        or not 0.0 <= minimum_coverage <= 1.0
    ):
        raise ValueError("minimum observation coverage is invalid")
    coverage_values = (
        (1.0,) * len(tiles) if coverages is None else tuple(float(v) for v in coverages)
    )
    if len(coverage_values) != len(tiles) or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in coverage_values
    ):
        raise ValueError("observation coverages are invalid")

    codeword_parts: list[bytes] = []
    confidences: list[float] = []
    evidence: list[CarrierEvidence] = []
    accepted_phase_union: set[int] = set()
    for carrier_class in (0, 1):
        selected = [
            (tile, coverage_values[index])
            for index, tile in enumerate(tiles)
            if carrier_class_for_tile(tile.tile_x, tile.tile_y) == carrier_class
            and coverage_values[index] >= minimum_coverage
        ]
        if (
            len(selected) < minimum_tiles_per_class
        ):
            return None
        normalized: list[np.ndarray] = []
        accepted_coverages: list[float] = []
        accepted_phases: set[int] = set()
        for tile, coverage in selected:
            scores = np.asarray(tile.logical_scores, dtype=np.float64)
            energy = float(np.median(np.abs(scores)))
            if not np.isfinite(energy) or energy <= 1e-9:
                continue
            normalized.append(scores / energy)
            accepted_coverages.append(coverage)
            accepted_phases.add(tile.phase)
        if len(normalized) < minimum_tiles_per_class:
            return None
        accepted_phase_union.update(accepted_phases)
        aggregate = np.mean(np.stack(normalized), axis=0)
        codeword_parts.append(_scores_to_bytes(aggregate))
        confidences.extend(
            float(value / (1.0 + value))
            for value in (
                np.min(np.abs(aggregate[start : start + 8]))
                for start in range(0, 64, 8)
            )
        )
        evidence.append(
            CarrierEvidence(
                carrier_class=carrier_class,
                tile_count=len(normalized),
                phase_count=len(accepted_phases),
                minimum_coverage=min(accepted_coverages),
                signal_score=float(np.mean(np.abs(aggregate))),
            )
        )
    if len(accepted_phase_union) < minimum_phases:
        return None
    return V4Observation(
        observed_codeword=b"".join(codeword_parts),
        byte_confidences=tuple(confidences),
        class_evidence=(evidence[0], evidence[1]),
        elapsed_seconds=float(max(0.0, clock() - started)),
    )


__all__ = ("CarrierEvidence", "V4Observation", "extract_observation")
