from dataclasses import dataclass, field
from math import isfinite


Coordinate = tuple[int, int]
CoefficientPair = tuple[Coordinate, Coordinate]
FrequencyVector = tuple[float, float]


@dataclass(frozen=True, slots=True)
class V4Config:
    version: int = field(default=4, init=False)
    codec: str = field(
        default="hmac32_rs_8_4_full_repeat_sync_v4",
        init=False,
    )
    tile_size: int = field(default=128, init=False)
    grid_size: int = field(default=8, init=False)
    cell_size: int = field(default=16, init=False)
    dct_margin: float = 6.0
    dct_margin_range: tuple[float, float] = field(
        default=(2.0, 10.0), init=False
    )
    dct_margin_calibration: tuple[int, int, int] = field(
        default=(4, 6, 8), init=False
    )
    coefficient_pairs: tuple[CoefficientPair, CoefficientPair] = field(
        default=(((2, 3), (3, 2)), ((2, 4), (4, 2))), init=False
    )
    pilot_amplitude: float = 0.75
    pilot_amplitude_range: tuple[float, float] = field(
        default=(0.25, 1.25), init=False
    )
    pilot_amplitude_calibration: tuple[float, float, float] = field(
        default=(0.5, 0.75, 1.0), init=False
    )
    pilot_frequency_vectors: tuple[
        FrequencyVector, FrequencyVector, FrequencyVector, FrequencyVector
    ] = field(
        default=(
            (0.0703125, 0.1093750),
            (0.1015625, 0.1562500),
            (0.1406250, 0.0859375),
            (0.1718750, 0.1250000),
        ),
        init=False,
    )
    analysis_max_side: int = 1024
    minimum_coverage: float = 0.70
    minimum_tiles: int = 2
    minimum_phases: int = 2
    candidate_limit: int = 3
    online_p95_seconds: float = 10.0
    hard_timeout_seconds: float = 300.0
    stage_budgets_seconds: tuple[float, float, float, float, float, float] = (
        0.3,
        0.6,
        1.0,
        1.6,
        5.0,
        1.0,
    )

    def __post_init__(self) -> None:
        integer_fields = (
            "tile_size",
            "grid_size",
            "cell_size",
            "analysis_max_side",
            "minimum_tiles",
            "minimum_phases",
            "candidate_limit",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")

        real_fields = (
            "dct_margin",
            "pilot_amplitude",
            "minimum_coverage",
            "online_p95_seconds",
            "hard_timeout_seconds",
        )
        for name in real_fields:
            if not _is_finite_number(getattr(self, name)):
                raise TypeError(f"{name} must be a finite number")

        tuple_shapes = (
            ("dct_margin_range", self.dct_margin_range, 2),
            ("dct_margin_calibration", self.dct_margin_calibration, 3),
            ("coefficient_pairs", self.coefficient_pairs, 2),
            ("pilot_amplitude_range", self.pilot_amplitude_range, 2),
            ("pilot_amplitude_calibration", self.pilot_amplitude_calibration, 3),
            ("pilot_frequency_vectors", self.pilot_frequency_vectors, 4),
            ("stage_budgets_seconds", self.stage_budgets_seconds, 6),
        )
        for name, value, length in tuple_shapes:
            _require_tuple(name, value, length)

        _require_nested_pairs("coefficient_pairs", self.coefficient_pairs)
        _require_pair_items("pilot_frequency_vectors", self.pilot_frequency_vectors)

        integer_tuples = (
            ("dct_margin_calibration", self.dct_margin_calibration),
            (
                "coefficient_pairs",
                tuple(
                    component
                    for pair in self.coefficient_pairs
                    for coordinate in pair
                    for component in coordinate
                ),
            ),
        )
        for name, values in integer_tuples:
            if any(type(value) is not int for value in values):
                raise TypeError(f"{name} contains an invalid number")

        real_tuples = (
            ("dct_margin_range", self.dct_margin_range),
            ("pilot_amplitude_range", self.pilot_amplitude_range),
            ("pilot_amplitude_calibration", self.pilot_amplitude_calibration),
            (
                "pilot_frequency_vectors",
                tuple(value for vector in self.pilot_frequency_vectors for value in vector),
            ),
            ("stage_budgets_seconds", self.stage_budgets_seconds),
        )
        for name, values in real_tuples:
            if any(not _is_finite_number(value) for value in values):
                raise TypeError(f"{name} contains an invalid number")

        if self.tile_size != self.grid_size * self.cell_size:
            raise ValueError("tile_size must equal grid_size * cell_size")

        dct_low, dct_high = self.dct_margin_range
        if dct_low > dct_high:
            raise ValueError("dct_margin_range must be ordered")
        if not dct_low <= self.dct_margin <= dct_high:
            raise ValueError("dct_margin must be within dct_margin_range")
        if any(not dct_low <= value <= dct_high for value in self.dct_margin_calibration):
            raise ValueError("dct_margin_calibration must be within dct_margin_range")

        pilot_low, pilot_high = self.pilot_amplitude_range
        if pilot_low > pilot_high:
            raise ValueError("pilot_amplitude_range must be ordered")
        if not pilot_low <= self.pilot_amplitude <= pilot_high:
            raise ValueError("pilot_amplitude must be within pilot_amplitude_range")
        if any(
            not pilot_low <= value <= pilot_high
            for value in self.pilot_amplitude_calibration
        ):
            raise ValueError(
                "pilot_amplitude_calibration must be within pilot_amplitude_range"
            )

        for pair in self.coefficient_pairs:
            for row, column in pair:
                if not (0 <= row < self.cell_size and 0 <= column < self.cell_size):
                    raise ValueError("coefficient coordinate out of bounds")

        if self.analysis_max_side < self.tile_size:
            raise ValueError("analysis_max_side must be at least tile_size")
        if not 0.70 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between 0.70 and one")
        if self.minimum_tiles < 2:
            raise ValueError("minimum_tiles must be at least 2")
        if not 2 <= self.minimum_phases <= 4:
            raise ValueError("minimum_phases must be between 2 and 4")
        if not 1 <= self.candidate_limit <= 3:
            raise ValueError("candidate_limit must be between 1 and 3")

        if self.online_p95_seconds <= 0:
            raise ValueError("online_p95_seconds must be positive")
        if self.online_p95_seconds > 10.0:
            raise ValueError("online_p95_seconds must be at most 10.0")
        if self.hard_timeout_seconds <= 0:
            raise ValueError("hard_timeout_seconds must be positive")
        if self.hard_timeout_seconds > 300.0:
            raise ValueError("hard_timeout_seconds must be at most 300.0")
        if any(value <= 0 for value in self.stage_budgets_seconds):
            raise ValueError("stage_budgets_seconds values must be positive")
        if sum(self.stage_budgets_seconds) > self.online_p95_seconds:
            raise ValueError("stage budgets must not exceed online_p95_seconds")
        if self.hard_timeout_seconds < self.online_p95_seconds:
            raise ValueError("hard_timeout_seconds must be at least online_p95_seconds")


def _is_finite_number(value: object) -> bool:
    if type(value) is int:
        return True
    if type(value) is float:
        return isfinite(value)
    return False


def _require_tuple(name: str, value: object, length: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(value) != length:
        raise ValueError(f"{name} has invalid shape")


def _require_nested_pairs(name: str, values: tuple[object, ...]) -> None:
    for pair in values:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name} has invalid shape")
        for coordinate in pair:
            if type(coordinate) is not tuple or len(coordinate) != 2:
                raise ValueError(f"{name} has invalid shape")


def _require_pair_items(name: str, values: tuple[object, ...]) -> None:
    for pair in values:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name} has invalid shape")
