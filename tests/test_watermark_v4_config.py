import pytest
from dataclasses import FrozenInstanceError

from watermark_v4 import V4Config


def test_defaults_match_v4_contract() -> None:
    config = V4Config()

    assert config.version == 4
    assert config.codec == "hmac32_rs_8_4_full_repeat_sync_v4"
    assert (config.tile_size, config.grid_size, config.cell_size) == (128, 8, 16)
    assert config.dct_margin == 6
    assert config.dct_margin_range == (2.0, 10.0)
    assert config.dct_margin_calibration == (4, 6, 8)
    assert config.coefficient_pairs == (((2, 3), (3, 2)), ((2, 4), (4, 2)))
    assert config.pilot_amplitude == 0.75
    assert config.pilot_amplitude_range == (0.25, 1.25)
    assert config.pilot_amplitude_calibration == (0.5, 0.75, 1.0)
    assert config.pilot_frequency_vectors == (
        (0.0703125, 0.1093750),
        (0.1015625, 0.1562500),
        (0.1406250, 0.0859375),
        (0.1718750, 0.1250000),
    )
    assert config.analysis_max_side == 1024
    assert config.minimum_coverage == 0.70
    assert config.minimum_tiles == 2
    assert config.minimum_phases == 2
    assert config.candidate_limit == 3
    assert config.online_p95_seconds == 10.0
    assert config.hard_timeout_seconds == 300.0
    assert config.stage_budgets_seconds == (0.3, 0.6, 1.0, 1.6, 5.0, 1.0)


@pytest.mark.parametrize("coverage", [0.0, 0.69])
def test_config_cannot_weaken_seventy_percent_tile_coverage(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        V4Config(minimum_coverage=coverage)


def test_is_frozen_slotted_and_keeps_protocol_identity_out_of_constructor() -> None:
    config = V4Config()

    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.tile_size = 256  # type: ignore[misc]
    with pytest.raises(TypeError):
        V4Config(version=5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        V4Config(codec="other")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tile_size", 128),
        ("grid_size", 8),
        ("cell_size", 16),
        ("dct_margin_range", (2, 10)),
        ("dct_margin_calibration", (4, 6, 8)),
        ("coefficient_pairs", (((2, 3), (3, 2)), ((2, 4), (4, 2)))),
        ("pilot_amplitude_range", (0.25, 1.25)),
        ("pilot_amplitude_calibration", (0.5, 0.75, 1.0)),
        (
            "pilot_frequency_vectors",
            (
                (0.0703125, 0.1093750),
                (0.1015625, 0.1562500),
                (0.1406250, 0.0859375),
                (0.1718750, 0.1250000),
            ),
        ),
    ),
)
def test_hard_format_fields_cannot_be_passed_to_constructor(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError):
        V4Config(**{field: value})  # type: ignore[arg-type]


def test_tunable_numeric_fields_accept_finite_ints_and_floats() -> None:
    config = V4Config(
        dct_margin=5.5,
        pilot_amplitude=1,
        stage_budgets_seconds=(1, 1.0, 1, 1.0, 1, 1.0),
    )

    assert config.dct_margin == 5.5
    assert config.pilot_amplitude == 1
    assert config.stage_budgets_seconds == (1, 1.0, 1, 1.0, 1, 1.0)


def test_huge_scalar_integer_reaches_normal_range_validation() -> None:
    with pytest.raises(ValueError, match="dct_margin_range"):
        V4Config(dct_margin=10**1000)


def test_huge_tuple_integer_reaches_normal_budget_validation() -> None:
    with pytest.raises(ValueError, match="stage budgets"):
        V4Config(stage_budgets_seconds=(10**1000, 1, 1, 1, 1, 1))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"online_p95_seconds": 10.1, "hard_timeout_seconds": 16.0},
        {"online_p95_seconds": 10.0, "hard_timeout_seconds": 300.1},
    ),
)
def test_latency_limits_cannot_exceed_service_hard_caps(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must be at most"):
        V4Config(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("analysis_max_side", 1024.0),
        ("minimum_tiles", True),
        ("minimum_phases", 2.0),
        ("candidate_limit", False),
    ),
)
def test_integer_fields_require_exact_integers(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be an integer"):
        V4Config(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dct_margin", True),
        ("dct_margin", float("nan")),
        ("pilot_amplitude", True),
        ("pilot_amplitude", float("nan")),
        ("minimum_coverage", float("inf")),
        ("online_p95_seconds", False),
        ("hard_timeout_seconds", float("-inf")),
    ),
)
def test_real_fields_require_finite_non_boolean_numbers(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a finite number"):
        V4Config(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_budgets_seconds", [0.3, 0.6, 1.0, 1.6, 5.0, 1.0]),
    ),
)
def test_structured_fields_require_immutable_tuples(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a tuple"):
        V4Config(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_budgets_seconds", (0.3, 0.6)),
    ),
)
def test_structured_fields_require_exact_shapes(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"{field} has invalid shape"):
        V4Config(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stage_budgets_seconds", (0.3, 0.6, 1.0, 1.6, True, 1.0)),
    ),
)
def test_structured_numeric_values_have_exact_finite_types(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=rf"{field} contains an invalid number"):
        V4Config(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dct_margin": 1},
        {"dct_margin": 11},
        {"pilot_amplitude": 0.2},
        {"pilot_amplitude": 1.3},
    ),
)
def test_rejects_dct_and_pilot_values_outside_hard_ranges(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="range"):
        V4Config(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"analysis_max_side": 64},
        {"minimum_coverage": -0.01},
        {"minimum_coverage": 1.01},
        {"minimum_tiles": 1},
        {"minimum_phases": 1},
        {"minimum_phases": 5},
        {"candidate_limit": 0},
        {"candidate_limit": 4},
    ),
)
def test_rejects_invalid_analysis_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        V4Config(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"stage_budgets_seconds": (0.3, 0.6, 0.0, 1.6, 5.0, 1.0)},
        {"stage_budgets_seconds": (2.0, 2.0, 2.0, 2.0, 2.0, 1.0)},
        {"online_p95_seconds": 0.0},
        {"hard_timeout_seconds": 0.0},
        {"hard_timeout_seconds": 9.0},
    ),
)
def test_rejects_invalid_latency_budgets(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        V4Config(**kwargs)  # type: ignore[arg-type]
