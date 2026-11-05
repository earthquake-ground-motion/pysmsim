from __future__ import annotations

import pytest

from pysmsim.params import Medium, OscillatorCorrection, PathDuration
from pysmsim.scenario import Scenario, ScenarioGrid


def test_enums_take_their_string_values() -> None:
    assert OscillatorCorrection("bt15_wna") is OscillatorCorrection.BT15_WNA


def test_medium_defaults() -> None:
    medium = Medium(density=2.8, shear_velocity=3.6)
    assert medium.partition == 0.707
    assert medium.radiation == 0.55
    assert medium.free_surface == 2.0


def test_path_duration_checks() -> None:
    with pytest.raises(ValueError, match="same length"):
        PathDuration(distances=(0.0, 10.0), durations=(0.0,))


def test_scenario_checks() -> None:
    assert Scenario(magnitude=6.0, rrup=10.0, rjb=8.0).rjb == 8.0
    with pytest.raises(ValueError, match="rrup must be positive"):
        Scenario(magnitude=6.0, rrup=0.0)
    with pytest.raises(ValueError, match="between 0 and rrup"):
        Scenario(magnitude=6.0, rrup=10.0, rjb=12.0)


def test_scenario_grid_needs_a_measure() -> None:
    with pytest.raises(ValueError, match="at least one period"):
        ScenarioGrid(magnitudes=(6.0,), distances=(10.0,), pga=False, pgv=False)
