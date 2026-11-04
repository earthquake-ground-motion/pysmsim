from __future__ import annotations

import math

import pytest

from pysmsim.spectrum.stress import (
    STRESS_MODELS,
    HoerlStress,
    LogLinearStress,
    Stafford2022Stress,
)


def test_registry_names_match_classes() -> None:
    assert set(STRESS_MODELS) == {"log_linear", "hoerl", "stafford_2022"}
    for name, cls in STRESS_MODELS.items():
        assert cls.name == name


def test_log_linear() -> None:
    stress = LogLinearStress(stress_ref=100.0, dlog10_stress_dm=0.1, magnitude_ref=6.0)
    assert stress(7.0) == pytest.approx(100.0 * 10**0.1)
    assert LogLinearStress(stress_ref=80.0)(3.0) == 80.0
    with pytest.raises(ValueError, match="stress_ref must be positive"):
        LogLinearStress(stress_ref=0.0)


def test_hoerl() -> None:
    stress = HoerlStress(c1=26900.0, c2=-5.057, c3=0.652)
    assert stress(6.5) == pytest.approx(144.0, rel=0.01)  # see params-file notes


def test_stafford_2022() -> None:
    stress = Stafford2022Stress(c1=4.599, c2=0.4624)
    assert stress(6.0) == pytest.approx(math.exp(4.599))
    assert stress(4.0) == pytest.approx(math.exp(4.599 - 0.4624))
