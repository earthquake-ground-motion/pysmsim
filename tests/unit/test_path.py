from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pysmsim.spectrum.attenuation import Q_MODELS, PiecewiseQ, QModel, Stafford2022Q
from pysmsim.spectrum.finite_fault import (
    FINITE_FAULT_MODELS,
    Generalized,
    NoFiniteFault,
    OneLine,
    RpsCombination,
    Stafford2022FiniteFault,
    TwoLines,
    TwoLinesSmoothed,
)
from pysmsim.spectrum.spreading import (
    SPREADING_MODELS,
    Atkinson2011,
    Atkinson2012,
    AtkinsonBoore2014,
    AtkinsonBoore2014R10,
    BooreTrial,
    FrequencyDependentSpreading,
    Hinged,
    SpreadingSegment,
    StaffordGeneralized,
    StaffordOptimal,
)

FREQS = np.logspace(-2, 2, 41)
BT15: dict[str, Any] = {"c1": -1.72, "c2": 0.43, "c3": -0.405, "c4": 0.235}
HINGED = Hinged(
    segments=(
        SpreadingSegment(r_start=1.0, a=-1.0, b=0.1, m_ref=6.5),
        SpreadingSegment(r_start=40.0, a=-0.5),
    )
)
FREQUENCY_DEPENDENT = [
    Atkinson2011(),
    BooreTrial(r1=50.0, slope_lf=-1.0, slope_hf=-1.5, slope_far=-0.5, ft1=1.0, ft2=4.0),
    Atkinson2012(),
    AtkinsonBoore2014(h=10.0),
    AtkinsonBoore2014R10(h=10.0),
]


def test_registries() -> None:
    for registry in (FINITE_FAULT_MODELS, SPREADING_MODELS, Q_MODELS):
        for name, cls in registry.items():
            assert cls.name == name
    assert len(FINITE_FAULT_MODELS) == 6
    assert len(SPREADING_MODELS) == 8
    assert len(Q_MODELS) == 2


# --- Finite-fault factor ------------------------------------------------------


def test_no_finite_fault() -> None:
    assert NoFiniteFault().rps(12.0, 7.0) == 12.0
    assert NoFiniteFault().h(7.0) == 0.0


def test_combinations() -> None:
    quad = OneLine(c1=-0.05, c2=0.15)
    total = OneLine(c1=-0.05, c2=0.15, combination=RpsCombination.SUM)
    h = 10 ** (-0.05 + 0.15 * 6.0)
    assert quad.h(6.0) == pytest.approx(h)
    assert quad.rps(10.0, 6.0) == pytest.approx(math.hypot(10.0, h))
    assert total.rps(10.0, 6.0) == pytest.approx(10.0 + h)


def test_two_lines_meet_at_the_hinge() -> None:
    ff = TwoLines(**BT15)
    mh = (BT15["c1"] - BT15["c3"]) / (BT15["c4"] - BT15["c2"])
    assert ff.h(mh - 1e-9) == pytest.approx(ff.h(mh + 1e-9), rel=1e-6)


def test_smoothed_lines_are_smooth() -> None:
    ff = TwoLinesSmoothed(**BT15, delta_m=2.0)
    lines = TwoLines(**BT15)
    mh = (BT15["c1"] - BT15["c3"]) / (BT15["c4"] - BT15["c2"])
    for m in (mh - 1.0, mh + 1.0):  # value and slope match the lines at both ends
        for dm in (-1e-6, 1e-6):
            assert ff.h(m + dm) == pytest.approx(lines.h(m + dm), rel=1e-5)
    assert ff.h(mh) < lines.h(mh)


def test_generalized_follows_the_lines_away_from_the_hinge() -> None:
    ff = Generalized(**BT15, h_ratio=0.8939)
    lines = TwoLines(**BT15)
    assert ff.h(3.0) == pytest.approx(lines.h(3.0), rel=0.02)
    assert ff.h(8.5) == pytest.approx(lines.h(8.5), rel=0.02)


def test_stafford_finite_fault_slopes() -> None:
    ff = Stafford2022FiniteFault(
        h_alpha=-0.8712, h_beta=0.4451, h_gamma=1.1513, h_delta=5.0948, h_epsilon=7.2725
    )
    # ln h has slope h_gamma well below h_epsilon and h_beta well above it
    slope_low = math.log(ff.h(4.01) / ff.h(4.0)) / 0.01
    slope_high = math.log(ff.h(10.01) / ff.h(10.0)) / 0.01
    assert slope_low == pytest.approx(1.1513, rel=1e-3)
    assert slope_high == pytest.approx(0.4451, rel=1e-3)


def test_finite_fault_checks() -> None:
    with pytest.raises(ValueError, match="parallel"):
        TwoLines(c1=1.0, c2=0.4, c3=0.0, c4=0.4)
    with pytest.raises(ValueError, match="delta_m"):
        TwoLinesSmoothed(**BT15, delta_m=0.0)
    with pytest.raises(ValueError, match="h_ratio"):
        Generalized(**BT15, h_ratio=1.0)


# --- Geometrical spreading ----------------------------------------------------


def test_hinged_is_continuous_and_extends_the_first_segment() -> None:
    m = 7.0
    slope1 = -1.0 + 0.1 * (m - 6.5)
    assert HINGED.value(40.0 - 1e-9, 40.0, m) == pytest.approx(HINGED.value(40.0, 40.0, m))
    assert HINGED.value(0.5, 0.5, m) == pytest.approx(0.5**slope1)
    assert HINGED.value(100.0, 100.0, m) == pytest.approx(40.0**slope1 * (100.0 / 40.0) ** -0.5)


def test_hinged_checks() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Hinged(
            segments=(SpreadingSegment(r_start=40.0, a=-0.5), SpreadingSegment(r_start=1.0, a=-1.0))
        )
    with pytest.raises(ValueError, match="at least one segment"):
        Hinged(segments=())


def test_stafford_generalized_slopes() -> None:
    g = StaffordGeneralized(gamma1=-1.3, r_t=50.0, gamma2=-0.5, g_ratio=1.2)
    near = math.log(g.value(1.01, 0, 6) / g.value(1.0, 0, 6)) / math.log(1.01)
    far = math.log(g.value(10100.0, 0, 6) / g.value(10000.0, 0, 6)) / math.log(1.01)
    assert near == pytest.approx(-1.3, abs=0.01)
    assert far == pytest.approx(-0.5, abs=0.01)
    # g_ratio is G over the near line at r_t (exactly so only when r_t >> r_ref)
    assert g.value(50.0, 0, 6) == pytest.approx(1.2 * 50.0**-1.3, rel=1e-4)


def test_stafford_optimal_uses_rrup_in_the_second_term() -> None:
    g = StaffordOptimal(gamma1=-1.1611, r_t=50.0, gamma2=-0.5)
    assert g.value(20.0, 15.0, 6.0) != g.value(20.0, 20.0, 6.0)
    assert g.value(1.0, 1.0, 6.0) == pytest.approx(1.0)


def test_frequency_independent_factor_has_the_shape_of_f() -> None:
    assert HINGED.factor(np.ones((2, 3)), 10.0, 10.0, 6.0).shape == (2, 3)
    assert np.all(HINGED.factor(FREQS, 10.0, 10.0, 6.0) == HINGED.value(10.0, 10.0, 6.0))
    assert not HINGED.frequency_dependent


@pytest.mark.parametrize("spreading", FREQUENCY_DEPENDENT, ids=lambda s: s.name)
@pytest.mark.parametrize("r", [0.5, 5.0, 30.0, 55.0, 200.0])
def test_spreading_kernel_matches_numpy_twin(
    spreading: FrequencyDependentSpreading, r: float
) -> None:
    args = spreading.kernel_args(r, r, 6.0)
    np.testing.assert_allclose(
        spreading.kernel.compiled(FREQS, *args), spreading.kernel.numpy(FREQS, *args), rtol=1e-14
    )
    assert spreading.factor(np.ones((2, 3)), r, r, 6.0).shape == (2, 3)


def test_boore_trial_is_continuous_at_r1() -> None:
    g = FREQUENCY_DEPENDENT[1]
    np.testing.assert_allclose(
        g.factor(FREQS, 50.0, 50.0, 6.0), g.factor(FREQS, 50.0 + 1e-9, 0.0, 6.0), rtol=1e-9
    )


def test_ab14_r10_is_continuous_at_10_km() -> None:
    g = AtkinsonBoore2014R10(h=5.0)
    np.testing.assert_allclose(
        g.factor(FREQS, 10.0 - 1e-9, 0.0, 6.0), g.factor(FREQS, 10.0, 0.0, 6.0), rtol=1e-8
    )


def test_spreading_checks() -> None:
    with pytest.raises(ValueError, match="not 1"):
        AtkinsonBoore2014(h=1.0)
    with pytest.raises(ValueError, match="ft1 < ft2"):
        BooreTrial(r1=50.0, slope_lf=-1.0, slope_hf=-1.5, slope_far=-0.5, ft1=4.0, ft2=1.0)
    with pytest.raises(ValueError, match="g_ratio"):
        StaffordGeneralized(gamma1=-1.3, r_t=50.0, gamma2=-0.5, g_ratio=1.0)


def test_reference_distance() -> None:
    assert Atkinson2011().reference_distance() == 1.0
    assert Atkinson2012(r_ref=2.0).reference_distance() == 2.0


# --- Q ------------------------------------------------------------------------

PIECEWISE = PiecewiseQ(fr1=1.0, qr1=100.0, s1=0.8, ft1=0.5, ft2=5.0, fr2=10.0, qr2=1000.0, s2=0.3)
STAFFORD = Stafford2022Q(q0=205.4, eta_a=0.6884, eta_b=0.1354, eta_g=5.1278)


@pytest.mark.parametrize("q", [PIECEWISE, STAFFORD], ids=lambda q: q.name)
def test_q_kernel_matches_numpy_twin(q: QModel) -> None:
    args = q.kernel_args(6.0)
    np.testing.assert_allclose(
        q.kernel.compiled(FREQS, *args), q.kernel.numpy(FREQS, *args), rtol=1e-14
    )


@pytest.mark.parametrize("q", [PIECEWISE, STAFFORD], ids=lambda q: q.name)
def test_no_attenuation_at_zero_frequency(q: QModel) -> None:
    assert q.q(0.0, 6.0) == np.inf
    assert q.attenuation(np.array([0.0, 1.0]), 6.0, 10.0, 10.0)[0] == 1.0


def test_piecewise_q_is_continuous() -> None:
    for ft in (0.5, 5.0):
        assert PIECEWISE.q(ft * (1 - 1e-9), 6.0) == pytest.approx(
            PIECEWISE.q(ft * (1 + 1e-9), 6.0), rel=1e-6
        )
    constant = PiecewiseQ(qr1=180.0, qr2=180.0)
    assert np.all(constant.q(FREQS, 6.0) == 180.0)


def test_path_factor_and_distances() -> None:
    f = np.array([1.0, 10.0])
    expected = np.exp(-math.pi * f * 50.0 / (3.5 * PIECEWISE.q(f, 6.0)))
    np.testing.assert_allclose(PIECEWISE.attenuation(f, 6.0, 50.0, 40.0), expected)
    assert PIECEWISE.distance(50.0, 40.0) == 50.0
    assert STAFFORD.distance(50.0, 40.0) == 40.0
    # The Stafford exponent depends on magnitude
    assert STAFFORD.q(10.0, 4.0) != STAFFORD.q(10.0, 7.0)


def test_q_checks() -> None:
    with pytest.raises(ValueError, match="positive"):
        PiecewiseQ(qr1=0.0, qr2=180.0)
    with pytest.raises(ValueError, match="ft1 <= ft2"):
        PiecewiseQ(qr1=180.0, qr2=180.0, ft1=2.0, ft2=1.0)
    with pytest.raises(ValueError, match="c_q"):
        Stafford2022Q(q0=200.0, eta_a=0.7, eta_b=0.1, eta_g=5.0, c_q=0.0)
