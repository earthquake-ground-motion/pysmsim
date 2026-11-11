"""Unit tests of the whole spectrum, the responses, the distances and the FAS loop."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pysmsim.drivers.loop_fas import (
    FAS_DETAIL_COLUMNS,
    FAS_SCENARIO_COLUMNS,
    fas_column,
    fas_loop,
    fas_loop_details,
    params_for,
)
from pysmsim.io.config import load_config
from pysmsim.params import Medium
from pysmsim.scenario import DepthFunction, DistanceMetric, Mechanism, RjbToRrup, ScenarioGrid
from pysmsim.spectrum.distance import cy14_ztor, rjb_to_rrup, rjb_to_rrup_depth
from pysmsim.spectrum.fas import fourier_spectrum, scenario_terms, spectrum_factors
from pysmsim.spectrum.response import (
    GroundMotion,
    MotionQuantity,
    Oscillator,
    WoodAnderson,
    WWSSNShortPeriod,
)
from pysmsim.spectrum.source import Haddon1996
from pysmsim.spectrum.stress import LogLinearStress

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
PARAMS = load_config(EXAMPLE).params
FREQS = np.concatenate(([0.0], np.logspace(-2, 2, 81)))

# --- Responses ----------------------------------------------------------------


def test_ground_motion_powers() -> None:
    f = np.array([0.0, 0.5, 2.0])
    w = 2.0 * math.pi * f
    assert GroundMotion().factor(f) == pytest.approx(np.ones(3))
    assert GroundMotion(MotionQuantity.VELOCITY).factor(f) == pytest.approx(w)
    assert GroundMotion(MotionQuantity.ACCELERATION).factor(f) == pytest.approx(w**2)
    assert GroundMotion().factor(np.ones((2, 3))).shape == (2, 3)


def test_oscillator_kernel_matches_numpy_twin() -> None:
    osc = Oscillator(period=0.3, damping=0.05)
    np.testing.assert_allclose(
        osc.kernel.compiled(FREQS, osc.frequency, 0.05),
        osc.kernel.numpy(FREQS, osc.frequency, 0.05),
        rtol=1e-14,
    )


def test_oscillator_response() -> None:
    osc = Oscillator(period=0.5, damping=0.05)
    w = 2.0 * math.pi * osc.frequency
    # Resonance: w * 1/(2*damping); far above: w (PSV = w * ground displacement)
    assert osc.factor(osc.frequency) == pytest.approx(w / 0.1)
    assert osc.factor(1.0e4) == pytest.approx(w, rel=1e-6)
    assert osc.factor(0.0) == 0.0
    assert osc.factor(np.ones((2, 2))).shape == (2, 2)
    with pytest.raises(ValueError, match="period"):
        Oscillator(period=0.0)
    with pytest.raises(ValueError, match="damping"):
        Oscillator(period=1.0, damping=1.0)


def test_instruments_have_their_gain() -> None:
    assert WoodAnderson().factor(100.0) == pytest.approx(2080.0)
    assert WWSSNShortPeriod().factor(1.0) == pytest.approx(1.0)
    assert WoodAnderson().factor(0.0) == 0.0
    # Wood-Anderson: flat displacement response well above 1.25 Hz
    assert WoodAnderson().factor(50.0) == pytest.approx(2080.0, rel=1e-3)


# --- Distances ----------------------------------------------------------------


def test_cy14_ztor() -> None:
    assert cy14_ztor(4.0, Mechanism.STRIKE_SLIP) == pytest.approx(2.673**2)
    assert cy14_ztor(6.0, Mechanism.STRIKE_SLIP) == pytest.approx((2.673 - 1.136 * 1.03) ** 2)
    assert cy14_ztor(6.0, Mechanism.NORMAL) == cy14_ztor(6.0, Mechanism.STRIKE_SLIP)
    assert cy14_ztor(9.0, Mechanism.STRIKE_SLIP) == 0.0
    assert cy14_ztor(5.0, Mechanism.REVERSE) == pytest.approx(2.704**2)


def test_rjb_to_rrup() -> None:
    assert rjb_to_rrup(10.0, 6.0, conversion=RjbToRrup.NONE) == 10.0
    h = cy14_ztor(6.0, Mechanism.REVERSE)
    rrup = rjb_to_rrup(10.0, 6.0, conversion=RjbToRrup.CY14_ZTOR, mechanism=Mechanism.REVERSE)
    assert rrup == pytest.approx(math.hypot(10.0, h))
    depth = DepthFunction(m1=3.75, h1=8.0, c2=-1.706666667, c3=0.303407407, m2=7.5, h2=0.0)
    assert (
        rjb_to_rrup_depth(3.0, conversion=RjbToRrup.MAGNITUDE_FUNCTION, depth_function=depth) == 8.0
    )
    assert rjb_to_rrup_depth(
        7.5 - 1e-9, conversion=RjbToRrup.MAGNITUDE_FUNCTION, depth_function=depth
    ) == pytest.approx(0.0, abs=1e-6)
    assert rjb_to_rrup_depth(6.0, conversion=RjbToRrup.MAGNITUDE_FUNCTION) == 0.0


# --- Spectrum -----------------------------------------------------------------


def test_radiation_constant() -> None:
    medium = Medium(density=2.8, shear_velocity=3.6, partition=0.71)
    expected = 0.71 * 0.55 * 2.0 * 1e-20 / (4.0 * math.pi * 2.8 * 3.6**3)
    assert medium.radiation_constant() == pytest.approx(expected, rel=1e-15)
    assert medium.radiation_constant(2.0) == pytest.approx(expected / 2.0, rel=1e-15)


def test_scenario_terms() -> None:
    terms = scenario_terms(PARAMS, 6.0, 20.0)
    assert terms.rps == pytest.approx(PARAMS.finite_fault.rps(20.0, 6.0))
    assert terms.finite_fault_h == pytest.approx(PARAMS.finite_fault.h(6.0))
    assert terms.source.m0 == pytest.approx(10.0 ** (1.5 * 6.0 + 16.05))


def test_spectrum_is_the_product_of_its_factors() -> None:
    factors = spectrum_factors(FREQS, PARAMS, 6.0, 20.0)
    product = factors.terms.radiation_constant * factors.terms.source.m0
    for name in (
        "source",
        "spreading",
        "q_path",
        "site_amplification",
        "site_diminution",
        "low_cut",
        "high_cut",
        "response",
    ):
        part = getattr(factors, name)
        assert part.shape == FREQS.shape, name
        product = product * part
    np.testing.assert_allclose(fourier_spectrum(FREQS, PARAMS, 6.0, 20.0), product, rtol=1e-14)


def test_spectrum_with_a_response() -> None:
    acc = GroundMotion(MotionQuantity.ACCELERATION)
    fds = fourier_spectrum(FREQS, PARAMS, 6.0, 20.0)
    fas = fourier_spectrum(FREQS, PARAMS, 6.0, 20.0, acc)
    np.testing.assert_allclose(fas, (2.0 * math.pi * FREQS) ** 2 * fds, rtol=1e-14)
    assert fourier_spectrum(np.ones((2, 3)), PARAMS, 6.0, 20.0).shape == (2, 3)


def test_displacement_spectrum_is_flat_at_low_frequency() -> None:
    # Below the corner and the site terms, A = C * M0 * G * A_site(f_min)
    terms = scenario_terms(PARAMS, 5.0, 10.0)
    low = fourier_spectrum(0.0, PARAMS, 5.0, 10.0)
    gsprd = PARAMS.spreading.factor(0.0, terms.rps, 10.0, 5.0)
    expected = terms.radiation_constant * terms.source.m0 * gsprd
    expected = expected * PARAMS.site.amplification(0.0)
    assert low == pytest.approx(expected, rel=1e-14)


# --- FAS loop -----------------------------------------------------------------

GRID = ScenarioGrid(magnitudes=(5.0, 6.0), distances=(0.0, 10.0, 50.0))
F = np.array([0.0, 0.1, 1.0, 10.0])


def test_fas_loop_details_table() -> None:
    table = fas_loop_details(PARAMS, GRID, F)
    assert tuple(table.columns) == FAS_DETAIL_COLUMNS
    assert len(table) == 2 * 3 * 4
    # Loop order: magnitude, distance, frequency
    assert list(table["magnitude"][:12]) == [5.0] * 12
    assert list(table["rjb"][:8]) == [0.0] * 4 + [10.0] * 4
    assert list(table["f"][:4]) == list(F)
    assert table["period"][0] == math.inf
    row = table.iloc[6]  # M 5, Rjb 10 km, 1 Hz
    assert row["h"] == pytest.approx(cy14_ztor(5.0, Mechanism.STRIKE_SLIP))
    assert row["rrup"] == pytest.approx(math.hypot(10.0, row["h"]))
    expected = fourier_spectrum(1.0, PARAMS, 5.0, row["rrup"])
    assert row["fds"] == pytest.approx(expected)
    assert row["fas"] == pytest.approx((2.0 * math.pi) ** 2 * row["fds"])


def test_fas_loop_one_row_per_scenario() -> None:
    grid = replace(GRID, kappa0=(0.01, 0.04))
    details = fas_loop_details(PARAMS, grid, F)
    table = fas_loop(PARAMS, grid, F)
    names = ["fas(0)", "fas(0.1)", "fas(1)", "fas(10)"]
    assert list(table.columns) == [*FAS_SCENARIO_COLUMNS, *names]
    # Order: kappa_0, magnitude, distance
    assert len(table) == 2 * 2 * 3
    assert table.kappa0.tolist() == [0.01] * 6 + [0.04] * 6
    assert table.magnitude.tolist()[:6] == [5.0] * 3 + [6.0] * 3
    assert table.rjb.tolist()[:3] == [0.0, 10.0, 50.0]
    for i, f in enumerate(F):
        rows = details[details.f == f]
        np.testing.assert_array_equal(table[names[i]], rows.fas)
        for name in FAS_SCENARIO_COLUMNS:
            np.testing.assert_array_equal(table[name], rows[name], err_msg=name)


def test_fas_loop_rejects_bad_frequencies() -> None:
    for f in ([], [1.0, 1.0000001], np.ones((2, 2))):
        with pytest.raises(ValueError, match="frequencies"):
            fas_loop(PARAMS, GRID, f)


def test_fas_column() -> None:
    assert fas_column(0.01) == "fas(0.01)"
    assert fas_column(100.0) == "fas(100)"


def test_fas_loop_with_rrup() -> None:
    grid = replace(GRID, distances=(10.0,), distance_metric=DistanceMetric.RRUP)
    table = fas_loop_details(PARAMS, grid, F)
    assert table["rjb"].isna().all() and table["h"].isna().all()
    assert (table["rrup"] == 10.0).all()


def test_fas_loop_kappa_and_stress_loops() -> None:
    grid = replace(GRID, kappa0=(0.01, 0.04), stress=(50.0, 100.0, 200.0))
    table = fas_loop_details(PARAMS, grid, F)
    assert len(table) == 2 * 3 * 2 * 3 * 4
    assert list(table["kappa0"].unique()) == [0.01, 0.04]
    assert list(table["stress"][: 2 * 3 * 4].unique()) == [50.0]
    with pytest.raises(ValueError, match="one-dimensional"):
        fas_loop_details(PARAMS, GRID, np.ones((2, 2)))


def test_params_for() -> None:
    site = replace(PARAMS.site, dkappa_dmag=0.01, kappa_magnitude_ref=6.0)
    params = replace(PARAMS, site=site)
    assert params_for(params).site == site
    assert params_for(params, fortran_compat=True).site.dkappa_dmag == 0.0
    override = params_for(params, kappa0=0.02).site
    assert (override.kappa0, override.dkappa_dmag) == (0.02, 0.0)
    stressed = params_for(params, stress=123.0).source
    assert stressed.stress == LogLinearStress(stress_ref=123.0)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="no stress parameter"):
        params_for(replace(params, source=Haddon1996()), stress=100.0)
