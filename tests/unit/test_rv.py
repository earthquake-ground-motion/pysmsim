"""Unit tests of the random-vibration modules and the RV loop."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from pysmsim.drivers.loop_fas import params_for
from pysmsim.drivers.loop_rv import (
    RV_DETAIL_COLUMNS,
    SCENARIO_COLUMNS,
    psa_column,
    rv_loop,
    rv_loop_details,
)
from pysmsim.io.bt15_tables import BT15_TABLE_NAMES, load_bt15_table, read_bt15_table
from pysmsim.io.config import load_config
from pysmsim.params import OscillatorCorrection as Choice
from pysmsim.params import PathDuration, RVConfig
from pysmsim.rv.duration import (
    BooreJoyner1984,
    BooreThompson2015,
    BooreThompson2015Average,
    LiuPezeshk1999,
    NoCorrection,
    oscillator_correction,
    oscillator_decay_time,
)
from pysmsim.rv.gm_rv import rv_fup, rv_motion
from pysmsim.rv.moments import (
    GAUSS_WEIGHTS,
    KRONROD_WEIGHTS,
    NODES,
    SpectralMoments,
    integrate,
    spectral_moments,
    upper_frequency,
)
from pysmsim.rv.peak_factors import (
    cartwright_longuet_higgins,
    cartwright_longuet_higgins_asymptotic,
    der_kiureghian_1980,
    der_kiureghian_1980_asymptotic,
    der_kiureghian_1980_cdf,
    der_kiureghian_1980_fortran,
    der_kiureghian_1980_pdf,
    extrema,
    zero_crossings,
)
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.fas import fourier_spectrum
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator, WoodAnderson

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
PARAMS = load_config(EXAMPLE).params
DATA = Path(__file__).parents[2] / "src" / "pysmsim" / "data" / "bt15"

# --- Integration and moments ------------------------------------------------------


@pytest.mark.parametrize("degree", range(23))
def test_gauss_kronrod_rules_are_exact_for_polynomials(degree: int) -> None:
    exact = (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)
    assert np.sum(KRONROD_WEIGHTS * NODES**degree) == pytest.approx(exact, abs=1e-14)
    if degree <= 13:
        assert np.sum(GAUSS_WEIGHTS * NODES**degree) == pytest.approx(exact, abs=1e-14)


def test_integrate_peaked_and_smooth_functions() -> None:
    def func(x: np.ndarray) -> np.ndarray:
        return np.vstack([1.0 / ((x - 1.0) ** 2 + 1e-4), np.exp(-x) * x**2])

    lorentz = (math.atan(2.0 / 1e-2) + math.atan(1.0 / 1e-2)) / 1e-2
    gamma = 2.0 - math.exp(-3.0) * 17.0
    result = integrate(func, [0.0, 3.0], rtol=1e-10)
    np.testing.assert_allclose(result, [lorentz, gamma], rtol=1e-10)


def test_integrate_rejects_bad_edges() -> None:
    with pytest.raises(ValueError, match="increasing"):
        integrate(lambda x: x[None, :], [1.0, 0.0])


def test_upper_frequency() -> None:
    assert upper_frequency(0.0, 0.035, 1e-3) == pytest.approx(-math.log(1e-3) / (math.pi * 0.035))
    assert upper_frequency(15.0, 0.0, 1e-3) == pytest.approx(15.0 / 1e-3**0.25)
    assert upper_frequency(5.0, 0.035, 1e-3) == pytest.approx(5.0 / 1e-3**0.25)
    assert upper_frequency(100.0, 0.035, 1e-3) == pytest.approx(-math.log(1e-3) / (math.pi * 0.035))
    with pytest.raises(ValueError, match="kappa or an fmax"):
        upper_frequency(0.0, 0.0, 1e-3)
    with pytest.raises(ValueError, match="negative"):
        upper_frequency(0.0, -0.01, 1e-3)
    with pytest.raises(ValueError, match="amp_cutoff"):
        upper_frequency(0.0, 0.03, 1.5)


def test_moments_of_a_box_spectrum() -> None:
    # A = 1 below 1 Hz: m_k = 2 * (2*pi)**k / (k + 1)
    moments = spectral_moments(
        lambda f: (f < 1.0).astype(np.float64), 2.0, rtol=1e-10, breakpoints=(1.0,)
    )
    w = 2.0 * math.pi
    assert moments.m0 == pytest.approx(2.0, rel=1e-10)
    assert moments.m1 == pytest.approx(w, rel=1e-10)
    assert moments.m2 == pytest.approx(2.0 * w**2 / 3.0, rel=1e-10)
    assert moments.m4 == pytest.approx(2.0 * w**4 / 5.0, rel=1e-10)
    assert moments.dominant_frequency == pytest.approx(math.sqrt(1.0 / 3.0))
    assert moments.extrema_frequency == pytest.approx(math.sqrt(3.0 / 5.0))
    assert moments.bandwidth == pytest.approx(math.sqrt(1.0 - 0.75))
    assert moments.xi == pytest.approx(math.sqrt(5.0) / 3.0)
    assert moments.epsilon == pytest.approx(2.0 / 3.0)


def test_moments_round_off_gives_zero_bandwidth() -> None:
    moments = SpectralMoments(m0=1.0, m1=1.0 + 1e-12, m2=1.0, m4=1.0 - 1e-12)
    assert moments.bandwidth == 0.0
    assert moments.epsilon == 0.0


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_moments_of_an_oscillator_match_scipy_quad() -> None:
    osc = Oscillator(period=0.3, damping=0.02)
    fup = rv_fup(PARAMS, 6.0)

    def spectrum(f: np.ndarray) -> np.ndarray:
        return fourier_spectrum(f, PARAMS, 6.0, 20.0, osc)

    moments = spectral_moments(spectrum, fup, rtol=1e-8, breakpoints=(osc.frequency,))
    for k, value in [(0, moments.m0), (1, moments.m1), (2, moments.m2), (4, moments.m4)]:
        expected, _ = quad(
            lambda f, k=k: float(spectrum(np.array([f]))[0]) ** 2 * (2.0 * math.pi * f) ** k,
            0.0,
            fup,
            points=[osc.frequency],
            limit=1000,
            epsrel=1e-9,
        )
        assert value == pytest.approx(2.0 * expected, rel=1e-7)


def test_fup_uses_kappa_at_the_magnitude() -> None:
    params = replace(PARAMS, site=replace(PARAMS.site, dkappa_dmag=0.01, kappa_magnitude_ref=6.0))
    assert rv_fup(params, 7.0) == pytest.approx(upper_frequency(0.0, 0.045, 1e-3))
    assert rv_fup(params, 7.0, fortran_compat=True) == pytest.approx(
        upper_frequency(0.0, PARAMS.site.kappa0, 1e-3)
    )


# --- Durations ------------------------------------------------------------------


def test_path_duration() -> None:
    path = PathDuration(distances=(10.0, 70.0, 130.0), durations=(1.0, 9.6, 7.8), slope_beyond=0.04)
    assert path.duration(5.0) == 1.0
    assert path.duration(40.0) == pytest.approx(1.0 + 30.0 * 8.6 / 60.0)
    assert path.duration(130.0) == pytest.approx(7.8)
    assert path.duration(230.0) == pytest.approx(7.8 + 4.0)


def test_simple_corrections() -> None:
    kw = {"period": 2.0, "damping": 0.05, "magnitude": 6.0, "rps": 20.0, "bandwidth": 0.3}
    tosc = oscillator_decay_time(2.0, 0.05)
    assert tosc == pytest.approx(2.0 / (2.0 * math.pi * 0.05))
    assert NoCorrection().rms_duration(5.0, **kw) == 5.0
    r = (5.0 / 2.0) ** 3
    assert BooreJoyner1984().rms_duration(5.0, **kw) == pytest.approx(5.0 + tosc * r / (r + 1 / 3))
    r = (5.0 / 2.0) ** 2
    alpha = 0.3 * math.sqrt(2.0 * math.pi)
    assert LiuPezeshk1999().rms_duration(5.0, **kw) == pytest.approx(5.0 + tosc * r / (r + alpha))
    for correction in (NoCorrection(), BooreJoyner1984(), LiuPezeshk1999()):
        assert correction.peak_motion_factor(MotionQuantity.ACCELERATION, 6.0, 20.0) == 1.0


@pytest.mark.parametrize("name", BT15_TABLE_NAMES)
def test_bt15_tables_load(name: str) -> None:
    table = load_bt15_table(name)
    assert table.magnitudes.tolist() == [2.0 + 0.5 * i for i in range(13)]
    assert 10.0 ** table.log_distances[0] == pytest.approx(2.0)
    assert 10.0 ** table.log_distances[-1] == pytest.approx(1262.0)
    assert table.coefficients.shape == (13, 15, 7)
    assert load_bt15_table(name) is table  # cached
    assert read_bt15_table(DATA / f"{name}.pars").td_rv_pga.tolist() == table.td_rv_pga.tolist()


def test_bt15_table_values() -> None:
    table = load_bt15_table("bt15_wna_acr")
    # The second row of the file: M 2.5, Rps 2 km
    c = table.coefficients[1, 0]
    assert c.tolist() == [9.0625e-01, -1.3842e-02, 2.0, 1.0, 1.6713e00, 2.0, 1.0667e00]
    assert table.td_rv_pga[1, 0] == 1.0500 and table.td_rv_pgv[1, 0] == 1.0355
    # At a grid point the interpolation returns the formula there
    eta, damping = 0.4, 0.05
    f = c[0] + c[1] * (1 - eta ** c[2]) / (1 + eta ** c[2])
    osc = 1 + c[3] / (2 * math.pi * damping) * (eta / (1 + c[4] * eta ** c[5])) ** c[6]
    assert table.rms_to_excitation(eta, damping, 2.5, 2.0) == pytest.approx(f * osc, rel=1e-12)
    rps = 10.0 ** table.log_distances[-1]
    assert table.td_rv_ratio(MotionQuantity.VELOCITY, 8.0, rps) == pytest.approx(
        table.td_rv_pgv[-1, -1]
    )
    assert table.td_rv_ratio(MotionQuantity.DISPLACEMENT, 20.0, 1.0) == 1.0
    # Outside the grid
    assert math.isnan(table.rms_to_excitation(eta, damping, 8.1, 20.0))
    assert math.isnan(table.rms_to_excitation(eta, damping, 6.0, 1.9))
    assert math.isnan(table.td_rv_ratio(MotionQuantity.ACCELERATION, 1.9, 20.0))


def test_bt15_interpolation_is_bilinear_in_log() -> None:
    table = load_bt15_table("bt15_ena_scr")
    m, rps = 6.25, 10.0 ** (0.5 * (table.log_distances[4] + table.log_distances[5]))
    corners = [
        table.rms_to_excitation(0.3, 0.05, mm, 10.0**lr)
        for mm in (6.0, 6.5)
        for lr in table.log_distances[4:6]
    ]
    assert table.rms_to_excitation(0.3, 0.05, m, rps) == pytest.approx(
        10.0 ** np.mean(np.log10(corners)), rel=1e-12
    )


def test_oscillator_correction_from_settings() -> None:
    expected = {
        Choice.NONE: NoCorrection,
        Choice.BJ84: BooreJoyner1984,
        Choice.LP99: LiuPezeshk1999,
        Choice.BT15_WNA: BooreThompson2015,
        Choice.BT15_ENA: BooreThompson2015,
        Choice.BT15_AVERAGE: BooreThompson2015Average,
    }
    for choice, cls in expected.items():
        assert type(oscillator_correction(RVConfig(osc_correction=choice))) is cls
    wna = oscillator_correction(RVConfig(osc_correction=Choice.BT15_WNA))
    ena = oscillator_correction(RVConfig(osc_correction=Choice.BT15_ENA))
    assert isinstance(wna, BooreThompson2015) and wna.table.name == "bt15_wna_acr"
    assert isinstance(ena, BooreThompson2015) and ena.table.name == "bt15_ena_scr"
    average = oscillator_correction(RVConfig(osc_correction=Choice.BT15_AVERAGE))
    kw = {"period": 1.0, "damping": 0.05, "magnitude": 6.0, "rps": 20.0, "bandwidth": 0.3}
    assert average.rms_duration(4.0, **kw) == pytest.approx(
        0.5 * (wna.rms_duration(4.0, **kw) + ena.rms_duration(4.0, **kw))
    )
    assert average.peak_motion_factor(MotionQuantity.VELOCITY, 6.0, 20.0) == pytest.approx(
        0.5
        * (
            wna.peak_motion_factor(MotionQuantity.VELOCITY, 6.0, 20.0)
            + ena.peak_motion_factor(MotionQuantity.VELOCITY, 6.0, 20.0)
        )
    )
    with pytest.raises(ValueError, match="unknown BT15 table"):
        oscillator_correction(RVConfig(bt15_tables=("nope", "bt15_ena_scr")))


# --- Peak factors ------------------------------------------------------------------


def test_numbers_of_crossings_and_extrema() -> None:
    assert zero_crossings(2.0, 10.0) == 40.0
    assert zero_crossings(0.05, 10.0) == 1.33
    assert extrema(3.0, 10.0) == 60.0
    assert extrema(0.05, 10.0) == 1.002


def test_asymptotic_peak_factors() -> None:
    first = math.sqrt(2.0 * math.log(100.0))
    assert cartwright_longuet_higgins_asymptotic(100.0, terms=1) == pytest.approx(first)
    assert cartwright_longuet_higgins_asymptotic(100.0) == pytest.approx(first + 0.5772 / first)
    with pytest.raises(ValueError):
        cartwright_longuet_higgins_asymptotic(100.0, terms=3)
    # Der Kiureghian's reduced number of zero crossings, in its three ranges
    for nz, q, n in [
        (10.0, 0.05, 2.1),
        (100.0, 0.08, 2 * 0.08 * 100.0),
        (100.0, 0.4, (1.63 * 0.4**0.45 - 0.38) * 100.0),
    ]:
        first = math.sqrt(2.0 * math.log(n))
        assert der_kiureghian_1980_asymptotic(nz, q) == pytest.approx(first + 0.5772 / first)
    assert der_kiureghian_1980_asymptotic(100.0, 0.8) == pytest.approx(
        cartwright_longuet_higgins_asymptotic(100.0)
    )
    # For an oscillator q is at least the damping
    assert der_kiureghian_1980_asymptotic(100.0, 0.01, damping=0.08) == pytest.approx(
        der_kiureghian_1980_asymptotic(100.0, 0.08)
    )
    assert math.isnan(der_kiureghian_1980_asymptotic(1.5, 0.3))


def test_cartwright_longuet_higgins_tends_to_asymptote() -> None:
    # Narrow band (xi = 1), many extrema
    n = 1.0e5
    assert cartwright_longuet_higgins(n, 1.0) == pytest.approx(
        cartwright_longuet_higgins_asymptotic(n), rel=2e-3
    )
    # Broad band: fewer effective peaks, smaller factor
    assert cartwright_longuet_higgins(n, 0.5) < cartwright_longuet_higgins(n, 1.0)


def test_der_kiureghian_distribution() -> None:
    n, q = 40.0, 0.5
    assert der_kiureghian_1980_cdf(0.0, n, q) == 0.0
    assert der_kiureghian_1980_cdf(8.0, n, q) == pytest.approx(1.0)
    a = np.linspace(0.5, 5.0, 10)
    h = 1e-6
    numeric = [
        (der_kiureghian_1980_cdf(x + h, n, q) - der_kiureghian_1980_cdf(x - h, n, q)) / (2 * h)
        for x in a
    ]
    np.testing.assert_allclose(der_kiureghian_1980_pdf(a, n, q), numeric, rtol=1e-6)
    mean, _ = quad(lambda x: x * float(der_kiureghian_1980_pdf(np.array([x]), n, q)[0]), 0, 12)
    assert der_kiureghian_1980(n, q) == pytest.approx(mean, rel=1e-9)


@pytest.mark.parametrize(
    ("n", "q", "pf", "steps"),
    # Values from the golden rv/ files (ENA M 4, Rjb 1 and 5 km, PGV)
    [(38.18, 0.68582, 2.85, 20), (44.50, 0.68487, 2.91, 80)],
)
def test_der_kiureghian_fortran_sum(n: float, q: float, pf: float, steps: int) -> None:
    value, na = der_kiureghian_1980_fortran(n, q, 1e-5)
    assert na == steps
    assert value == pytest.approx(pf, abs=5e-3)
    assert value == pytest.approx(der_kiureghian_1980(n, q), rel=2e-4)


def test_der_kiureghian_fortran_sum_truncates_at_six() -> None:
    # With very many zero crossings the mass beyond a = 6 matters
    value, _ = der_kiureghian_1980_fortran(1.0e5, 0.2, 1e-5)
    assert value < der_kiureghian_1980(1.0e5, 0.2) * (1 - 1e-3)


# --- One scenario --------------------------------------------------------------------


def test_rv_motion_oscillator() -> None:
    osc = Oscillator(period=1.0, damping=0.05)
    result = rv_motion(PARAMS, 6.0, 20.0, osc)
    assert result.dur_ex == pytest.approx(result.dur_source + result.dur_path)
    assert result.dur_source == pytest.approx(PARAMS.source.duration(result.terms.source))
    assert result.dur_path == pytest.approx(PARAMS.path_duration.duration(result.terms.rps))
    assert result.dur_rms > result.dur_ex
    assert result.rms == pytest.approx(math.sqrt(result.moments.m0 / result.dur_rms))
    assert result.peak_motion_factor == 1.0
    assert result.peak == pytest.approx(result.rms * result.peak_factor)
    assert result.psa == pytest.approx(2.0 * math.pi * result.peak)
    assert result.sd == pytest.approx(result.peak / (2.0 * math.pi))
    assert result.peak_factor_steps is None
    compat = rv_motion(PARAMS, 6.0, 20.0, osc, fortran_compat=True)
    assert compat.peak_factor_steps is not None
    assert compat.peak == pytest.approx(result.peak, rel=1e-3)


def test_rv_motion_ground_motion() -> None:
    acc = GroundMotion(MotionQuantity.ACCELERATION)
    result = rv_motion(PARAMS, 6.0, 20.0, acc)
    assert result.dur_rms == result.dur_ex
    correction = oscillator_correction(PARAMS.rv)
    assert result.peak_motion_factor == pytest.approx(
        correction.peak_motion_factor(MotionQuantity.ACCELERATION, 6.0, result.terms.rps)
    )
    assert result.peak == pytest.approx(result.peak_motion_factor * result.rms * result.peak_factor)
    with pytest.raises(TypeError):
        _ = result.psa
    with pytest.raises(TypeError):
        _ = result.sd
    # Without a BT15 table the peak is rms times the peak factor
    plain = replace(PARAMS, rv=RVConfig(osc_correction=Choice.NONE))
    assert rv_motion(plain, 6.0, 20.0, acc).peak_motion_factor == 1.0


def test_rv_motion_outside_bt15_table_is_nan() -> None:
    result = rv_motion(PARAMS, 8.5, 20.0, Oscillator(period=1.0))
    assert math.isnan(result.peak)
    assert math.isnan(rv_motion(PARAMS, 8.5, 20.0, GroundMotion(MotionQuantity.VELOCITY)).peak)


def test_rv_motion_rejects_instruments() -> None:
    with pytest.raises(TypeError, match="ground motion and oscillator"):
        rv_motion(PARAMS, 6.0, 20.0, WoodAnderson())  # type: ignore[arg-type]


# --- The loop ----------------------------------------------------------------------


GRID = ScenarioGrid(
    magnitudes=(5.0, 7.0),
    distances=(10.0, 50.0),
    periods=(0.1, 1.0),
    kappa0=(0.02, 0.04),
    stress=(50.0,),
)


def test_rv_loop_details_table() -> None:
    table = rv_loop_details(PARAMS, GRID)
    assert tuple(table.columns) == RV_DETAIL_COLUMNS
    assert len(table) == 2 * 4 * 2 * 2
    # Order: kappa_0, measure (PGV, PGA, periods), magnitude, distance
    assert table.measure.tolist()[:12] == ["pgv"] * 4 + ["pga"] * 4 + ["psa"] * 4
    assert table.kappa0.tolist() == [0.02] * 16 + [0.04] * 16
    assert table.magnitude.tolist()[:4] == [5.0, 5.0, 7.0, 7.0]
    assert table.rjb.tolist()[:4] == [10.0, 50.0, 10.0, 50.0]
    assert (table.stress == 50.0).all()
    np.testing.assert_allclose(table.rrup, np.hypot(table.rjb, table.h))
    psa = table[table.measure == "psa"]
    np.testing.assert_allclose(psa.psa, 2 * math.pi / psa.period * psa.psv)
    np.testing.assert_allclose(psa.sd, psa.period / (2 * math.pi) * psa.psv)
    assert psa.pga.isna().all() and psa.pgv.isna().all()
    pga = table[table.measure == "pga"]
    assert pga.psa.isna().all() and pga.period.isna().all()
    assert table.peak_factor_steps.isna().all()
    # One row recomputed on its own
    row = psa.iloc[0]
    params = params_for(PARAMS, kappa0=0.02, stress=50.0)
    single = rv_motion(params, row.magnitude, row.rrup, Oscillator(period=row.period))
    assert row.psv == pytest.approx(single.peak, rel=1e-12)


def test_rv_loop_one_row_per_scenario() -> None:
    details = rv_loop_details(PARAMS, GRID)
    table = rv_loop(PARAMS, GRID)
    assert tuple(table.columns) == (*SCENARIO_COLUMNS, "pgv", "pga", "psa(0.1)", "psa(1)")
    # Order: kappa_0, magnitude, distance
    assert len(table) == 2 * 2 * 2
    assert table.kappa0.tolist() == [0.02] * 4 + [0.04] * 4
    assert table.magnitude.tolist()[:4] == [5.0, 5.0, 7.0, 7.0]
    assert table.rjb.tolist()[:4] == [10.0, 50.0, 10.0, 50.0]
    for name in SCENARIO_COLUMNS:
        for measure in ("pgv", "pga", "psa"):
            rows = details[details.measure == measure]
            if measure == "psa":
                rows = rows[rows.period == 1.0]
            np.testing.assert_array_equal(table[name], rows[name], err_msg=name)
    np.testing.assert_array_equal(table.pgv, details.pgv[details.measure == "pgv"])
    np.testing.assert_array_equal(table.pga, details.pga[details.measure == "pga"])
    for period in GRID.periods:
        psa = details.psa[details.period == period]
        np.testing.assert_array_equal(table[psa_column(period)], psa)


def test_psa_column() -> None:
    assert psa_column(0.01) == "psa(0.01)"
    assert psa_column(10.0) == "psa(10)"
    assert psa_column(0.0123456789) == "psa(0.0123457)"


def test_rv_loop_without_pgv_and_pga() -> None:
    grid = ScenarioGrid(magnitudes=(6.0,), distances=(20.0,), periods=(0.2,), pgv=False, pga=False)
    table = rv_loop(PARAMS, grid)
    assert tuple(table.columns) == (*SCENARIO_COLUMNS, "psa(0.2)")
    assert len(table) == 1


def test_rv_loop_rrup_grid_and_options() -> None:
    grid = ScenarioGrid(
        magnitudes=(6.0,),
        distances=(20.0,),
        distance_metric=DistanceMetric.RRUP,
        periods=(0.2,),
        pga=False,
        pgv=False,
        damping=0.02,
    )
    table = rv_loop_details(PARAMS, grid, fortran_compat=True)
    assert len(table) == 1
    row = table.iloc[0]
    assert row.rrup == 20.0 and math.isnan(row.rjb) and math.isnan(row.h)
    assert row.damping == 0.02 and row.measure == "psa"
    assert row.peak_factor_steps >= 20
