"""The single-scenario drivers (pysmsim.drivers.single)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysmsim.drivers.loop_rv import rv_loop_details
from pysmsim.drivers.single import (
    FAS_SCENARIO_COLUMNS,
    RV_SCENARIO_COLUMNS,
    SERIES_COLUMNS,
    TD_SCENARIO_COLUMNS,
    fas_scenario,
    osc_columns,
    rv_scenario,
    td_scenario,
)
from pysmsim.io.config import load_config
from pysmsim.rv.gm_rv import rv_motion
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.fas import fourier_spectrum
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator
from pysmsim.td.gm_td import td_motion
from pysmsim.td.intensity import velocity_displacement

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
CONFIG = load_config(EXAMPLE)
# Few, short simulations so that the tests stay quick without the compiled kernels
PARAMS = dataclasses.replace(
    CONFIG.params, td=dataclasses.replace(CONFIG.params.td, nsims=5, dt=0.01, tshift=5.0)
)
PERIODS = (0.1, 1.0)


def test_rv_scenario() -> None:
    rv = rv_scenario(PARAMS, 6.0, 20.0, PERIODS, 0.03)
    acc = GroundMotion(MotionQuantity.ACCELERATION)
    assert rv.pga == rv_motion(PARAMS, 6.0, 20.0, acc)
    assert rv.pgd.peak == rv_motion(PARAMS, 6.0, 20.0, GroundMotion()).peak
    assert rv.spectrum[1] == rv_motion(PARAMS, 6.0, 20.0, Oscillator(period=1.0, damping=0.03))
    assert rv.arias == pytest.approx(math.pi / (2 * 981.0) * rv.pga.moments.m0, rel=1e-15)
    np.testing.assert_array_equal(rv.periods, PERIODS)
    np.testing.assert_allclose(rv.psa, [r.psa for r in rv.spectrum])

    table = rv.table()
    assert list(table.columns) == list(RV_SCENARIO_COLUMNS)
    assert list(table.measure) == ["pgd", "pgv", "pga", "arias", "psa", "psa"]
    assert (table.magnitude == 6.0).all() and (table.rrup == 20.0).all()
    np.testing.assert_allclose(
        table.value, [rv.pgd.peak, rv.pgv.peak, rv.pga.peak, rv.arias, *rv.psa]
    )
    osc = table[table.measure == "psa"]
    np.testing.assert_allclose(osc.psv, [r.peak for r in rv.spectrum])
    np.testing.assert_allclose(osc.sd, [r.sd for r in rv.spectrum])
    np.testing.assert_allclose(osc.psa_pga, rv.psa / rv.pga.peak)
    assert (osc.damping == 0.03).all()
    arias = table[table.measure == "arias"].iloc[0]
    assert math.isnan(arias.m0) and arias.stress == rv.terms.source.stress
    assert table.drop(columns="measure").dtypes.eq(np.float64).all()


def test_rv_scenario_matches_loop() -> None:
    grid = ScenarioGrid(
        magnitudes=(5.0,),
        distances=(30.0,),
        distance_metric=DistanceMetric.RRUP,
        periods=PERIODS,
    )
    loop = rv_loop_details(PARAMS, grid, fortran_compat=True)
    table = rv_scenario(PARAMS, 5.0, 30.0, PERIODS, fortran_compat=True).table()
    for measure in ("pgv", "pga", "psa"):
        ours = table[table.measure == measure]
        theirs = loop[loop.measure == measure]
        np.testing.assert_allclose(ours.value, theirs[measure], rtol=1e-14)
        np.testing.assert_allclose(ours.m4, theirs.m4, rtol=1e-14)


def test_rv_scenario_no_periods_and_errors() -> None:
    rv = rv_scenario(PARAMS, 5.0, 10.0)
    assert rv.spectrum == () and rv.periods.size == 0
    assert len(rv.table()) == 4
    with pytest.raises(ValueError, match="periods must be positive"):
        rv_scenario(PARAMS, 5.0, 10.0, (0.1, 0.0))


def test_td_scenario() -> None:
    td = td_scenario(PARAMS, 5.0, 20.0, PERIODS, keep_series=True)
    reference = td_motion(PARAMS, 5.0, 20.0, PERIODS)
    np.testing.assert_array_equal(td.result.pga, reference.pga)
    np.testing.assert_array_equal(td.result.sd, reference.sd)

    table = td.table()
    assert list(table.columns) == list(TD_SCENARIO_COLUMNS)
    assert list(table.measure) == [
        "pgd",
        "pgv",
        "pga",
        "arias",
        "dur_75_05",
        "dur_95_05",
        "psa",
        "psa",
    ]
    assert (table.nsims == 5).all() and (table.npts == td.result.grid.npts).all()
    pga = table[table.measure == "pga"].iloc[0]
    assert pga["mean"] == pytest.approx(reference.pga.mean())
    assert pga["std"] == pytest.approx(reference.pga.std(ddof=1))
    assert pga["sem"] == pytest.approx(pga["std"] / math.sqrt(5))
    assert pga.gmean == pytest.approx(math.exp(np.log(reference.pga).mean()))
    assert pga.gsem == pytest.approx(pga.gfactor ** (1 / math.sqrt(5)))
    psa = table[table.measure == "psa"]
    np.testing.assert_allclose(psa["mean"], reference.psa.mean(axis=0))
    assert (psa.damping == 0.05).all() and math.isnan(pga.damping)

    series = td.series(2)
    assert list(series.columns) == list(SERIES_COLUMNS)
    assert len(series) == td.result.grid.npts
    np.testing.assert_allclose(series.time, np.arange(len(series)) * td.result.grid.dt)
    assert td.result.acc is not None
    np.testing.assert_array_equal(series.acc, td.result.acc[2])
    vel, dis = velocity_displacement(td.result.acc[2], td.result.grid.dt)
    np.testing.assert_array_equal(series.vel, vel)
    np.testing.assert_array_equal(series.dis, dis)
    assert np.abs(series.vel).max() == pytest.approx(td.result.pgv[2])
    pd.testing.assert_frame_equal(td.series(-1), td.series(4))
    with pytest.raises(IndexError, match="out of range"):
        td.series(5)


def test_td_scenario_options() -> None:
    td = td_scenario(PARAMS, 5.0, 20.0, nsims=2)
    assert td.result.nsims == 2 and len(td.table()) == 6
    with pytest.raises(ValueError, match="keep_series=True"):
        td.series(0)


def test_fas_scenario() -> None:
    f = np.array([0.0, 0.1, 1.0, 10.0])
    table = fas_scenario(PARAMS, 6.0, 20.0, f, PERIODS, 0.05)
    assert list(table.columns[: len(FAS_SCENARIO_COLUMNS)]) == list(FAS_SCENARIO_COLUMNS)
    assert list(table.columns[len(FAS_SCENARIO_COLUMNS) :]) == [
        *osc_columns(0.1),
        *osc_columns(1.0),
    ]
    assert table.period.iloc[0] == np.inf
    np.testing.assert_allclose(table.period.iloc[1:], 1 / f[1:])
    fds = fourier_spectrum(f, PARAMS, 6.0, 20.0)
    np.testing.assert_allclose(table.fds, fds)
    w = 2 * np.pi * f
    np.testing.assert_allclose(table.fvs, w * fds)
    np.testing.assert_allclose(
        table.fas, fourier_spectrum(f, PARAMS, 6.0, 20.0, GroundMotion(MotionQuantity.ACCELERATION))
    )
    product = (
        table.source
        * table.gsprd
        * table.q_path
        * table.site_amp
        * table.site_dim
        * table.low_cut
        * table.high_cut
    )
    np.testing.assert_allclose(table.fds.iloc[1:], (product / product.iloc[2] * fds[2]).iloc[1:])
    osc = Oscillator(period=1.0, damping=0.05)
    psv = fourier_spectrum(f, PARAMS, 6.0, 20.0, osc)
    np.testing.assert_allclose(table["osc_sd(1)"], psv / (2 * np.pi))
    np.testing.assert_allclose(table["osc_sa(1)"], w * w * psv / (2 * np.pi))
    # At the oscillator's frequency, the relative velocity is the pseudo-velocity
    assert table["osc_sv(1)"].iloc[2] == pytest.approx(psv[2])


def test_fas_scenario_errors() -> None:
    with pytest.raises(ValueError, match="values >= 0"):
        fas_scenario(PARAMS, 6.0, 20.0, [1.0, -1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        fas_scenario(PARAMS, 6.0, 20.0, [[1.0]])
    with pytest.raises(ValueError, match="distinct"):
        fas_scenario(PARAMS, 6.0, 20.0, [1.0], (0.1, 0.1000000001))
    assert list(fas_scenario(PARAMS, 6.0, 20.0, [1.0]).columns) == list(FAS_SCENARIO_COLUMNS)
