"""The time-domain method: grid, window, noise, series, measures and loops."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pysmsim.drivers.loop_fas import params_for
from pysmsim.drivers.loop_td import (
    TD_DETAIL_COLUMNS,
    TD_SCENARIO_COLUMNS,
    td_loop,
    td_loop_details,
    td_loop_series,
    td_table,
    td_time_grids,
)
from pysmsim.drivers.run import TD_COLUMNS, run
from pysmsim.io.config import load_config
from pysmsim.io.hdf5 import read_hdf5, read_td_series, write_hdf5, write_td_series
from pysmsim.io.results import results_table, write_results
from pysmsim.params import NoiseDistribution, TDConfig, WindowShape
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.filters import LowCutFilter
from pysmsim.td.acc_ts import acc_ts, fft_frequencies, target_spectrum
from pysmsim.td.gm_td import MEASURES, Statistics, statistics, td_motion
from pysmsim.td.intensity import (
    SERIES_MEASURES,
    G,
    arias_intensity,
    cumulative_squared,
    integrate,
    series_measures,
    series_measures_numpy,
    significant_duration,
    velocity_displacement,
)
from pysmsim.td.noise import white_noise
from pysmsim.td.response import oscillator_displacement, oscillator_displacement_numpy
from pysmsim.td.window import exponential_window_coefficients, time_grid, window

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
CONFIG = load_config(EXAMPLE)
# Few, short simulations so that the tests stay quick without the compiled kernels
PARAMS = dataclasses.replace(
    CONFIG.params, td=dataclasses.replace(CONFIG.params.td, nsims=6, dt=0.01, tshift=5.0)
)
NO_CUT = LowCutFilter()
EXP = TDConfig(dt=0.01, tshift=2.0)
BOX = TDConfig(window=WindowShape.BOX, taper=0.1, dt=0.01, tshift=2.0)
GRID = ScenarioGrid(
    magnitudes=(4.0, 5.0),
    distances=(10.0, 40.0),
    periods=(0.1, 1.0),
    stress=(50.0, 100.0),
    pgv=False,
)


# --- Time grid and windows ----------------------------------------------------------


def test_time_grid_exponential() -> None:
    td = TDConfig(f_tb2te=2.0, f_te_xtnd=1.5, dt=0.01, tshift=3.0, dur_fctr=1.3)
    grid = time_grid(td, NO_CUT, 4.0, 1.0)
    assert grid.dur_ex == 5.0
    assert grid.te == pytest.approx(10.0)
    assert grid.tw == pytest.approx(15.0)
    assert grid.nstart == 300
    assert grid.nmotion == int(15.0 / 0.01) + 1
    assert grid.ntaper == 0
    assert grid.npts == 4096  # (301 + 1.3 * 1500) * dt + 0 pad = 22.52 s, 2253 points
    assert grid.tpad == 0.0 and grid.tshift == 3.0
    t = grid.times()
    assert t.shape == (grid.npts,) and t[0] == 0.0
    assert t[1] == pytest.approx(0.01) and t[-1] == pytest.approx((grid.npts - 1) * 0.01)
    assert t[grid.nstart] == pytest.approx(grid.tshift)


def test_time_grid_box_and_low_cut_pad() -> None:
    grid = time_grid(BOX, LowCutFilter(fcut=0.1, order=4), 3.0, 1.0)
    assert grid.ntaper == int(0.1 * 400)
    assert grid.nmotion == 400 + 40 + 1
    assert grid.te == grid.tw == 0.0
    # A pad of 1.5 * (4 // 2) / 0.1 = 30 s, longer than tshift
    assert grid.tpad == pytest.approx(30.0)
    assert grid.tshift == grid.tpad
    assert grid.nstart == int(30.0 / 0.01)
    assert grid.npts & (grid.npts - 1) == 0
    assert grid.npts * grid.dt >= (grid.nstart + 1 + 1.3 * 440) * grid.dt + 30.0


@pytest.mark.parametrize("td", [EXP, BOX], ids=["exponential", "box"])
def test_time_grid_too_short(td: TDConfig) -> None:
    with pytest.raises(ValueError, match="shorter than dt"):
        time_grid(td, NO_CUT, 0.001, 0.001)


def test_exponential_window_coefficients() -> None:
    eps, eta = 0.2, 0.05
    a, b, c = exponential_window_coefficients(eps, eta)
    w = lambda x: a * x**b * math.exp(-c * x)  # noqa: E731
    assert w(eps) == pytest.approx(1.0)
    assert w(1.0) == pytest.approx(eta)
    assert w(eps * 0.99) < 1.0 and w(eps * 1.01) < 1.0


def test_window_exponential() -> None:
    grid = time_grid(EXP, NO_CUT, 3.0, 1.0)
    w = window(grid, EXP)
    assert w.shape == (grid.npts,)
    assert not w[: grid.nstart].any() and not w[grid.nstop + 1 :].any()
    assert w[grid.nstart] == 0.0
    peak = int(np.argmax(w))
    assert w[peak] == pytest.approx(1.0, abs=1e-4)
    assert (peak - grid.nstart) * grid.dt == pytest.approx(EXP.eps_w * grid.te, abs=grid.dt)
    assert w[grid.nstart + round(grid.te / grid.dt)] == pytest.approx(EXP.eta_w, rel=1e-3)


def test_window_box() -> None:
    grid = time_grid(BOX, NO_CUT, 3.0, 1.0)
    w = window(grid, BOX)[grid.nstart : grid.nstop + 1]
    ntaper = grid.ntaper
    np.testing.assert_allclose(w, w[::-1], atol=1e-12)
    assert np.all(w[ntaper:-ntaper] == 1.0)
    assert np.all(np.diff(w[: ntaper + 1]) >= 0) and w[0] < 0.01
    no_taper = dataclasses.replace(BOX, taper=0.0)
    grid = time_grid(no_taper, NO_CUT, 3.0, 1.0)
    w = window(grid, no_taper)
    assert grid.ntaper == 0
    assert np.all(w[grid.nstart : grid.nstop + 1] == 1.0)
    assert w.sum() == grid.nmotion


# --- Noise and one series -----------------------------------------------------------


def test_white_noise() -> None:
    gauss = white_noise(np.random.default_rng(1), (3, 20000), NoiseDistribution.GAUSSIAN)
    assert gauss.shape == (3, 20000)
    assert abs(gauss.mean()) < 0.02 and gauss.std() == pytest.approx(1.0, abs=0.02)
    uniform = white_noise(np.random.default_rng(1), 20000, NoiseDistribution.UNIFORM)
    assert uniform.min() >= -0.5 and uniform.max() < 0.5
    again = white_noise(np.random.default_rng(1), (3, 20000), NoiseDistribution.GAUSSIAN)
    np.testing.assert_array_equal(gauss, again)


def test_acc_ts() -> None:
    grid = time_grid(EXP, NO_CUT, 3.0, 1.0)
    w = window(grid, EXP)
    spectrum = target_spectrum(grid, PARAMS, 5.0, 20.0)
    f = fft_frequencies(grid)
    assert f.size == grid.npts // 2 + 1 and f[-1] == pytest.approx(0.5 / grid.dt)
    assert spectrum[0] == 0.0 and np.all(spectrum[1:] > 0)

    noise = np.random.default_rng(3).standard_normal((4, grid.nmotion))
    acc = acc_ts(noise, grid, w, spectrum)
    assert acc.shape == (4, grid.npts)
    peak = np.abs(acc[2]).max()
    np.testing.assert_allclose(acc[2], acc_ts(noise[2], grid, w, spectrum), atol=1e-12 * peak)
    assert acc.mean(axis=-1) == pytest.approx(0.0, abs=1e-9)  # no 0 Hz

    # With a flat spectrum, the shaped spectrum has unit mean square
    flat = np.ones_like(spectrum)
    x = np.fft.rfft(acc_ts(noise[0], grid, w, flat)) * grid.dt
    assert np.mean(np.abs(x[1 : grid.npts // 2]) ** 2) == pytest.approx(1.0)
    # The noise scale does not matter
    scaled = acc_ts(5.0 * noise[0], grid, w, spectrum)
    np.testing.assert_allclose(scaled, acc[0], atol=1e-12 * np.abs(acc[0]).max())

    with pytest.raises(ValueError, match="noise must have"):
        acc_ts(noise[:, 1:], grid, w, spectrum)


# --- Measures of a series -----------------------------------------------------------


def test_integrals() -> None:
    dt = 0.01
    t = np.arange(1001) * dt
    acc = np.full(t.size, 2.0)
    vel, dis = velocity_displacement(acc, dt)
    np.testing.assert_allclose(vel, 2.0 * t, atol=1e-12)
    np.testing.assert_allclose(dis, t**2, atol=1e-9)
    np.testing.assert_allclose(integrate(np.vstack([t, t]), dt)[1], t**2 / 2, atol=1e-12)
    a2 = cumulative_squared(acc, dt)
    np.testing.assert_allclose(a2, 4.0 * t, atol=1e-12)
    # Exact for a linear acceleration: the integral of t**2 is t**3/3
    np.testing.assert_allclose(cumulative_squared(t, dt), t**3 / 3, atol=1e-9)
    assert arias_intensity(a2[-1]) == pytest.approx(math.pi / (2 * G) * 40.0)
    assert G == 981.0


def test_significant_duration() -> None:
    dt = 0.01
    cum = np.arange(1001) * dt  # a uniform build-up over 10 s
    assert significant_duration(cum, dt, 0.05, 0.75) == pytest.approx(7.0)
    assert significant_duration(cum, dt, 0.05, 0.95) == pytest.approx(9.0)
    two = significant_duration(np.vstack([cum, cum**2]), dt, 0.05, 0.95)
    assert two.shape == (2,) and two[0] == pytest.approx(9.0)


def test_series_measures_twins() -> None:
    rng = np.random.default_rng(3)
    acc = rng.standard_normal((2, 3, 500)) * np.hanning(500)
    ours = series_measures(acc, 0.01)
    theirs = series_measures_numpy(acc, 0.01)
    assert list(ours) == list(SERIES_MEASURES) == list(MEASURES)
    for name in SERIES_MEASURES:
        assert ours[name].shape == (2, 3)
        np.testing.assert_array_equal(ours[name], theirs[name])
    one = series_measures(acc[1, 2], 0.01)
    assert one["pga"].shape == () and one["arias"] == ours["arias"][1, 2]
    vel, dis = velocity_displacement(acc[0, 0], 0.01)
    assert ours["pgv"][0, 0] == np.abs(vel).max() and ours["pgd"][0, 0] == np.abs(dis).max()
    with pytest.raises(ValueError, match="at least two points"):
        series_measures(np.ones(1), 0.01)


def test_oscillator_step_response() -> None:
    # A constant acceleration from t = 0: the peak displacement is
    # (1 + exp(-pi * zeta / sqrt(1 - zeta**2))) * a / omega**2, exactly
    # what the recurrence gives at the samples near the peak
    dt, period, zeta, a = 0.001, 0.5, 0.05, 100.0
    omega = 2.0 * math.pi / period
    acc = np.full(3000, a)
    sd = oscillator_displacement(acc, [period], zeta, dt)
    exact = (1.0 + math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2))) * a / omega**2
    assert sd.shape == (1,)
    assert sd[0] == pytest.approx(exact, rel=1e-4)


def test_oscillator_twins() -> None:
    acc = np.random.default_rng(5).standard_normal((3, 800))
    periods = [0.05, 0.3, 2.0]
    sd = oscillator_displacement(acc, periods, 0.05, 0.01)
    assert sd.shape == (3, 3)
    np.testing.assert_allclose(
        oscillator_displacement_numpy(acc, periods, 0.05, 0.01), sd, rtol=1e-12
    )
    np.testing.assert_allclose(oscillator_displacement(acc[1], periods, 0.05, 0.01), sd[1])
    np.testing.assert_allclose(oscillator_displacement_numpy(acc[1], periods, 0.05, 0.01), sd[1])


def test_statistics() -> None:
    s = statistics([1.0, 10.0, 100.0])
    assert s.mean == pytest.approx(37.0)
    assert s.std == pytest.approx(np.std([1.0, 10.0, 100.0], ddof=1))
    assert s.gmean == pytest.approx(10.0)
    assert s.gfactor == pytest.approx(10.0)
    one = statistics([4.0])
    assert one == Statistics(mean=4.0, std=0.0, gmean=pytest.approx(4.0), gfactor=1.0)


# --- Simulations of one scenario ----------------------------------------------------


def test_td_motion() -> None:
    periods = [0.1, 1.0]
    result = td_motion(PARAMS, 5.0, 20.0, periods, keep_series=True)
    n = PARAMS.td.nsims
    assert result.nsims == n
    assert result.sd.shape == (n, 2)
    assert result.acc is not None and result.acc.shape == (n, result.grid.npts)
    np.testing.assert_allclose(result.pga, np.abs(result.acc).max(axis=1))
    omega = 2 * np.pi / np.array(periods)
    np.testing.assert_allclose(result.psv, result.sd * omega)
    np.testing.assert_allclose(result.psa, result.sd * omega**2)
    assert np.all(result.dur_95_05 >= result.dur_75_05)
    assert np.all(result.dur_95_05 <= result.grid.nmotion * result.grid.dt)
    for measure in MEASURES:
        stats = result.statistics(measure)
        assert isinstance(stats, Statistics)
        assert stats.mean == pytest.approx(getattr(result, measure).mean())
    psa = result.statistics("psa")
    assert isinstance(psa, list) and len(psa) == 2
    with pytest.raises(ValueError, match="unknown measure"):
        result.statistics("pgx")

    # The seed of the parameters: the same simulations every time
    again = td_motion(PARAMS, 5.0, 20.0, periods)
    np.testing.assert_array_equal(again.sd, result.sd)
    assert again.acc is None
    other = td_motion(PARAMS, 5.0, 20.0, periods, rng=np.random.default_rng(99))
    assert not np.array_equal(other.pga, result.pga)
    # Fewer simulations are the first ones of a longer run, also when chunked
    few = td_motion(PARAMS, 5.0, 20.0, periods, nsims=2)
    np.testing.assert_array_equal(few.pga, result.pga[:2])


def test_td_motion_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = td_motion(PARAMS, 5.0, 20.0, [0.2])
    monkeypatch.setattr("pysmsim.td.gm_td._CHUNK_POINTS", 1)
    chunked = td_motion(PARAMS, 5.0, 20.0, [0.2])
    np.testing.assert_allclose(chunked.sd, whole.sd, rtol=1e-12)
    np.testing.assert_allclose(chunked.arias, whole.arias, rtol=1e-12)


def test_td_motion_no_periods_and_errors() -> None:
    result = td_motion(PARAMS, 4.0, 10.0, nsims=1)
    assert result.sd.shape == (1, 0) and result.statistics("psa") == []
    assert result.pga.shape == (1,)
    with pytest.raises(ValueError, match="nsims"):
        td_motion(PARAMS, 4.0, 10.0, nsims=0)
    with pytest.raises(ValueError, match="periods"):
        td_motion(PARAMS, 4.0, 10.0, [0.0])


def test_td_motion_scales_with_magnitude() -> None:
    small = td_motion(PARAMS, 4.0, 20.0, [1.0])
    large = td_motion(PARAMS, 6.0, 20.0, [1.0])
    assert large.pga.mean() > 3 * small.pga.mean()
    assert large.psa.mean() > 10 * small.psa.mean()
    assert large.dur_95_05.mean() > small.dur_95_05.mean()


# --- Loops over scenarios -----------------------------------------------------------


def test_td_loop() -> None:
    details = td_loop_details(PARAMS, GRID)
    assert list(details.columns) == list(TD_DETAIL_COLUMNS)
    per_scenario = 6 + len(GRID.periods)
    assert len(details) == 2 * 2 * 2 * per_scenario
    first = details.iloc[:per_scenario]
    expected = ["pgv", "pga", "pgd", "arias", "dur_75_05", "dur_95_05", "psa", "psa"]
    assert list(first.measure) == expected
    assert first.period.iloc[-1] == 1.0 and first.damping.iloc[-1] == GRID.damping
    assert math.isnan(first.damping.iloc[0])
    assert list(details.stress.iloc[::per_scenario]) == [50.0] * 4 + [100.0] * 4
    assert list(details.magnitude.iloc[::per_scenario]) == [4.0, 4.0, 5.0, 5.0] * 2

    # Each scenario is td_motion
    k = 6 * per_scenario  # stress 100, M 5, Rjb 10
    params = params_for(PARAMS, kappa0=None, stress=100.0)
    one = td_motion(params, 5.0, float(details.rrup.iloc[k]), [0.1, 1.0])
    row = details.iloc[k + 1]
    assert row.measure == "pga" and row["mean"] == pytest.approx(one.pga.mean())

    table = td_table(details, GRID)
    assert list(table.columns) == [
        *TD_SCENARIO_COLUMNS,
        "pga",
        "pgd",
        "arias",
        "dur_75_05",
        "dur_95_05",
        "psa(0.1)",
        "psa(1)",
    ]
    assert len(table) == 8 and table.nsims.dtype == np.int64
    assert table["psa(1)"].iloc[3] == pytest.approx(details["mean"].iloc[4 * per_scenario - 1])
    pd.testing.assert_frame_equal(td_loop(PARAMS, GRID), table)


def test_td_loop_rrup_and_kappa() -> None:
    grid = ScenarioGrid(
        magnitudes=(4.5,),
        distances=(15.0,),
        periods=(),
        distance_metric=DistanceMetric.RRUP,
        kappa0=(0.01, 0.04),
    )
    table = td_loop(PARAMS, grid)
    assert list(table.rrup) == [15.0, 15.0]
    assert table.rjb.isna().all() and table.h.isna().all()
    assert list(table.kappa0) == [0.01, 0.04]
    assert table.pga.iloc[0] > table.pga.iloc[1]  # less kappa, more high frequencies
    assert "pgv" in table and not any(c.startswith("psa") for c in table.columns)


# --- In a run, and its files --------------------------------------------------------


def test_run_with_td(tmp_path: Path) -> None:
    grid = dataclasses.replace(GRID, stress=None, frequencies=(1.0,))
    result = run(PARAMS, grid, td=True)
    assert result.td is not None and result.td_details is not None
    assert list(result.td.columns[: len(TD_COLUMNS)]) == list(TD_COLUMNS)
    assert "pgv" not in result.td and "psa(1)" in result.td
    assert len(result.td) == len(result.scenarios) == 4
    # The same scenarios as the RV and FAS tables
    np.testing.assert_allclose(result.td.dur_ex, result.rv.dur_ex)

    table = results_table(result)
    assert {"td_pga", "td_psa(1)", "td_nsims", "pga", "fas(1)"} <= set(table.columns)
    assert table.columns.get_loc("td_pga") < table.columns.get_loc("fas(1)")

    write_hdf5(result, tmp_path / "td.h5")
    back = read_hdf5(tmp_path / "td.h5")
    assert back.td is not None and back.td_details is not None
    pd.testing.assert_frame_equal(back.td, result.td)
    pd.testing.assert_frame_equal(back.td_details, result.td_details)

    write_results(result, tmp_path / "td.csv", "csv")
    csv = pd.read_csv(tmp_path / "td.csv")
    np.testing.assert_allclose(csv["td_psa(0.1)"], result.td["psa(0.1)"])

    no_td = run(PARAMS, grid)
    assert no_td.td is None and no_td.td_details is None
    write_hdf5(no_td, tmp_path / "rv.h5")
    assert read_hdf5(tmp_path / "rv.h5").td is None

    only_td = run(PARAMS, dataclasses.replace(grid, frequencies=()), rv=False, td=True)
    assert only_td.rv is None and only_td.fas is None
    assert list(results_table(only_td).columns[:2]) == ["magnitude", "rjb"]


def test_td_series(tmp_path: Path) -> None:
    grid = dataclasses.replace(GRID, stress=None)
    grids = td_time_grids(PARAMS, grid)
    series = list(td_loop_series(PARAMS, grid, nsims=2))
    assert len(series) == len(grids) == 4
    assert [s.grid for s in series] == grids
    for s in series:
        assert s.acc is not None and s.acc.shape == (2, s.grid.npts)
    # The first simulations of td_loop_details
    details = td_loop_details(PARAMS, grid)
    full = td_motion(PARAMS, 4.0, float(details.rrup.iloc[0]), [0.1, 1.0])
    np.testing.assert_array_equal(series[0].sd, full.sd[:2])

    path = tmp_path / "series.h5"
    write_hdf5(run(PARAMS, grid, td=True), path)
    options = {"tolerance": 1e-4, "dtype": np.float64}
    write_td_series(path, PARAMS, grid, nsims=2, **options)
    write_td_series(path, PARAMS, grid, nsims=2, **options)  # replaces the group
    back = read_td_series(path)
    npts = np.array([g.npts for g in grids])
    assert back.acceleration.shape == (2, npts.max(), 4)
    np.testing.assert_allclose(back.times, grids[int(np.argmax(npts))].times())
    assert back.dt == PARAMS.td.dt and back.tolerance == 1e-4
    assert np.array_equal(back.nstart, [g.nstart for g in grids])
    assert np.array_equal(back.nstop, [g.nstop for g in grids])
    assert np.array_equal(back.npts, npts)
    for k, s in enumerate(series):
        assert s.acc is not None
        lo, hi = back.first[k], back.last[k]
        # The stored part covers the window and more, and holds the series
        assert lo < back.nstart[k] and hi > back.nstop[k] and hi < npts[k]
        np.testing.assert_array_equal(back.acceleration[:, lo : hi + 1, k], s.acc[:, lo : hi + 1])
        assert np.isnan(back.acceleration[:, :lo, k]).all()
        assert np.isnan(back.acceleration[:, hi + 1 :, k]).all()
        # What is left out is below 1e-4 of the peak
        peak = np.abs(s.acc).max(axis=1, keepdims=True)
        outside = np.concatenate([s.acc[:, :lo], s.acc[:, hi + 1 :]], axis=1)
        assert np.all(np.abs(outside) <= 1e-4 * peak)
    assert read_hdf5(path).td is not None  # the statistics are still read
    one = read_td_series(path, scenarios=[2])
    assert one.acceleration.shape == (2, npts.max(), 1)
    np.testing.assert_array_equal(one.acceleration[..., 0], back.acceleration[..., 2])
    assert one.first[0] == back.first[2]

    # A new file with the defaults: the whole series, in float32
    whole = tmp_path / "whole.h5"
    write_td_series(whole, PARAMS, grid, compression=None)
    with h5py.File(whole) as h5:
        assert h5["td/series/acceleration"].dtype == np.float32
    full = read_td_series(whole)
    assert full.tolerance == 0.0 and full.acceleration.dtype == np.float64
    assert full.acceleration.shape[0] == PARAMS.td.nsims
    assert np.array_equal(full.first, [0] * 4) and np.array_equal(full.last, npts - 1)
    np.testing.assert_allclose(full.acceleration[:2, :, 0][:, : npts[0]], series[0].acc, rtol=1e-6)
    with pytest.raises(ValueError, match="nsims"):
        write_td_series(whole, PARAMS, grid, nsims=0)
    with pytest.raises(ValueError, match="tolerance"):
        write_td_series(whole, PARAMS, grid, tolerance=1.0)
    with pytest.raises(ValueError, match="dtype"):
        write_td_series(whole, PARAMS, grid, dtype=np.int32)
    with pytest.raises(KeyError, match="no time-domain series"):
        read_td_series(_rv_only(tmp_path, grid))


def _rv_only(tmp_path: Path, grid: ScenarioGrid) -> Path:
    path = tmp_path / "rv.h5"
    write_hdf5(run(PARAMS, grid), path)
    return path
