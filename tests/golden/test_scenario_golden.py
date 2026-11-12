"""The single-scenario drivers against SMSIM's (golden ``scenario/``).

``gm_rv_drvr``, ``FAS_DRVR`` and ``gm_td_drvr`` were run for every shipped
params file (tests/golden/README.md lists the scenarios). Their output is
printed with 3 to 6 significant digits, which sets the tolerances here;
the computations behind them are tested more closely by the loop-driver
and factor tests.

- RV: the .col file (SD, PSV, PSA, PSA/PGA, the dominant frequency and the
  durations of each period) and the summary (PGD, PGV, PGA with their
  moments, the Arias intensity, and the moments of each period).
- FAS: the displacement, velocity and acceleration spectra and the
  oscillator spectra.
- TD: the averages of 800 simulations, within a few standard errors (SMSIM's
  random numbers cannot be reproduced), and the velocity and displacement
  of the saved simulation, integrated from its acceleration.
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from oracle_files import GOLDEN, SHIPPED_MEDIA

from pysmsim.drivers.single import fas_scenario, rv_scenario, td_scenario
from pysmsim.io.legacy import read_params
from pysmsim.params import Params
from pysmsim.td.intensity import velocity_displacement

pytestmark = pytest.mark.golden

SMSIM_FILES = Path(__file__).parents[1] / "data" / "smsim"
SCENARIO = GOLDEN / "scenario"
STEMS = sorted(SHIPPED_MEDIA)
CASES = ((4.0, 10.0), (6.0, 20.0), (7.5, 100.0))
PERIODS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FAS_PERIODS = (0.1, 0.3, 1.0, 3.0)
DAMPING = 0.05
_NUMBER = re.compile(r"^-?\d")


def _params(stem: str) -> Params:
    return read_params(SMSIM_FILES / f"{stem}.params")[0]


def _rows(lines: list[str]) -> list[list[float]]:
    """The lines that hold only numbers, as numbers."""
    rows = []
    for line in lines:
        parts = line.split()
        if parts and all(_NUMBER.match(p) for p in parts):
            rows.append([float(p) for p in parts])
    return rows


def _col(path: Path) -> pd.DataFrame:
    """A .col file: a title line, a header line, then rows of numbers."""
    lines = path.read_text().splitlines()
    names = lines[1].split()
    return pd.DataFrame(_rows(lines[2:]), columns=names)


def _close(actual: npt.ArrayLike, desired: npt.ArrayLike, digits: int, slack: float = 1.0) -> None:
    """Agreement to the printed digits (half a unit of the last), times ``slack``."""
    a = np.asarray(actual, dtype=np.float64)
    d = np.asarray(desired, dtype=np.float64)
    np.testing.assert_allclose(a, d, rtol=slack * 0.5 * 10.0 ** (1 - digits) * 1.01)


# --- Random vibration ---------------------------------------------------------------


def _rv_summary(path: Path) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """PGA, PGV, PGD rows; the Arias intensity; the rows of each period."""
    lines = path.read_text().splitlines()
    motions = []
    arias = math.nan
    for k, line in enumerate(lines):
        words = line.split()
        if words and words[0] in ("pga(cm/s2)", "pgv(cm/s)", "pgd(cm)"):
            values = [float(v) for v in lines[k + 1].split()]
            motions.append([words[0].split("(")[0], *values])
        elif words[:2] == ["Arias", "intensity"]:
            arias = float(lines[k + 1])
    columns = "measure peak domfreq eps nx nz pk_rms m0 m1 m2 m4".split()
    start = next(k for k, line in enumerate(lines) if line.split()[:2] == ["per(s)", "freq"])
    names = lines[start].split()
    periods = pd.DataFrame(_rows(lines[start + 1 :]), columns=names)
    return pd.DataFrame(motions, columns=columns).set_index("measure"), arias, periods


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize(("magnitude", "rrup"), CASES)
def test_rv_scenario(stem: str, magnitude: float, rrup: float) -> None:
    name = f"{stem}.M{magnitude}.R{rrup}"
    col = _col(SCENARIO / "rv" / f"{name}.col")
    motions, arias, periods = _rv_summary(SCENARIO / "rv" / f"{name}.sum")
    params = _params(stem)
    rv = rv_scenario(params, magnitude, rrup, PERIODS, DAMPING, fortran_compat=True)
    table = rv.table()

    # The scenario
    np.testing.assert_allclose(col.per, PERIODS)
    np.testing.assert_allclose(col.M, magnitude)
    np.testing.assert_allclose(col.R, rrup)
    np.testing.assert_allclose(col.F_FF, rv.terms.finite_fault_h, atol=0.006)
    np.testing.assert_allclose(col.Rmod, rv.terms.rps, atol=0.006)
    stress = rv.terms.source.stress
    np.testing.assert_allclose(col.stress, 0.0 if stress is None else stress, atol=0.006)

    # The response spectrum. SMSIM's moments are off by up to a few 1e-4
    # (see test_rv_golden.py), so the 4 printed digits get some slack.
    osc = table[table.measure == "psa"]
    psa = col.filter(like="psa:").iloc[:, 0]
    _close(osc.sd, col.filter(like="sd:").iloc[:, 0], 4, slack=6)
    _close(osc.psv, col.filter(like="psv:").iloc[:, 0], 4, slack=6)
    _close(osc.value, psa, 4, slack=6)
    _close(osc.psa_pga, col.filter(like="psa/pga:").iloc[:, 0], 4, slack=6)
    np.testing.assert_allclose(osc.dominant_frequency, col.domfrqpsv, rtol=1e-3, atol=6e-5)
    for ours, theirs in (("dur_source", "dursource"), ("dur_path", "durpath")):
        _close(osc[ours], col[theirs], 4)
    _close(osc.dur_ex, col.durex, 4)
    _close(osc.dur_rms, col.trms, 4)
    for m in ("m0", "m1", "m2", "m4"):
        np.testing.assert_allclose(osc[m], periods[f"amom{m[1]}"], rtol=3e-3)

    # PGD, PGV, PGA and the Arias intensity
    for measure in ("pgd", "pgv", "pga"):
        row = table[table.measure == measure].iloc[0]
        got = motions.loc[measure]
        _close(row.value, got.peak, 3, slack=2)
        np.testing.assert_allclose(row.dominant_frequency, got.domfreq, atol=0.006)
        np.testing.assert_allclose(row.epsilon, got.eps, atol=6e-5)
        np.testing.assert_allclose(row.n_extrema, got.nx, rtol=1e-3, atol=0.006)
        np.testing.assert_allclose(row.n_zero_crossings, got.nz, rtol=1e-3, atol=0.006)
        np.testing.assert_allclose(row.peak_factor, got.pk_rms, atol=0.006)
        for m in ("m0", "m1", "m2", "m4"):
            np.testing.assert_allclose(row[m], got[m], rtol=3e-3)
    _close(rv.arias, arias, 3, slack=2)
    assert table[table.measure == "arias"].value.iloc[0] == rv.arias


# --- Fourier spectra ----------------------------------------------------------------


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize(("magnitude", "rrup"), CASES)
def test_fas_scenario(stem: str, magnitude: float, rrup: float) -> None:
    col = _col(SCENARIO / "fas" / f"{stem}.M{magnitude}.R{rrup}.col")
    # FAS_DRVR's frequencies: freqlow * (freqhigh/freqlow)**((i-1)/(n-1))
    freqs = 0.01 * (100.0 / 0.01) ** (np.arange(41) / 40.0)
    np.testing.assert_allclose(col["Freq(Hz)"], freqs, rtol=5e-4)
    table = fas_scenario(_params(stem), magnitude, rrup, freqs, FAS_PERIODS, DAMPING)
    np.testing.assert_allclose(col.F_FF, table.f_ff, atol=0.006)
    np.testing.assert_allclose(col.Rps, table.rps, atol=0.006)

    def check(ours: pd.Series, theirs: pd.Series, digits: int) -> None:
        # SMSIM writes 0 below 1e-30
        printed = theirs.to_numpy() > 0
        assert (ours[~printed] < 1e-30).all()
        _close(ours[printed], theirs[printed], digits)

    check(table.fds, col.filter(like="fds:").iloc[:, 0], 6)
    check(table.fvs, col.filter(like="fvs:").iloc[:, 0], 6)
    check(table.fas, col.filter(like="fas:").iloc[:, 0], 6)
    for period in FAS_PERIODS:
        tag = f"T{period:07.3f}"
        for kind in ("sd", "sv", "sa"):
            check(table[f"osc_{kind}({period:g})"], col[f"FAS_OSC{kind[1].upper()}_{tag}"], 5)


# --- Time domain --------------------------------------------------------------------


def _td_summary(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Mean, relative standard error, geometric mean and its error factor."""
    lines = path.read_text().splitlines()
    out: dict[str, tuple[float, float, float, float]] = {}
    for k, line in enumerate(lines):
        words = line.split()
        if words[:1] == ["pgdam(cm)"]:
            am = [float(v) for v in lines[k + 1].split()]
            gm = [float(v) for v in lines[k + 3].split()]
            for j, measure in enumerate(("pgd", "pgv", "pga")):
                out[measure] = (am[2 * j], am[2 * j + 1], gm[2 * j], gm[2 * j + 1])
        elif words[:2] == ["Arias", "intensity(cm/s)"]:
            mean, sem = (float(v) for v in lines[k + 1].split())
            gmean, gsem = (float(v) for v in lines[k + 3].split())
            out["arias"] = (mean, sem, gmean, gsem)
    return out


def _td_z_scores(stem: str) -> pd.DataFrame:
    """Python minus SMSIM averages, in standard errors of the difference."""
    name = f"{stem}.M6.0.R20.0"
    col = _col(SCENARIO / "td" / f"{name}.col")
    summary = _td_summary(SCENARIO / "td" / f"{name}.sum")
    params = _params(stem)
    td = td_scenario(params, 6.0, 20.0, PERIODS, DAMPING)
    table = td.table().set_index(["measure", "period"])
    n = td.result.nsims
    assert n == 800

    rows = []
    expected = [(m, math.nan, *summary[m]) for m in ("pgd", "pgv", "pga", "arias")]
    psa, sem, gm, gsem = (
        col.filter(like=prefix).iloc[:, 0].to_numpy()
        for prefix in ("psa:", "sem/psv:", "psagm:", "sem_psvgm:")
    )
    for k, period in enumerate(col.per):
        expected.append(("psa", period, psa[k], sem[k], gm[k], gsem[k]))
    for measure, period, mean, rel_sem, gmean, gsem in expected:
        key = (measure, period) if measure == "psa" else (measure, table.loc[measure].index[0])
        row = table.loc[key]
        se = math.hypot(row["sem"], rel_sem * mean)
        z_geometric = math.nan
        if measure != "arias":  # SMSIM's sem_Ariasgm is not a factor (< 1)
            gse = math.hypot(math.log10(row["gsem"]), math.log10(gsem))
            z_geometric = math.log10(row["gmean"] / gmean) / gse
        rows.append(
            {
                "measure": measure,
                "period": period,
                "ratio": row["mean"] / mean,
                "z": (row["mean"] - mean) / se,
                "z_geometric": z_geometric,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="800 simulations need the compiled kernels")
@pytest.mark.parametrize("stem", STEMS)
def test_td_scenario_means(stem: str) -> None:
    z = _td_z_scores(stem)
    assert len(z) == 4 + len(PERIODS)
    for column in ("z", "z_geometric"):
        values = z[column].abs().dropna()
        # Within 4 standard errors (the rows share the same noise)
        assert values.max() < 4.0, z.loc[values.idxmax()].to_dict()
    assert z.ratio.between(0.95, 1.05).all()


def test_td_series_integration() -> None:
    """The saved simulation: our velocity and displacement from SMSIM's acceleration."""
    with gzip.open(SCENARIO / "td" / "ofr.M6.0.R20.0.avd.col.gz", "rt") as f:
        data = np.loadtxt(f, skiprows=1)
    t, acc, vel, dis = data.T
    params = _params("ofr")
    td = td_scenario(params, 6.0, 20.0, nsims=1, keep_series=True)
    series = td.series(0)
    assert len(series) == len(t) == td.result.grid.npts
    np.testing.assert_allclose(series.time, t, atol=5e-7)
    # The acceleration is printed with 5 digits; its rounding errors add up
    # in the integrals to far less than 1e-4 of the peaks.
    v, d = velocity_displacement(acc, td.result.grid.dt)
    assert np.abs(v - vel).max() < 1e-4 * np.abs(vel).max()
    assert np.abs(d - dis).max() < 1e-4 * np.abs(dis).max()
