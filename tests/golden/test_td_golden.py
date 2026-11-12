"""The time-domain method against the Fortran oracle.

Two kinds of comparison (tests/golden/README.md describes the files):

- ``td_series/``: one SMSIM simulation with its white noise (``td_dump``).
  Fed the same noise, pysmsim must give the same acceleration series, to
  1e-5 of its peak (the phase 3 gate), and the same peaks, Arias intensity,
  durations and oscillator responses.
- ``td/``: ``tmrsk_loop_td_drvr`` with 800 simulations of each scenario.
  SMSIM's random numbers cannot be reproduced, so the means are compared
  within a few standard errors.
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from oracle_files import GOLDEN, ORACLE_INPUTS, SHIPPED_MEDIA

from pysmsim.drivers.loop_td import td_loop_details
from pysmsim.io.legacy import read_params
from pysmsim.params import Params
from pysmsim.scenario import ScenarioGrid
from pysmsim.spectrum.fas import scenario_terms
from pysmsim.td.acc_ts import acc_ts, target_spectrum
from pysmsim.td.intensity import cumulative_squared, significant_duration, velocity_displacement
from pysmsim.td.response import oscillator_displacement, oscillator_displacement_numpy
from pysmsim.td.window import time_grid, window

pytestmark = pytest.mark.golden

SMSIM_FILES = Path(__file__).parents[1] / "data" / "smsim"
SERIES = sorted((GOLDEN / "td_series").glob("*.txt.gz"))
_COMMENT = re.compile(r"^ *!")


@dataclass(frozen=True)
class Dump:
    """A td_dump file: scalars from its header, periods and SD, the series."""

    head: dict[str, float]
    periods: npt.NDArray[np.float64]
    sd: npt.NDArray[np.float64]
    noise: npt.NDArray[np.float64]
    acc: npt.NDArray[np.float64]


def read_dump(path: Path) -> Dump:
    with gzip.open(path, "rt") as f:
        text = f.read().splitlines()
    head: dict[str, float] = {}
    osc = []
    body = []
    for line in text:
        if line.startswith("!"):
            if "=" not in line or "params =" in line:
                continue
            key, value = (part.strip() for part in line[1:].split("=", 1))
            if key == "period, rd":
                osc.append([float(v) for v in value.split()])
            else:
                head[key] = float(value)
        elif line.split()[0] != "noise":
            body.append(line)
    data = np.loadtxt(body)
    o = np.array(osc)
    return Dump(head, o[:, 0], o[:, 1], data[:, 0], data[:, 1])


def _td_case_params(name: str, tmp_path: Path) -> Path:
    """The params file of a case of oracle/inputs/td_cases.txt, as make_golden.sh writes it."""
    for line in (ORACLE_INPUTS / "td_cases.txt").read_text().splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        if line.split("|")[0].strip() == name:
            _, low_cut, win, timing = (part.strip() for part in line.split("|"))
            break
    else:  # pragma: no cover
        raise KeyError(name)
    lines = {30: low_cut, 35: win, 36: timing}
    out = []
    n = 0
    for line in (SMSIM_FILES / "ofr.params").read_text().splitlines():
        if not _COMMENT.match(line):
            n += 1
            line = f" {lines[n]}" if n in lines else line
        out.append(line)
    path = tmp_path / f"{name}.params"
    path.write_text("\n".join(out) + "\n")
    return path


def _params_for(path: Path, tmp_path: Path) -> Params:
    case = path.name.split(".M")[0]
    source = SMSIM_FILES / "ofr.params" if case == "ofr" else _td_case_params(case, tmp_path)
    return read_params(source)[0]


@pytest.mark.parametrize("path", SERIES, ids=lambda p: p.name.removesuffix(".txt.gz"))
def test_noise_injection(path: Path, tmp_path: Path) -> None:
    dump = read_dump(path)
    h = dump.head
    params = _params_for(path, tmp_path)
    m, rrup = h["m"], h["rrup"]
    terms = scenario_terms(params, m, rrup)
    dur_source = params.source.duration(terms.source)
    dur_path = params.path_duration.duration(terms.rps)
    assert dur_source == pytest.approx(h["dur_source"], rel=1e-9)
    assert dur_path == pytest.approx(h["dur_path"], rel=1e-9)

    grid = time_grid(params.td, params.filters.low_cut, dur_source, dur_path)
    assert (grid.nstart + 1, grid.nstop + 1) == (h["nstart"], h["nstop"])
    assert (grid.npts, grid.ntaper) == (h["npw2"], h["ntaper"])
    assert grid.tshift == h["tshift"]
    assert grid.te == pytest.approx(h["te"], rel=1e-12, abs=0.0)

    noise = dump.noise[grid.nstart : grid.nstop + 1]
    assert np.count_nonzero(dump.noise) == np.count_nonzero(noise)  # the noise is where expected
    acc = acc_ts(noise, grid, window(grid, params.td), target_spectrum(grid, params, m, rrup))
    peak = np.abs(dump.acc).max()
    # The gate: the same series to 1e-5 of its peak (they agree to about 1e-9)
    assert np.abs(acc - dump.acc).max() < 1e-5 * peak

    dt = grid.dt
    vel, dis = velocity_displacement(acc, dt)
    a2 = cumulative_squared(acc, dt)
    np.testing.assert_allclose(
        [np.abs(acc).max(), np.abs(vel).max(), np.abs(dis).max(), a2[-1]],
        [h["pga"], h["pgv"], h["pgd"], h["a_sq_int"]],
        rtol=1e-6,
    )
    assert significant_duration(a2, dt, 0.05, 0.75) == pytest.approx((h["j75"] - h["j05"]) * dt)
    assert significant_duration(a2, dt, 0.05, 0.95) == pytest.approx((h["j95"] - h["j05"]) * dt)
    sd = oscillator_displacement(acc, dump.periods, h["damping"], dt)
    np.testing.assert_allclose(sd, dump.sd, rtol=1e-6)
    np.testing.assert_allclose(
        oscillator_displacement_numpy(acc, dump.periods, h["damping"], dt), sd, rtol=1e-12
    )


# --- Statistics of 800 simulations --------------------------------------------------

# The grid of oracle/inputs/tmrsk_loop_td_drvr.ctl
TD_GRID = ScenarioGrid(
    magnitudes=(4.0, 6.0, 8.0),
    distances=(5.0, 20.0, 100.0),
    periods=(0.01, 0.1, 0.2, 1.0, 2.0),
    damping=0.05,
)

TD_COL_NAMES = (
    "damp T F M Rjb h Rrup F_FF Rps k0 numsource S psa psa_g psv sd fa fb arias "
    "dur_75_05 dur_95_05 psagm psagm_g psvgm sdgm ariasgm dur_75_05gm dur_95_05gm "
    "sd_am_sdev sd_gm_fctr log_psa log_psa_g dur_source dur_path dur_ex gmim_g gmim_cgs q"
).split()


def read_td_col(path: Path) -> pd.DataFrame:
    rows = [
        [float(v) for v in line.split()]
        for line in path.read_text().splitlines()
        if line.strip() and line.split()[0][:1].isdigit()
    ]
    return pd.DataFrame(rows, columns=TD_COL_NAMES)


def _z_scores(stem: str) -> pd.DataFrame:
    """Python minus SMSIM means, in standard errors of the difference."""
    params = read_params(SMSIM_FILES / f"{stem}.params")[0]
    golden = read_td_col(GOLDEN / "td" / f"{stem}.col")
    details = td_loop_details(params, TD_GRID)
    n = 800
    out = []
    for _, g in golden.iterrows():
        at = (details.magnitude == g.M) & (details.rjb == g.Rjb)
        if g["T"] > 0:
            checks = [("psa", g["T"], g.psa, g.psagm, g.sd_am_sdev * (2 * math.pi / g["T"]) ** 2)]
        elif g["T"] == 0:
            checks = [("pga", math.nan, g.psa, g.psagm, math.nan)]
            checks += [
                (m, math.nan, g[m], g[m + "gm"], math.nan)
                for m in ("arias", "dur_75_05", "dur_95_05")
            ]
        else:
            checks = [("pgv", math.nan, g.psv, g.psvgm, math.nan)]
        for measure, period, mean, gmean, std in checks:
            sel = at & (details.measure == measure)
            if measure == "psa":
                sel &= np.isclose(details.period, period)
            row = details[sel].iloc[0]
            # SMSIM's standard deviation when it gives one, else the Python one
            std_f = row["std"] if math.isnan(std) else std
            se = math.sqrt((row["std"] ** 2 + std_f**2) / n)
            gse = math.log10(row["gfactor"]) * math.sqrt(2.0 / n)
            out.append(
                {
                    "measure": measure,
                    "period": period,
                    "magnitude": g.M,
                    "rjb": g.Rjb,
                    "ratio": row["mean"] / mean,
                    "z": (row["mean"] - mean) / se,
                    "z_geometric": math.log10(row["gmean"] / gmean) / gse,
                }
            )
    return pd.DataFrame(out)


@pytest.mark.slow
@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="800 simulations need the compiled kernels")
@pytest.mark.parametrize("stem", sorted(SHIPPED_MEDIA))
def test_means_of_800_simulations(stem: str) -> None:
    z = _z_scores(stem)
    assert len(z) == 9 * (5 + 4 + 1)
    for column in ("z", "z_geometric"):
        values = z[column].abs()
        # Within 4 standard errors everywhere, and rarely beyond 3. The rows
        # are not independent: every scenario reuses the same noise, in
        # SMSIM as in pysmsim.
        assert values.max() < 4.0, z.loc[values.idxmax()].to_dict()
        assert (values > 3.0).mean() <= 0.03
    assert z.ratio.between(0.95, 1.05).all()
