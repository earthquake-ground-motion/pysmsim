"""The RV loop against SMSIM's tmrsk_loop_rv_drvr (golden rv/, rv_cases/, example/).

SMSIM integrates the spectral moments with Numerical Recipes' odeint, whose
error grows with the number of steps. Its moments are therefore off by up
to a few times 1e-4 (2.6e-3 in one case), while pysmsim's agree with
scipy.integrate.quad to about 1e-6. For narrow-band oscillator responses
the bandwidth sqrt(1 - m1**2/(m0*m2)) amplifies those errors, and a few
rows (5 of about 4,300 compared here) differ from SMSIM by more than 1e-3:
1.8e-3 at most for the shipped params files, 4.3e-3 for one synthetic case
with 2% damping. The tests allow those few rows.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from oracle_files import GOLDEN, read_rv_cases, shipped_params

from pysmsim.drivers.loop_rv import rv_loop, rv_loop_details
from pysmsim.params import Params
from pysmsim.scenario import ScenarioGrid

pytestmark = pytest.mark.golden

# The grid of oracle/inputs/tmrsk_loop_rv_drvr.ctl (PGV, PGA and 6 periods)
GRID = ScenarioGrid(
    magnitudes=(4.0, 5.0, 6.0, 7.0, 8.0),
    distances=(1.0, 5.0, 20.0, 50.0, 100.0, 300.0),
    periods=(0.01, 0.1, 0.3, 1.0, 3.0, 10.0),
    damping=0.05,
)

COL_NAMES = (
    "damp T F M Rjb h Rrup F_FF Rps k0 numsource S psa psa_g psv sd gmim_g gmim_cgs fa fb "
    "dur_source dur_path dur_ex dur_rms drms_dex t_dex dur_osc dur_osc_dex avib avib_bj84 "
    "avib_lp99 rms log_psa log_psa_g na anz m0 m1 m2 m4 delta pf_dk slope gsprd pf_cl q"
).split()

SENTINEL = -999.0


def read_col(path: Path) -> pd.DataFrame:
    """Read a .col file of tmrsk_loop_rv_drvr (full output)."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("!")]
    rows = [[float(v) for v in line.split()] for line in lines[2:] if line.strip()]
    return pd.DataFrame(rows, columns=COL_NAMES)


def peak(table: pd.DataFrame) -> npt.NDArray[np.float64]:
    """SMSIM's gmim column (cgs): PGV, PGA or PSA."""
    return np.asarray(
        np.where(
            table.measure == "psa",
            table.psa,
            np.where(table.measure == "pga", table.pga, table.pgv),
        ),
        dtype=np.float64,
    )


def check_against_col(
    params: Params, grid: ScenarioGrid, golden: pd.DataFrame, *, fortran_compat: bool
) -> npt.NDArray[np.float64]:
    """Compare a loop with a .col file; return the relative differences of the peaks."""
    table = rv_loop_details(params, grid, fortran_compat=fortran_compat)
    assert len(table) == len(golden)
    np.testing.assert_allclose(table.magnitude, golden.M, atol=5e-4)
    np.testing.assert_allclose(table.rjb, golden.Rjb, atol=5e-3)
    np.testing.assert_allclose(table.rps, golden.Rps, atol=5e-3)
    np.testing.assert_allclose(table.kappa0, golden.k0, atol=5e-5)

    # Where SMSIM writes sentinels (a BT15 table does not reach), pysmsim
    # gives NaN
    y = peak(table)
    computed = golden.gmim_cgs.to_numpy() > SENTINEL
    np.testing.assert_array_equal(np.isnan(y), ~computed)
    rows = table[computed]
    gold = golden[computed]

    # Intermediate values against their printed precision and SMSIM's
    # integration error
    for name, col, rtol, atol in [
        ("fa", "fa", 0.0, 5e-4),
        ("dur_source", "dur_source", 1e-3, 0.0),
        ("dur_path", "dur_path", 1e-3, 1e-3),
        ("dur_ex", "dur_ex", 1e-3, 0.0),
        ("dur_rms", "dur_rms", 1e-3, 0.0),
        ("m0", "m0", 3e-3, 0.0),
        ("m2", "m2", 2e-3, 0.0),
        ("m4", "m4", 2e-3, 0.0),
        ("n_zero_crossings", "anz", 2e-3, 5e-3),
        ("rms", "rms", 2e-3, 0.0),
        ("peak_factor", "pf_dk", 0.0, 0.015),
        ("peak_factor_cl", "pf_cl", 2e-3, 0.0),
    ]:
        np.testing.assert_allclose(rows[name], gold[col], rtol=rtol, atol=atol, err_msg=name)
    if fortran_compat:
        np.testing.assert_array_equal(rows.peak_factor_steps, gold.na)

    rel = np.abs(y[computed] / golden.gmim_cgs.to_numpy()[computed] - 1.0)
    return np.asarray(rel, dtype=np.float64)


@pytest.mark.parametrize("fortran_compat", [True, False], ids=["compat", "default"])
@pytest.mark.parametrize("path", sorted((GOLDEN / "rv").glob("*.col")), ids=lambda p: p.stem)
def test_rv_loop_shipped(path: Path, fortran_compat: bool) -> None:
    rel = check_against_col(
        shipped_params(path.stem), GRID, read_col(path), fortran_compat=fortran_compat
    )
    assert rel.max() < 2e-3
    assert np.count_nonzero(rel > 1e-3) <= 1
    assert np.median(rel) < 5e-5


RV_CASES = read_rv_cases()


@pytest.mark.parametrize("name", sorted(set(RV_CASES) - {"rv_quirk_kappa_mdep"}))
def test_rv_loop_cases(name: str) -> None:
    case = RV_CASES[name]
    golden = read_col(GOLDEN / "rv_cases" / f"{name}.col")
    rel = check_against_col(
        case.params, replace(GRID, damping=case.damping), golden, fortran_compat=True
    )
    assert rel.max() < 5e-3
    assert np.count_nonzero(rel > 1e-3) <= 2
    assert np.median(rel) < 5e-5


def test_rv_cases_cover_every_correction() -> None:
    corrections = {case.params.rv.osc_correction for case in RV_CASES.values()}
    assert len(corrections) == 6 - 1  # all but "none", which ofr.params uses


def test_quirk_loop_ignores_dkappa_dmag() -> None:
    # SMSIM's loop drivers set dkappadmag to 0; pysmsim keeps the slope
    # unless asked to behave like SMSIM
    case = RV_CASES["rv_quirk_kappa_mdep"]
    assert case.params.site.dkappa_dmag != 0
    golden = read_col(GOLDEN / "rv_cases" / "rv_quirk_kappa_mdep.col")
    grid = replace(GRID, damping=case.damping)
    rel = check_against_col(case.params, grid, golden, fortran_compat=True)
    assert rel.max() < 2e-3
    y = peak(rv_loop_details(case.params, grid))
    at_m8 = (golden.M == 8.0).to_numpy()
    assert not np.allclose(y[at_m8], golden.gmim_cgs[at_m8], rtol=0.05)


# --- The shipped example ---------------------------------------------------------

EXAMPLE_STEM = "wna.scf.bt15acr_fff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms"
# The grid of the example's control file: 200 log-spaced periods from 0.01
# to 100 s with PGV and PGA, M 4, 6 and 8, Rjb 10 km, kappa_0 0.02 and
# 0.05 s, and a stress of 100 bars
EXAMPLE_GRID = ScenarioGrid(
    magnitudes=(4.0, 6.0, 8.0),
    distances=(10.0,),
    periods=tuple(0.01 * 10.0 ** (i * math.log10(100.0 / 0.01) / 199) for i in range(200)),
    kappa0=(0.02, 0.05),
    stress=(100.0,),
)
EXAMPLE_NAMES = "damp T F M Rjb h Rrup F_FF Rps k0 numsource S gmim_g gmim_cgs".split()


def read_example(name: str) -> pd.DataFrame:
    path = GOLDEN / "example" / f"tmrsk_loop_rv_drvr.{name}.col"
    lines = [line for line in path.read_text().splitlines() if not line.startswith("!")]
    rows = [[float(v) for v in line.split()] for line in lines[2:] if line.strip()]
    return pd.DataFrame(rows, columns=EXAMPLE_NAMES)


@pytest.mark.parametrize("fortran_compat", [True, False], ids=["compat", "default"])
@pytest.mark.parametrize(("name", "tolerance"), [("double", 1e-3), ("shipped", 2e-3)])
def test_shipped_example(name: str, tolerance: float, fortran_compat: bool) -> None:
    # The phase 2 gate: PSA within 1e-3 of the double-precision build. The
    # shipped .col file was made with single precision and differs from
    # that build by up to 1.2e-3.
    golden = read_example(name)
    table = rv_loop_details(
        shipped_params(EXAMPLE_STEM), EXAMPLE_GRID, fortran_compat=fortran_compat
    )
    assert len(table) == len(golden) == 1212
    # SMSIM's loop order is SKMRT; pysmsim's is SKTMR, so sort both
    table["T"] = np.where(
        table.measure == "pgv", -1.0, np.where(table.measure == "pga", 0.0, table.period)
    )
    table["y"] = peak(table)
    ours = table.sort_values(["kappa0", "magnitude", "T"], kind="stable")
    theirs = golden.sort_values(["k0", "M", "T"], kind="stable")
    np.testing.assert_allclose(ours["T"], theirs["T"], rtol=1e-4)
    np.testing.assert_allclose(ours.y, theirs.gmim_cgs, rtol=tolerance)


def test_shipped_example_one_row_per_scenario() -> None:
    # The wide table: a row per kappa_0 and magnitude, PGV in cm/s, PGA and
    # PSA in cm/s^2
    golden = read_example("double")
    table = rv_loop(shipped_params(EXAMPLE_STEM), EXAMPLE_GRID)
    assert list(table.columns[-202:-200]) == ["pgv", "pga"]
    assert len(table) == 2 * 3
    for i, row in enumerate(table.itertuples(index=False)):
        rows = golden[(golden.k0 == row.kappa0) & (golden.M == row.magnitude)]
        assert len(rows) == 202
        y = table.iloc[i, -202:].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(y, rows.gmim_cgs, rtol=1e-3)
