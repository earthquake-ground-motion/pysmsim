"""The FAS loop driver against SMSIM's fmrsk_loop_fas_drvr (golden fas/ files)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from oracle_files import GOLDEN, shipped_params

from pysmsim.drivers.loop_fas import fas_column, fas_loop, fas_loop_details
from pysmsim.scenario import DistanceMetric, RjbToRrup, ScenarioGrid
from pysmsim.spectrum.attenuation import Stafford2022Q

pytestmark = pytest.mark.golden

# The grid of oracle/inputs/fmrsk_loop_fas_drvr.ctl
GRID = ScenarioGrid(
    magnitudes=(4.0, 5.0, 6.0, 7.0, 8.0),
    distances=(1.0, 5.0, 20.0, 50.0, 100.0, 300.0),
    distance_metric=DistanceMetric.RJB,
    rjb_to_rrup=RjbToRrup.CY14_ZTOR,
)
FREQUENCIES = 0.01 * 10.0 ** (np.arange(21) * (math.log10(100.0 / 0.01) / 20))

COL_NAMES = (
    "f period magnitude rjb h rrup f_ff rps kappa0 numsource stress fds fas fas_g gsprd q gsprd_q"
).split()


def read_col(path: Path) -> pd.DataFrame:
    """Read a .col file of fmrsk_loop_fas_drvr."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("!")]
    rows = [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
    return pd.DataFrame(rows, columns=COL_NAMES)


@pytest.mark.parametrize("path", sorted((GOLDEN / "fas").glob("*.col")), ids=lambda p: p.stem)
def test_fas_loop(path: Path) -> None:
    params = shipped_params(path.stem)
    golden = read_col(path)
    table = fas_loop_details(params, GRID, FREQUENCIES)
    assert len(table) == len(golden)

    # Each value against the precision it is printed with
    for name, rel, absolute in [
        ("f", 1e-4, 0.0),
        ("magnitude", 0.0, 5e-4),
        ("rjb", 0.0, 5e-4),
        ("h", 0.0, 5e-3),
        ("rrup", 0.0, 5e-4),
        ("f_ff", 0.0, 5e-3),
        ("rps", 0.0, 5e-4),
        ("kappa0", 0.0, 5e-5),
        ("stress", 1e-5, 0.0),
        ("fds", 1e-7, 0.0),
        ("fas", 1e-7, 0.0),
        ("gsprd", 1e-4, 0.0),
        ("q", 1e-4, 0.0),
    ]:
        np.testing.assert_allclose(table[name], golden[name], rtol=rel, atol=absolute, err_msg=name)
    np.testing.assert_allclose(table["period"], 1.0 / golden["f"], rtol=1e-4)

    # SMSIM's gsprd*Qop column uses Rps for every Q model, though its spectrum
    # uses Rrup for the Stafford et al. (2022) Q
    if isinstance(params.q, Stafford2022Q):
        expected = table["gsprd"] * np.exp(
            -math.pi * table["f"] * table["rps"] / (params.q.c_q * table["q"])
        )
        assert not np.allclose(table["gsprd_q"], golden["gsprd_q"], rtol=1e-3)
    else:
        expected = table["gsprd_q"]
    np.testing.assert_allclose(expected, golden["gsprd_q"], rtol=1e-4)


def test_fas_loop_one_row_per_scenario() -> None:
    path = sorted((GOLDEN / "fas").glob("*.col"))[0]
    golden = read_col(path)
    table = fas_loop(shipped_params(path.stem), GRID, FREQUENCIES)
    assert len(table) == len(GRID.magnitudes) * len(GRID.distances)
    names = [fas_column(f) for f in FREQUENCIES]
    assert list(table.columns[-21:]) == names
    expected = golden.fas.to_numpy().reshape(len(table), FREQUENCIES.size)
    np.testing.assert_allclose(table[names].to_numpy(), expected, rtol=1e-7)
