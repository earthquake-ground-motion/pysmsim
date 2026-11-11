"""Read the Boore and Thompson (2015) coefficient tables (``.pars`` files).

Two tables come with pysmsim, as package data in ``pysmsim/data/bt15``:

- ``bt15_wna_acr``: active crustal regions (SMSIM's
  ``bt15_wna_acr_trms4osc.dk_rms2pk.pars``);
- ``bt15_ena_scr``: stable continental regions
  (``bt15_ena_scr_trms4osc.dk_rms2pk.pars``).

Both are for the DK80 peak factor that SMSIM reports. A table is loaded by
name with :func:`load_bt15_table`, or from any file in the same format with
:func:`read_bt15_table`.
"""

from __future__ import annotations

import math
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np

from pysmsim.rv.duration import BT15Table

__all__ = ["BT15_TABLE_NAMES", "load_bt15_table", "parse_bt15_table", "read_bt15_table"]

BT15_TABLE_NAMES = ("bt15_wna_acr", "bt15_ena_scr")
"""Names of the bundled tables."""


def parse_bt15_table(text: str, name: str) -> BT15Table:
    """Parse the text of a ``.pars`` file (as SMSIM's ``get_params`` reads it).

    The file has a title line, a label line, the numbers of magnitudes and
    distances (nm, nr), a header line, then nm*nr rows of M, Rps, c1 to c7,
    TD/RV for PGA and TD/RV for PGV, with M varying fastest.
    """
    lines = text.splitlines()
    nm, nr = (int(v) for v in lines[2].split()[:2])
    rows = np.array(
        [[float(v) for v in line.split()[:11]] for line in lines[4 : 4 + nm * nr]],
        dtype=np.float64,
    )
    if rows.shape != (nm * nr, 11):
        raise ValueError(f"{name}: expected {nm * nr} rows of 11 values")
    grid = rows.reshape(nr, nm, 11).transpose(1, 0, 2)  # (nm, nr, 11)
    return BT15Table(
        name=name,
        magnitudes=grid[:, 0, 0].copy(),
        log_distances=np.array([math.log10(r) for r in grid[0, :, 1]]),
        coefficients=np.ascontiguousarray(grid[:, :, 2:9]),
        td_rv_pga=np.ascontiguousarray(grid[:, :, 9]),
        td_rv_pgv=np.ascontiguousarray(grid[:, :, 10]),
    )


def read_bt15_table(path: str | Path) -> BT15Table:
    """Read a ``.pars`` file of BT15 coefficients."""
    path = Path(path)
    return parse_bt15_table(path.read_text(), path.stem)


@cache
def load_bt15_table(name: str) -> BT15Table:
    """Load a bundled table by name (one of :data:`BT15_TABLE_NAMES`)."""
    if name not in BT15_TABLE_NAMES:
        raise ValueError(f"unknown BT15 table {name!r}; choose from {BT15_TABLE_NAMES}")
    text = resources.files("pysmsim").joinpath("data", "bt15", f"{name}.pars").read_text()
    return parse_bt15_table(text, name)
