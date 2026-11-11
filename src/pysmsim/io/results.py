"""Writing the results of a run: CSV, Parquet, HDF5 and SMSIM's ``.col``.

:func:`write_results` writes a :class:`~pysmsim.drivers.run.RunResult` in
one of the :class:`~pysmsim.scenario.OutputFormat` formats::

    from pysmsim.io.results import write_results
    write_results(result, "results.parquet", "parquet")

- ``csv`` and ``parquet``: one table, one row per scenario
  (:func:`results_table`): the scenario columns, then the RV columns, then
  ``fas(f)``.
- ``hdf5``: the three tables in one file (:func:`pysmsim.io.hdf5.write_hdf5`).
- ``col``: the layout of SMSIM's ``tmrsk_loop_rv_drvr`` ``.col`` file, one
  row per scenario and measure (:func:`write_col`), for comparison with
  the Fortran.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pysmsim.drivers.loop_rv import psa_column
from pysmsim.drivers.run import RunResult
from pysmsim.io.hdf5 import write_hdf5
from pysmsim.scenario import OutputFormat

__all__ = [
    "COL_SENTINEL",
    "SOURCE_NUMBERS",
    "format_for_path",
    "results_table",
    "write_col",
    "write_results",
]

SOURCE_NUMBERS = {
    "single_corner": 1,
    "joyner_1984": 2,
    "atkinson_1993": 3,
    "atkinson_silva_1997": 4,
    "haddon_1996": 5,
    "atkinson_boore_1998_california": 6,
    "boatwright_choy": 7,
    "joyner_ena": 8,
    "atkinson_silva_2000": 9,
    "atkinson_2005": 10,
    "two_corner_multiplicative": 11,
    "two_corner_additive": 12,
    "ji_archuleta_2020": 13,
}
"""Fortran ``numsource`` of each source model, for the ``.col`` output."""

COL_SENTINEL = -999.0
"""Value written in a ``.col`` file in place of NaN, as SMSIM does."""

_G = 981.0  # cm/s^2, only for SMSIM's gmim(g) column of the .col output

_SUFFIXES = {
    ".csv": OutputFormat.CSV,
    ".parquet": OutputFormat.PARQUET,
    ".pq": OutputFormat.PARQUET,
    ".h5": OutputFormat.HDF5,
    ".hdf5": OutputFormat.HDF5,
    ".hdf": OutputFormat.HDF5,
    ".col": OutputFormat.COL,
}


def format_for_path(path: str | Path) -> OutputFormat | None:
    """Return the output format a file name suggests, or None if unknown."""
    return _SUFFIXES.get(Path(path).suffix.lower())


def results_table(result: RunResult) -> pd.DataFrame:
    """One table of a run: one row per scenario.

    The columns of ``result.scenarios``, then those of ``result.rv`` and
    ``result.fas`` (when the run computed them).
    """
    table = result.scenarios
    for part in (result.rv, result.fas):
        if part is not None:
            table = table.join(part)
    return table


def _number(value: float, fmt: str) -> str:
    return format(COL_SENTINEL if math.isnan(value) else value, fmt)


def write_col(result: RunResult, path: str | Path) -> None:
    """Write the RV results in the layout of SMSIM's ``tmrsk_loop_rv_drvr``.

    One row per scenario and measure (PGV, PGA, then each period), with the
    columns damp, T (-1 for PGV, 0 for PGA), F (-1 for PGV, 999.99 for
    PGA, 1/T otherwise), M, Rjb, h, Rrup, F_FF, Rps, k0, numsource, S,
    gmim(g) and gmim(cgs). As in SMSIM, gmim(g) is PGV in cm/s for PGV, and
    NaN is written as -999.

    Raises
    ------
    ValueError
        If the run has no RV results.
    """
    if result.rv is None:
        raise ValueError("the .col output needs the RV results")
    grid = result.grid
    measures: list[tuple[str, float, float]] = []
    if grid.pgv:
        measures.append(("pgv", -1.0, -1.0))
    if grid.pga:
        measures.append(("pga", 0.0, 999.99))
    measures += [(psa_column(t), t, 1.0 / t) for t in grid.periods]
    numsource = SOURCE_NUMBERS.get(result.params.source.name, 0)
    title = result.params.title or "pysmsim run"
    lines = [
        f" Parameter file: {title}",
        "  damp      T:tmrs      F:tmrs      M     Rjb     h    Rrup     F_FF     Rps     k0"
        " numsource   S:tmrs                 gmim(g)                 gmim(cgs)",
    ]
    scenario_formats = [
        ("magnitude", "7.3f"),
        ("rjb", "8.2f"),
        ("h", "6.2f"),
        ("rrup", "8.2f"),
        ("f_ff", "9.3f"),
        ("rps", "8.2f"),
        ("kappa0", "7.4f"),
    ]
    scenarios = result.scenarios
    motions = result.rv
    for i in range(len(scenarios)):
        head = [_number(float(scenarios[name].iloc[i]), fmt) for name, fmt in scenario_formats]
        head += [f"{numsource:10d}", _number(float(scenarios["stress"].iloc[i]), "9.2f")]
        for column, period, frequency in measures:
            cgs = float(motions[column].iloc[i])
            g = cgs if column == "pgv" else cgs / _G
            fields = [
                f"{grid.damping:6.3f}",
                format(period, "11.4E"),
                format(frequency, "11.4E"),
                *head,
                _number(g, "24.6E"),
                _number(cgs, "26.6E"),
            ]
            lines.append(" ".join(fields))
    Path(path).write_text("\n".join(lines) + "\n")


def write_results(result: RunResult, path: str | Path, fmt: OutputFormat | str) -> None:
    """Write the results of a run to a file.

    Parameters
    ----------
    result : RunResult
        The results, from :func:`~pysmsim.drivers.run.run`.
    path : str or Path
        The file to write; an existing file is replaced.
    fmt : OutputFormat or str
        ``"csv"``, ``"parquet"``, ``"hdf5"`` or ``"col"`` (see the module
        documentation).
    """
    match OutputFormat(fmt):
        case OutputFormat.CSV:
            results_table(result).to_csv(path, index=False)
        case OutputFormat.PARQUET:
            results_table(result).to_parquet(path, index=False)
        case OutputFormat.HDF5:
            write_hdf5(result, path)
        case OutputFormat.COL:
            write_col(result, path)
