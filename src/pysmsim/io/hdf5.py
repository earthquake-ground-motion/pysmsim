"""Writing and reading the results of a run as one HDF5 file.

:func:`write_hdf5` stores a :class:`~pysmsim.drivers.run.RunResult` in one
file with a group for each of its tables, and :func:`read_hdf5` reads it
back::

    from pysmsim.io.hdf5 import read_hdf5, write_hdf5
    write_hdf5(result, "results.h5")
    result = read_hdf5("results.h5")

The layout, which any HDF5 library can read (n scenarios, nf frequencies,
nt periods)::

    /                  attributes: format ("pysmsim-run"), format_version,
                       pysmsim_version, fortran_compat, config (the
                       parameters and scenarios as the TOML of a run file)
    /scenarios/<name>  (n,) for each of SCENARIO_COLUMNS
    /fas/frequency     (nf,) Hz
    /fas/fas           (n, nf) cm/s, the Fourier acceleration spectrum
    /rv/fup, /rv/dur_source, /rv/dur_path, /rv/dur_ex   (n,)
    /rv/pgv, /rv/pga   (n,) cm/s and cm/s^2, when the grid asks for them
    /rv/period         (nt,) s; attribute damping
    /rv/psa            (n, nt) cm/s^2
    /td/dur_source, /td/dur_path, /td/dur_ex, /td/te, /td/npts, /td/nsims  (n,)
    /td/period         (nt,) s; attribute damping
    /td/<stat>/<measure>   stat: mean, std (arithmetic), gmean, gfactor
                       (geometric); measure: pgv, pga, pgd, arias,
                       dur_75_05, dur_95_05 (n,), and psa (n, nt)

``fas``, ``rv`` and ``td`` are left out when the run did not compute them.

Every dataset has a ``units`` attribute, and the multi-dimensional ones a
``dimensions`` attribute naming their axes.

:func:`write_td_series` adds the time-domain acceleration series to a file
(or writes a new one), and :func:`read_td_series` reads them back (nsims
simulations, ns time steps, the length of the longest series)::

    /td/series/times         (ns,) s: 0, dt, 2*dt, ...; attributes dt,
                             tolerance and, one value per scenario, nstart
                             and nstop (the window), first and last (the
                             stored points) and npts (the series length)
    /td/series/acceleration  (nsims, ns, n) cm/s^2: points first to last
                             of each scenario, NaN elsewhere (not stored)

With h5py, for example::

    import h5py
    with h5py.File("results.h5") as h5:
        magnitude = h5["scenarios/magnitude"][:]
        psa_1s = h5["rv/psa"][:, list(h5["rv/period"][:]).index(1.0)]
        pga_td = h5["td/mean/pga"][:]
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from pysmsim import __version__
from pysmsim.drivers.loop_fas import fas_column
from pysmsim.drivers.loop_rv import psa_column
from pysmsim.drivers.loop_td import (
    TD_DETAIL_COLUMNS,
    td_loop_series,
    td_table,
    td_time_grids,
)
from pysmsim.drivers.run import RV_COLUMNS, SCENARIO_COLUMNS, TD_COLUMNS, RunResult
from pysmsim.io.config import RunConfig, dumps_config, parse_config
from pysmsim.params import Params
from pysmsim.scenario import ScenarioGrid

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "SERIES_TOLERANCE",
    "TD_MEASURES",
    "TD_STATISTICS",
    "UNITS",
    "TDSeries",
    "read_hdf5",
    "read_td_series",
    "write_hdf5",
    "write_td_series",
]

FORMAT = "pysmsim-run"
"""Value of the ``format`` attribute of the files."""

FORMAT_VERSION = 1
"""Version of the layout; it changes when the layout does."""

UNITS = {
    "magnitude": "",
    "rjb": "km",
    "h": "km",
    "rrup": "km",
    "f_ff": "km",
    "rps": "km",
    "kappa0": "s",
    "stress": "bar",
    "fa": "Hz",
    "fb": "Hz",
    "frequency": "Hz",
    "fas": "cm/s",
    "fup": "Hz",
    "dur_source": "s",
    "dur_path": "s",
    "dur_ex": "s",
    "pgv": "cm/s",
    "pga": "cm/s^2",
    "period": "s",
    "psa": "cm/s^2",
    "te": "s",
    "npts": "",
    "nsims": "",
    "pgd": "cm",
    "arias": "cm/s",
    "dur_75_05": "s",
    "dur_95_05": "s",
}
"""Units of each dataset (the ``gfactor`` of the time-domain measures has none)."""

TD_MEASURES = ("pgv", "pga", "pgd", "arias", "dur_75_05", "dur_95_05")
"""The time-domain measures besides PSA, in the order of the details table."""

TD_STATISTICS = ("mean", "std", "gmean", "gfactor")
"""The statistics of the time-domain measures."""

_MIN_COMPRESSED_SIZE = 1024
# Smaller datasets are stored as they are: the overhead of a compressed
# (chunked) dataset outweighs the saving


def _write(
    group: h5py.Group,
    name: str,
    data: npt.ArrayLike,
    compression: str | None,
    dimensions: str | None = None,
    *,
    units: str | None = None,
) -> None:
    values = np.asarray(data, dtype=np.float64)
    options: dict[str, Any] = {}
    if compression is not None and values.size >= _MIN_COMPRESSED_SIZE:
        options = {"compression": compression, "shuffle": True}
    dataset = group.create_dataset(name, data=values, **options)
    dataset.attrs["units"] = UNITS[name] if units is None else units
    if dimensions is not None:
        dataset.attrs["dimensions"] = dimensions


def write_hdf5(result: RunResult, path: str | Path, *, compression: str | None = "gzip") -> None:
    """Write the results of a run to an HDF5 file.

    Parameters
    ----------
    result : RunResult
        The results, from :func:`~pysmsim.drivers.run.run`.
    path : str or Path
        The file to write; an existing file is replaced.
    compression : str or None
        HDF5 filter for the datasets of 1024 values or more: ``"gzip"``
        (any HDF5 library can read it), ``"lzf"`` (faster, but only h5py
        reads it) or None.
    """
    config = RunConfig(params=result.params, scenarios=result.grid)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = FORMAT
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["pysmsim_version"] = __version__
        h5.attrs["fortran_compat"] = result.fortran_compat
        h5.attrs["config"] = dumps_config(config)

        scenarios = h5.create_group("scenarios")
        for name in SCENARIO_COLUMNS:
            _write(scenarios, name, result.scenarios[name], compression)

        if result.fas is not None:
            fas = h5.create_group("fas")
            _write(fas, "frequency", result.frequencies, compression)
            _write(fas, "fas", result.fas, compression, "scenario, frequency")

        if result.rv is not None:
            rv = h5.create_group("rv")
            for name in RV_COLUMNS:
                _write(rv, name, result.rv[name], compression)
            for name in ("pgv", "pga"):
                if name in result.rv:
                    _write(rv, name, result.rv[name], compression)
            periods = result.grid.periods
            _write(rv, "period", periods, compression)
            rv["period"].attrs["damping"] = result.grid.damping
            psa = result.rv[[psa_column(t) for t in periods]].to_numpy(dtype=np.float64)
            _write(rv, "psa", psa, compression, "scenario, period")

        if result.td is not None and result.td_details is not None:
            _write_td(h5.create_group("td"), result, compression)


def _write_td(group: h5py.Group, result: RunResult, compression: str | None) -> None:
    td = result.td
    details = result.td_details
    assert td is not None and details is not None
    n = len(result.scenarios)
    for name in TD_COLUMNS:
        _write(group, name, td[name], compression)
    periods = result.grid.periods
    _write(group, "period", periods, compression)
    group["period"].attrs["damping"] = result.grid.damping
    for stat in TD_STATISTICS:
        sub = group.create_group(stat)
        units = "" if stat == "gfactor" else None
        for measure in TD_MEASURES:
            values = details.loc[details.measure == measure, stat].to_numpy()
            _write(sub, measure, values, compression, units=units)
        psa = details.loc[details.measure == "psa", stat].to_numpy().reshape(n, len(periods))
        _write(sub, "psa", psa, compression, "scenario, period", units=units)


def _read_td(group: h5py.Group, scenarios: pd.DataFrame, damping: float) -> pd.DataFrame:
    """Rebuild the table of :func:`~pysmsim.drivers.loop_td.td_loop_details` from /td."""
    n = len(scenarios)
    periods = group["period"][:]
    per_scenario = pd.concat(
        [scenarios, pd.DataFrame({name: group[name][:] for name in TD_COLUMNS})], axis=1
    )
    per_scenario = per_scenario.astype({"npts": np.int64, "nsims": np.int64})
    stats = {s: {m: group[s][m][:] for m in (*TD_MEASURES, "psa")} for s in TD_STATISTICS}
    blocks = [
        per_scenario.assign(
            measure=measure,
            period=np.nan,
            damping=np.nan,
            order=order,
            **{s: stats[s][measure] for s in TD_STATISTICS},
        )
        for order, measure in enumerate(TD_MEASURES)
    ]
    blocks += [
        per_scenario.assign(
            measure="psa",
            period=period,
            damping=damping,
            order=len(TD_MEASURES) + k,
            **{s: stats[s]["psa"].reshape(n, periods.size)[:, k] for s in TD_STATISTICS},
        )
        for k, period in enumerate(periods)
    ]
    long = pd.concat(blocks).rename_axis("scenario").reset_index()
    long = long.sort_values(["scenario", "order"], kind="stable").reset_index(drop=True)
    return pd.DataFrame(long[list(TD_DETAIL_COLUMNS)])


def read_hdf5(path: str | Path) -> RunResult:
    """Read the results of a run from a file of :func:`write_hdf5`.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    RunResult
        The results, with the parameters and scenarios rebuilt from the
        file's ``config`` attribute.

    Raises
    ------
    ValueError
        If the file is not a pysmsim results file of a version this
        reader knows.
    """
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a pysmsim results file")
        version = int(h5.attrs["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path} has format version {version}; this reader knows 1")
        config = parse_config(tomllib.loads(str(h5.attrs["config"])))
        fortran_compat = bool(h5.attrs["fortran_compat"])

        scenarios = pd.DataFrame(
            {name: h5["scenarios"][name][:] for name in SCENARIO_COLUMNS}, dtype=np.float64
        )

        frequencies = np.empty(0)
        fas = None
        if "fas" in h5:
            frequencies = h5["fas/frequency"][:]
            fas = pd.DataFrame(
                h5["fas/fas"][:].reshape(len(scenarios), frequencies.size),
                columns=[fas_column(x) for x in frequencies],
            )

        rv = None
        if "rv" in h5:
            group = h5["rv"]
            columns = {name: group[name][:] for name in RV_COLUMNS}
            for name in ("pgv", "pga"):
                if name in group:
                    columns[name] = group[name][:]
            periods = group["period"][:]
            psa = group["psa"][:].reshape(len(scenarios), periods.size)
            columns.update({psa_column(t): psa[:, i] for i, t in enumerate(periods)})
            rv = pd.DataFrame(columns, dtype=np.float64)

        td: pd.DataFrame | None = None
        td_details: pd.DataFrame | None = None
        if "td/mean" in h5:
            td_details = _read_td(h5["td"], scenarios, config.scenarios.damping)
            wide = td_table(td_details, config.scenarios)
            td = wide.drop(columns=list(SCENARIO_COLUMNS))

    return RunResult(
        params=config.params,
        grid=config.scenarios,
        fortran_compat=fortran_compat,
        scenarios=scenarios,
        frequencies=np.asarray(frequencies, dtype=np.float64),
        fas=fas,
        rv=rv,
        td=td,
        td_details=td_details,
    )


# --- Time-domain series ---------------------------------------------------------------

SERIES_TOLERANCE = 0.0
"""Default of :func:`write_td_series`: keep the whole series."""

_SERIES_CHUNK = 1024
"""Time steps in each chunk of ``/td/series/acceleration``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TDSeries:
    """Acceleration series of a run, as :func:`write_td_series` stores them.

    Attributes
    ----------
    times : ndarray
        Time of each point (s), shape ``(ns,)``: ``0, dt, 2*dt, ...``, as
        :meth:`~pysmsim.td.window.TimeGrid.times`, as long as the longest
        series.
    acceleration : ndarray
        Acceleration (cm/s^2), shape ``(nsims, ns, k)`` for the k
        scenarios read; NaN outside the stored part of each scenario.
    dt : float
        Time step (s).
    nstart, nstop : ndarray
        Window of each scenario: the indices of its first and last point
        (``TimeGrid.nstart`` and ``nstop``).
    first, last : ndarray
        Stored part of each scenario: the indices of its first and last
        stored point.
    npts : ndarray
        Number of points of each scenario's series.
    tolerance : float
        The tolerance the file was written with.
    """

    times: npt.NDArray[np.float64]
    acceleration: npt.NDArray[np.float64]
    dt: float
    nstart: npt.NDArray[np.int64]
    nstop: npt.NDArray[np.int64]
    first: npt.NDArray[np.int64]
    last: npt.NDArray[np.int64]
    npts: npt.NDArray[np.int64]
    tolerance: float


def _stored_range(acc: npt.NDArray[np.float64], tolerance: float) -> tuple[int, int]:
    """First and last index where any series exceeds tolerance times its peak."""
    if tolerance <= 0.0:
        return 0, acc.shape[-1] - 1
    above = np.abs(acc) > tolerance * np.abs(acc).max(axis=-1, keepdims=True)
    index = np.flatnonzero(above.any(axis=0))
    return int(index[0]), int(index[-1])


def write_td_series(
    path: str | Path,
    params: Params,
    grid: ScenarioGrid,
    *,
    nsims: int | None = None,
    tolerance: float = SERIES_TOLERANCE,
    fortran_compat: bool = False,
    compression: str | None = "gzip",
    dtype: npt.DTypeLike = np.float32,
) -> None:
    """Simulate the acceleration series of a grid and write them to an HDF5 file.

    The series are those of :func:`~pysmsim.drivers.run.run` with
    ``td=True`` (the same seed), simulated again one scenario at a time and
    written as they come, so a file can be larger than memory.

    By default the whole series is stored, in 32-bit floats (about 7
    significant digits, half the size of 64-bit). The FFT leaves a series
    nonzero everywhere, decaying away from the window (``nstart`` to
    ``nstop``) to about 1e-6 of its peak; with a ``tolerance`` above 0,
    each scenario is stored only from the first to the last point where
    one of its simulations exceeds ``tolerance`` times its peak (1e-4
    keeps a third of the size in a typical run), and the points outside
    are NaN and take no space in the file.

    Parameters
    ----------
    path : str or Path
        The file: the ``/td/series`` group is added to it (and replaced if
        there is one), or the file is created.
    params, grid : Params, ScenarioGrid
        The run.
    nsims : int, optional
        Write only the first ``nsims`` simulations of each scenario
        (default all, ``params.td.nsims``).
    tolerance : float
        Fraction of the peak below which the start and end of a series are
        not stored; 0 (the default) stores the whole series.
    fortran_compat : bool
        As for :func:`~pysmsim.drivers.run.run`.
    compression : str or None
        HDF5 filter for the acceleration (``"gzip"``, ``"lzf"`` or None).
    dtype : numpy dtype
        Type stored: ``float32`` (the default) or ``float64``.
    """
    grids = td_time_grids(params, grid, fortran_compat=fortran_compat)
    count = params.td.nsims if nsims is None else nsims
    if count < 1:
        raise ValueError("nsims must be at least 1")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be at least 0 and below 1")
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    dt = grids[0].dt
    npts = np.array([g.npts for g in grids], dtype=np.int64)
    ns = int(npts.max())
    chunk_t = min(ns, _SERIES_CHUNK)
    options: dict[str, Any] = {}
    if compression is not None:
        options = {"compression": compression, "shuffle": True}
    first = np.zeros(len(grids), dtype=np.int64)
    last = np.zeros(len(grids), dtype=np.int64)
    with h5py.File(path, "a") as h5:
        td = h5.require_group("td")
        if "series" in td:
            del td["series"]
        group = td.create_group("series")
        _write(group, "times", dt * np.arange(ns, dtype=np.float64), None, units="s")
        acc = group.create_dataset(
            "acceleration",
            shape=(count, ns, len(grids)),
            dtype=dtype,
            chunks=(min(count, max(1, (1 << 20) // chunk_t)), chunk_t, 1),
            fillvalue=np.nan,
            **options,
        )
        acc.attrs["units"] = "cm/s^2"
        acc.attrs["dimensions"] = "simulation, time, scenario"
        for k, result in enumerate(
            td_loop_series(params, grid, nsims=count, fortran_compat=fortran_compat)
        ):
            assert result.acc is not None
            first[k], last[k] = _stored_range(result.acc, tolerance)
            acc[:, first[k] : last[k] + 1, k] = result.acc[:, first[k] : last[k] + 1]
        attrs = group["times"].attrs
        attrs["dt"] = dt
        attrs["tolerance"] = tolerance
        attrs["nstart"] = np.array([g.nstart for g in grids], dtype=np.int64)
        attrs["nstop"] = np.array([g.nstop for g in grids], dtype=np.int64)
        attrs["first"] = first
        attrs["last"] = last
        attrs["npts"] = npts


def read_td_series(path: str | Path, scenarios: Sequence[int] | None = None) -> TDSeries:
    """Read the acceleration series that :func:`write_td_series` wrote.

    Parameters
    ----------
    path : str or Path
        The file.
    scenarios : sequence of int, optional
        Read only these scenarios (numbers from 0, in the order of the run);
        all by default.

    Raises
    ------
    KeyError
        If the file has no ``/td/series`` group.
    """
    with h5py.File(path, "r") as h5:
        if "td/series" not in h5:
            raise KeyError(f"{path} has no time-domain series (/td/series)")
        times = h5["td/series/times"]
        attrs = times.attrs
        dataset = h5["td/series/acceleration"]
        which = list(range(dataset.shape[2])) if scenarios is None else list(scenarios)
        acceleration = np.stack([dataset[:, :, k] for k in which], axis=-1)

        def per_scenario(name: str) -> npt.NDArray[np.int64]:
            return np.asarray(attrs[name], dtype=np.int64)[which]

        return TDSeries(
            times=times[:],
            acceleration=np.asarray(acceleration, dtype=np.float64),
            dt=float(attrs["dt"]),
            nstart=per_scenario("nstart"),
            nstop=per_scenario("nstop"),
            first=per_scenario("first"),
            last=per_scenario("last"),
            npts=per_scenario("npts"),
            tolerance=float(attrs["tolerance"]),
        )
