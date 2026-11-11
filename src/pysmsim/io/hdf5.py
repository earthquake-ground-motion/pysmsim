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

``fas`` and ``rv`` are left out when the run did not compute them. Every
dataset has a ``units`` attribute, and the two-dimensional ones a
``dimensions`` attribute naming their axes. With h5py, for example::

    import h5py
    with h5py.File("results.h5") as h5:
        magnitude = h5["scenarios/magnitude"][:]
        psa_1s = h5["rv/psa"][:, list(h5["rv/period"][:]).index(1.0)]
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from pysmsim import __version__
from pysmsim.drivers.loop_fas import fas_column
from pysmsim.drivers.loop_rv import psa_column
from pysmsim.drivers.run import RV_COLUMNS, SCENARIO_COLUMNS, RunResult
from pysmsim.io.config import RunConfig, dumps_config, parse_config

__all__ = ["FORMAT", "FORMAT_VERSION", "UNITS", "read_hdf5", "write_hdf5"]

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
}
"""Units of each dataset."""

_MIN_COMPRESSED_SIZE = 1024
# Smaller datasets are stored as they are: the overhead of a compressed
# (chunked) dataset outweighs the saving


def _write(
    group: h5py.Group,
    name: str,
    data: npt.ArrayLike,
    compression: str | None,
    dimensions: str | None = None,
) -> None:
    values = np.asarray(data, dtype=np.float64)
    options: dict[str, Any] = {}
    if compression is not None and values.size >= _MIN_COMPRESSED_SIZE:
        options = {"compression": compression, "shuffle": True}
    dataset = group.create_dataset(name, data=values, **options)
    dataset.attrs["units"] = UNITS[name]
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

    return RunResult(
        params=config.params,
        grid=config.scenarios,
        fortran_compat=fortran_compat,
        scenarios=scenarios,
        frequencies=np.asarray(frequencies, dtype=np.float64),
        fas=fas,
        rv=rv,
    )
