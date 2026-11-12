"""Fourier spectra, random-vibration and time-domain peak motions in one run.

:func:`run` computes, for every scenario of a
:class:`~pysmsim.scenario.ScenarioGrid`, the Fourier acceleration spectrum
at given frequencies, the random-vibration peak motions and, when asked,
the time-domain simulations, and returns tables that share their rows (one
per scenario, in the order stress, kappa_0, magnitude, distance)::

    import numpy as np
    from pysmsim.io.config import load_config
    from pysmsim.drivers.run import run
    from pysmsim.io.hdf5 import write_hdf5
    config = load_config("run.toml")
    result = run(config.params, config.scenarios, np.logspace(-2, 2, 41))
    result.scenarios  # magnitude, distances, kappa_0, stress, corner frequencies
    result.fas        # fas(f) for each frequency
    result.rv         # fup, durations, pgv, pga and psa(T) for each period
    write_hdf5(result, "results.h5")

    result = run(config.params, config.scenarios, td=True)
    result.td         # means of the simulations: pga, arias, psa(T), ...
    result.td_details # one row per scenario and measure, with the spreads

The tables share their index (the scenario number, from 0), so
``result.scenarios.join(result.rv)`` gives the table of
:func:`~pysmsim.drivers.loop_rv.rv_loop`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from pysmsim.drivers.loop_fas import fas_column, params_for
from pysmsim.drivers.loop_rv import psa_column, rv_loop
from pysmsim.drivers.loop_td import TD_COLUMNS, TD_SCENARIO_COLUMNS, td_loop_details, td_table
from pysmsim.params import Params
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.distance import rjb_to_rrup_depth
from pysmsim.spectrum.fas import spectrum_factors

__all__ = ["RV_COLUMNS", "SCENARIO_COLUMNS", "TD_COLUMNS", "RunResult", "run"]

SCENARIO_COLUMNS = (
    "magnitude",
    "rjb",
    "h",
    "rrup",
    "f_ff",
    "rps",
    "kappa0",
    "stress",
    "fa",
    "fb",
)
"""Columns of :attr:`RunResult.scenarios`.

magnitude, rjb (km; NaN when the grid gives Rrup), h (km, the depth of the
Rjb-to-Rrup conversion; NaN when the grid gives Rrup), rrup (km), f_ff
(finite-fault factor, km), rps (km), kappa0 (s), stress (bars; NaN for
source models without one), fa and fb (corner frequencies, Hz).
"""

RV_COLUMNS = ("fup", "dur_source", "dur_path", "dur_ex")
"""Columns of :attr:`RunResult.rv` before the peak motions.

fup (upper frequency of the moments, Hz), and the source, path and
excitation durations (s).
"""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class RunResult:
    """The tables of a run, one row per scenario.

    Attributes
    ----------
    params : Params
        The model parameters of the run.
    grid : ScenarioGrid
        The scenarios of the run.
    fortran_compat : bool
        Whether the run used SMSIM's conventions (see :func:`run`).
    scenarios : pandas.DataFrame
        The columns :data:`SCENARIO_COLUMNS`.
    frequencies : ndarray
        The frequencies of the FAS (Hz); empty when no FAS was computed.
    fas : pandas.DataFrame or None
        ``fas(f)``, the acceleration spectrum (cm/s), for each frequency.
    rv : pandas.DataFrame or None
        The columns :data:`RV_COLUMNS`, then ``pgv`` (cm/s), ``pga``
        (cm/s^2) (those the grid asks for) and ``psa(T)`` (cm/s^2) for each
        period of the grid. NaN where a BT15 table does not reach the
        magnitude and distance.
    td : pandas.DataFrame or None
        Time-domain simulations: the columns
        :data:`~pysmsim.drivers.loop_td.TD_COLUMNS`, then the arithmetic
        means of ``pgv`` (cm/s), ``pga`` (cm/s^2) (those the grid asks
        for), ``pgd`` (cm), ``arias`` (cm/s), ``dur_75_05`` and
        ``dur_95_05`` (s) and ``psa(T)`` (cm/s^2) for each period.
    td_details : pandas.DataFrame or None
        The table of :func:`~pysmsim.drivers.loop_td.td_loop_details`: one
        row per scenario and measure, with the arithmetic and geometric
        means and spreads of the simulations.
    """

    params: Params
    grid: ScenarioGrid
    fortran_compat: bool
    scenarios: pd.DataFrame
    frequencies: npt.NDArray[np.float64]
    fas: pd.DataFrame | None
    rv: pd.DataFrame | None
    td: pd.DataFrame | None = None
    td_details: pd.DataFrame | None = None


def run(
    params: Params,
    grid: ScenarioGrid,
    frequencies: npt.ArrayLike | None = None,
    *,
    rv: bool = True,
    td: bool = False,
    fortran_compat: bool = False,
    workers: int | None = 1,
) -> RunResult:
    """Compute the FAS and the peak motions of a grid.

    Parameters
    ----------
    params : Params
        Model parameters.
    grid : ScenarioGrid
        Magnitudes, distances, periods, damping, whether to compute PGV and
        PGA, the Rjb-to-Rrup conversion and optional kappa_0 and stress
        values.
    frequencies : array_like, optional
        Frequencies of the FAS (Hz), one-dimensional; ``grid.frequencies``
        if not given. No FAS when there are none.
    rv : bool
        Whether to compute the random-vibration peak motions.
    td : bool
        Whether to simulate ``params.td.nsims`` time series of every
        scenario (much slower than RV; see
        :func:`~pysmsim.drivers.loop_td.td_loop_details`).
    fortran_compat : bool
        Make kappa independent of magnitude even when the grid does not
        give kappa_0, compute fup from kappa_0, and compute the DK80 peak
        factor with SMSIM's sum (see
        :func:`~pysmsim.drivers.loop_rv.rv_loop_details`).
    workers : int or None
        Number of processes for the RV and TD loops: 1 (the default)
        computes in this process, None or 0 uses one per CPU. The results do
        not depend on it.

    Returns
    -------
    RunResult

    Raises
    ------
    ValueError
        If the frequencies are not a one-dimensional array of distinct
        values >= 0 (to 6 significant digits, the precision of the column
        names).
    """
    f = np.asarray(grid.frequencies if frequencies is None else frequencies, dtype=np.float64)
    if f.ndim != 1 or np.any(f < 0):
        raise ValueError("frequencies must be a one-dimensional array of values >= 0")
    names = [fas_column(x) for x in f]
    if len(set(names)) != len(names):
        raise ValueError("frequencies must be distinct to 6 significant digits")
    acc = (2.0 * math.pi * f) ** 2

    by_rjb = grid.distance_metric is DistanceMetric.RJB
    rows: list[tuple[float, ...]] = []
    spectra: list[npt.NDArray[np.float64]] = []
    for stress in grid.stress or (None,):
        for kappa0 in grid.kappa0 or (None,):
            run_params = params_for(
                params, kappa0=kappa0, stress=stress, fortran_compat=fortran_compat
            )
            for magnitude in grid.magnitudes:
                h = (
                    rjb_to_rrup_depth(
                        magnitude,
                        conversion=grid.rjb_to_rrup,
                        mechanism=grid.mechanism,
                        depth_function=grid.depth_function,
                    )
                    if by_rjb
                    else math.nan
                )
                for distance in grid.distances:
                    rjb = distance if by_rjb else math.nan
                    rrup = math.hypot(rjb, h) if by_rjb else distance
                    factors = spectrum_factors(f, run_params, magnitude, rrup)
                    terms = factors.terms
                    source = terms.source
                    rows.append(
                        (
                            magnitude,
                            rjb,
                            h,
                            rrup,
                            terms.finite_fault_h,
                            terms.rps,
                            run_params.site.kappa0,
                            math.nan if source.stress is None else source.stress,
                            source.fa,
                            source.fb,
                        )
                    )
                    spectra.append(acc * factors.product())

    scenarios = pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS), dtype=np.float64)
    fas = None
    if f.size:
        fas = pd.DataFrame(np.vstack(spectra), columns=names)

    rv_table = None
    if rv:
        wide = rv_loop(params, grid, fortran_compat=fortran_compat, workers=workers)
        motions = ["pgv"] * grid.pgv + ["pga"] * grid.pga + [psa_column(t) for t in grid.periods]
        rv_table = wide[[*RV_COLUMNS, *motions]].reset_index(drop=True)

    td_wide: pd.DataFrame | None = None
    td_details: pd.DataFrame | None = None
    if td:
        td_details = td_loop_details(params, grid, fortran_compat=fortran_compat, workers=workers)
        wide = td_table(td_details, grid)
        motions = [c for c in wide.columns if c not in TD_SCENARIO_COLUMNS]
        td_wide = wide[[*TD_COLUMNS, *motions]].reset_index(drop=True)

    return RunResult(
        params=params,
        grid=grid,
        fortran_compat=fortran_compat,
        scenarios=scenarios,
        frequencies=f,
        fas=fas,
        rv=rv_table,
        td=td_wide,
        td_details=td_details,
    )
