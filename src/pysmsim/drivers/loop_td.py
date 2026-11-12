"""Time-domain peak motions for a grid of scenarios (Fortran ``tmrsk_loop_td_drvr``).

:func:`td_loop` simulates ``params.td.nsims`` accelerograms for every
combination of stress, kappa_0, magnitude and distance in a
:class:`~pysmsim.scenario.ScenarioGrid`, and returns one row per scenario
with the arithmetic means of the simulations, as SMSIM reports them::

    from pysmsim.io.config import load_config
    from pysmsim.drivers.loop_td import td_loop
    config = load_config("run.toml")
    table = td_loop(config.params, config.scenarios)
    table[["magnitude", "rjb", "pga", "arias", "psa(1)"]]

:func:`td_loop_details` gives one row per scenario and measure with the
arithmetic and geometric means and spreads of the simulations, and
:func:`td_loop_series` the acceleration series of each scenario in turn
(:func:`pysmsim.io.hdf5.write_td_series` writes them to a file). The
drivers only compose :func:`pysmsim.td.gm_td.td_motion`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pysmsim.drivers._parallel import map_tasks
from pysmsim.drivers.loop_fas import params_for
from pysmsim.drivers.loop_rv import psa_column
from pysmsim.params import Params
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.distance import rjb_to_rrup_depth
from pysmsim.spectrum.fas import scenario_terms
from pysmsim.td.gm_td import Statistics, TDResult, td_motion
from pysmsim.td.window import TimeGrid, time_grid

__all__ = [
    "TD_COLUMNS",
    "TD_DETAIL_COLUMNS",
    "TD_SCENARIO_COLUMNS",
    "td_loop",
    "td_loop_details",
    "td_loop_series",
    "td_table",
    "td_time_grids",
]

TD_SCENARIO_COLUMNS = (
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
    "dur_source",
    "dur_path",
    "dur_ex",
    "te",
    "npts",
    "nsims",
)
"""Columns of :func:`td_table` that describe each scenario.

The scenario (as in :data:`~pysmsim.drivers.loop_rv.RV_DETAIL_COLUMNS`),
the source, path and excitation durations (s), te (s; the time at which the
exponential window falls to eta_w, 0 for the box window), npts (points in
each series) and nsims (simulations)."""

TD_COLUMNS = ("dur_source", "dur_path", "dur_ex", "te", "npts", "nsims")
"""Columns of :attr:`~pysmsim.drivers.run.RunResult.td` before the motions."""

TD_DETAIL_COLUMNS = (
    "measure",
    "period",
    "damping",
    *TD_SCENARIO_COLUMNS,
    "mean",
    "std",
    "gmean",
    "gfactor",
)
"""Columns of the table from :func:`td_loop_details`.

- measure: ``"pgv"`` (cm/s), ``"pga"`` (cm/s^2), ``"pgd"`` (cm),
  ``"arias"`` (cm/s), ``"dur_75_05"`` and ``"dur_95_05"`` (s), or
  ``"psa"`` (cm/s^2, an oscillator);
- period (s; NaN except for PSA) and damping;
- the columns :data:`TD_SCENARIO_COLUMNS`;
- mean and std (arithmetic mean and standard deviation of the
  simulations), gmean and gfactor (geometric mean, and the factor
  10**std(log10) that gives one geometric standard deviation).
"""

_MEASURES = ("pgv", "pga", "pgd", "arias", "dur_75_05", "dur_95_05")


def _scenarios(
    grid: ScenarioGrid,
) -> Iterator[tuple[float | None, float | None, float, float, float, float]]:
    """stress, kappa0, magnitude, rjb, h, rrup of each scenario, in loop order."""
    by_rjb = grid.distance_metric is DistanceMetric.RJB
    for stress in grid.stress or (None,):
        for kappa0 in grid.kappa0 or (None,):
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
                    yield stress, kappa0, magnitude, rjb, h, rrup


def _runs(
    params: Params, grid: ScenarioGrid, fortran_compat: bool
) -> Iterator[tuple[Params, float, float, float, float]]:
    """Parameters, magnitude, rjb, h and rrup of each scenario, in loop order."""
    cache: dict[tuple[float | None, float | None], Params] = {}
    for stress, kappa0, magnitude, rjb, h, rrup in _scenarios(grid):
        run = cache.get((stress, kappa0))
        if run is None:
            run = params_for(params, kappa0=kappa0, stress=stress, fortran_compat=fortran_compat)
            cache[(stress, kappa0)] = run
        yield run, magnitude, rjb, h, rrup


def td_time_grids(
    params: Params, grid: ScenarioGrid, *, fortran_compat: bool = False
) -> list[TimeGrid]:
    """Return the time grid of each scenario of a grid, in loop order, without simulating.

    Options as for :func:`td_loop_details`.
    """
    grids = []
    for run, magnitude, _, _, rrup in _runs(params, grid, fortran_compat):
        terms = scenario_terms(run, magnitude, rrup)
        dur_source = run.source.duration(terms.source)
        dur_path = run.path_duration.duration(terms.rps)
        grids.append(time_grid(run.td, run.filters.low_cut, dur_source, dur_path))
    return grids


def td_loop_series(
    params: Params,
    grid: ScenarioGrid,
    *,
    nsims: int | None = None,
    fortran_compat: bool = False,
) -> Iterator[TDResult]:
    """Yield the simulations of each scenario with their acceleration series.

    The same simulations as :func:`td_loop_details` (the same seed), one
    :class:`~pysmsim.td.gm_td.TDResult` per scenario in loop order, with
    ``acc`` kept; ``nsims`` takes the first simulations only. Memory:
    nsims * npts values of one scenario at a time.
    """
    for run, magnitude, _, _, rrup in _runs(params, grid, fortran_compat):
        yield td_motion(
            run, magnitude, rrup, grid.periods, grid.damping, nsims=nsims, keep_series=True
        )


_Scenario = tuple[float | None, float | None, float, float, float, float]


@dataclass
class _TDState:
    """What each task of :func:`td_loop_details` shares (sent once to each worker)."""

    params: Params
    grid: ScenarioGrid
    fortran_compat: bool
    runs: dict[tuple[float | None, float | None], Params] = field(default_factory=dict)


def _td_rows(state: _TDState, task: _Scenario) -> list[dict[str, Any]]:
    """Return the rows of :func:`td_loop_details` for one scenario."""
    stress, kappa0, magnitude, rjb, h, rrup = task
    grid = state.grid
    run = state.runs.get((stress, kappa0))
    if run is None:
        run = params_for(
            state.params, kappa0=kappa0, stress=stress, fortran_compat=state.fortran_compat
        )
        state.runs[(stress, kappa0)] = run
    result = td_motion(run, magnitude, rrup, grid.periods, grid.damping)
    terms = result.terms
    stress_value = terms.source.stress
    scenario = {
        "magnitude": magnitude,
        "rjb": rjb,
        "h": h,
        "rrup": rrup,
        "f_ff": terms.finite_fault_h,
        "rps": terms.rps,
        "kappa0": run.site.kappa0,
        "stress": math.nan if stress_value is None else stress_value,
        "fa": terms.source.fa,
        "fb": terms.source.fb,
        "dur_source": result.dur_source,
        "dur_path": result.dur_path,
        "dur_ex": result.grid.dur_ex,
        "te": result.grid.te,
        "npts": result.grid.npts,
        "nsims": result.nsims,
    }

    measures: list[tuple[str, float, Statistics]] = []
    for measure in _MEASURES:
        stats = result.statistics(measure)
        assert isinstance(stats, Statistics)
        measures.append((measure, math.nan, stats))
    psa = result.statistics("psa")
    assert isinstance(psa, list)
    measures += [("psa", period, stats) for period, stats in zip(grid.periods, psa, strict=True)]
    return [
        {
            "measure": measure,
            "period": period,
            "damping": grid.damping if measure == "psa" else math.nan,
            **scenario,
            "mean": stats.mean,
            "std": stats.std,
            "gmean": stats.gmean,
            "gfactor": stats.gfactor,
        }
        for measure, period, stats in measures
    ]


def td_loop_details(
    params: Params,
    grid: ScenarioGrid,
    *,
    fortran_compat: bool = False,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Time-domain simulations of a grid: the statistics of every measure.

    Parameters
    ----------
    params : Params
        Model parameters; ``params.td`` sets the simulations.
    grid : ScenarioGrid
        Magnitudes, distances, periods, damping, whether to report PGV and
        PGA, the Rjb-to-Rrup conversion and optional kappa_0 and stress
        values.
    fortran_compat : bool
        Make kappa independent of magnitude even when the grid does not
        give kappa_0, as SMSIM's loop drivers do (see
        :func:`~pysmsim.drivers.loop_fas.params_for`).
    workers : int or None
        Number of processes that share the scenarios: 1 (the default)
        computes in this process, None or 0 uses one per CPU. The results
        are the same whatever the number, since every scenario starts from
        the same seed; starting the processes takes a second or two, so
        more than one pays off only for grids that take longer than that.

    Returns
    -------
    pandas.DataFrame
        One row per scenario and measure (PGV, PGA, PGD, Arias, the two
        durations, then PSA at each period), with the columns
        :data:`TD_DETAIL_COLUMNS`. The scenarios are in the order stress,
        kappa_0, magnitude, distance.
    """
    state = _TDState(params, grid, fortran_compat)
    blocks = map_tasks(_td_rows, state, list(_scenarios(grid)), workers)
    rows = [row for block in blocks for row in block]
    return pd.DataFrame(rows, columns=list(TD_DETAIL_COLUMNS))


def td_table(details: pd.DataFrame, grid: ScenarioGrid) -> pd.DataFrame:
    """One row per scenario, with the arithmetic means of the simulations.

    Parameters
    ----------
    details : pandas.DataFrame
        A table from :func:`td_loop_details`.
    grid : ScenarioGrid
        Its grid, which says whether PGV and PGA are wanted.

    Returns
    -------
    pandas.DataFrame
        The columns :data:`TD_SCENARIO_COLUMNS`, then ``pgv`` (cm/s) and
        ``pga`` (cm/s^2) (those the grid asks for), ``pgd`` (cm),
        ``arias`` (cm/s), ``dur_75_05`` and ``dur_95_05`` (s) and
        ``psa(T)`` (cm/s^2) for each period.
    """
    label = details.measure.where(details.measure != "psa", details.period.map(psa_column))
    per_scenario = len(_MEASURES) + len(grid.periods)
    scenario = pd.Series(range(len(details)), index=details.index) // per_scenario
    first = details.measure == _MEASURES[0]
    table = details.loc[first, list(TD_SCENARIO_COLUMNS)].reset_index(drop=True)
    means = pd.DataFrame({"scenario": scenario, "label": label, "value": details["mean"]}).pivot(
        index="scenario", columns="label", values="value"
    )
    wanted = [m for m in _MEASURES if (m != "pgv" or grid.pgv) and (m != "pga" or grid.pga)]
    means = means[wanted + [psa_column(t) for t in grid.periods]]
    means.columns.name = None
    table = pd.concat([table, means.reset_index(drop=True)], axis=1)
    table["npts"] = table["npts"].astype("int64")
    table["nsims"] = table["nsims"].astype("int64")
    return pd.DataFrame(table)


def td_loop(
    params: Params,
    grid: ScenarioGrid,
    *,
    fortran_compat: bool = False,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Time-domain peak motions, one row per scenario of a grid.

    Parameters and options as for :func:`td_loop_details`.

    Returns
    -------
    pandas.DataFrame
        The table of :func:`td_table`: arithmetic means of the simulations.
    """
    return td_table(
        td_loop_details(params, grid, fortran_compat=fortran_compat, workers=workers), grid
    )
