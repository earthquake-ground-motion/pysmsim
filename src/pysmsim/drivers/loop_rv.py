"""Peak motions for a grid of scenarios (Fortran ``tmrsk_loop_rv_drvr``).

:func:`rv_loop` computes PGV, PGA and response spectra by random vibration
for every combination of stress, kappa_0, magnitude and distance in a
:class:`~pysmsim.scenario.ScenarioGrid`. It returns one row per scenario,
with the ground motions as columns: ``pgv``, ``pga`` and ``psa(T)`` for
each period T (see :func:`rv_table`)::

    from pysmsim.io.config import load_config
    from pysmsim.drivers.loop_rv import rv_loop
    config = load_config("run.toml")
    table = rv_loop(config.params, config.scenarios)
    table[["magnitude", "rjb", "pga", "psa(1)"]]

:func:`rv_loop_details` gives instead one row per scenario and ground-motion
measure, with every intermediate quantity (moments, durations, peak
factors), in the order of SMSIM's default loop (``SKTMR``: stress, kappa_0,
period, magnitude, distance). The drivers only compose
:func:`pysmsim.rv.gm_rv.rv_motion` and the functions it uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pysmsim.drivers._parallel import map_tasks
from pysmsim.drivers.loop_fas import params_for
from pysmsim.params import Params
from pysmsim.rv.duration import OscillatorCorrection, oscillator_correction
from pysmsim.rv.gm_rv import rv_motion
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.distance import rjb_to_rrup_depth
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator

__all__ = ["RV_DETAIL_COLUMNS", "SCENARIO_COLUMNS", "rv_loop", "rv_loop_details", "rv_table"]

RV_DETAIL_COLUMNS = (
    "measure",
    "period",
    "damping",
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
    "fup",
    "dur_source",
    "dur_path",
    "dur_ex",
    "dur_rms",
    "m0",
    "m1",
    "m2",
    "m4",
    "bandwidth",
    "n_zero_crossings",
    "n_extrema",
    "rms",
    "peak_factor",
    "peak_factor_steps",
    "peak_factor_cl",
    "peak_motion_factor",
    "pgv",
    "pga",
    "psv",
    "psa",
    "sd",
)
"""Columns of the table from :func:`rv_loop_details`.

- measure: ``"pgv"``, ``"pga"`` or ``"psa"`` (an oscillator);
- period (s; NaN for PGV and PGA) and damping (of the oscillators);
- magnitude, rjb (km; NaN when the grid gives Rrup), h (km, the depth of
  the Rjb-to-Rrup conversion; NaN when the grid gives Rrup), rrup, f_ff
  (finite-fault factor, km), rps (km), kappa0 (s), stress (bars; NaN for
  source models without one), fa and fb (corner frequencies, Hz);
- fup (upper frequency of the moments, Hz), dur_source, dur_path, dur_ex
  and dur_rms (s), the moments m0, m1, m2 and m4, bandwidth
  (sqrt(1 - m1**2/(m0*m2))), n_zero_crossings and n_extrema;
- rms (the RMS motion), peak_factor (DK80), peak_factor_steps (SMSIM's
  ``na_dk80_eq2`` with ``fortran_compat``, else NaN), peak_factor_cl
  (Cartwright and Longuet-Higgins eq. 6.8, for comparison) and
  peak_motion_factor (the BT15 TD/RV ratio for PGV and PGA, else 1);
- the peak motions in cgs units: pgv (cm/s) and pga (cm/s^2) on the PGV
  and PGA rows; psv (cm/s), psa (cm/s^2) and sd (cm) on the oscillator
  rows; NaN elsewhere, and NaN where a BT15 table
  does not reach the magnitude and distance.
"""


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
    "fup",
    "dur_source",
    "dur_path",
    "dur_ex",
)
"""The columns of :func:`rv_table` that describe each scenario (see
:data:`RV_DETAIL_COLUMNS`); the ground-motion columns follow them."""


def _responses(grid: ScenarioGrid) -> list[tuple[str, GroundMotion | Oscillator]]:
    responses: list[tuple[str, GroundMotion | Oscillator]] = []
    if grid.pgv:
        responses.append(("pgv", GroundMotion(MotionQuantity.VELOCITY)))
    if grid.pga:
        responses.append(("pga", GroundMotion(MotionQuantity.ACCELERATION)))
    for period in grid.periods:
        responses.append(("psa", Oscillator(period=period, damping=grid.damping)))
    return responses


def psa_column(period: float) -> str:
    """Name of the PSA column for a period (s): ``psa(T)``, T with up to 6 digits."""
    return f"psa({period:g})"


def rv_table(details: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario, with the ground motions as columns.

    Parameters
    ----------
    details : pandas.DataFrame
        A table from :func:`rv_loop_details`.

    Returns
    -------
    pandas.DataFrame
        The columns :data:`SCENARIO_COLUMNS`, then ``pgv`` (cm/s), ``pga``
        (cm/s^2) and ``psa(T)`` (cm/s^2) for each period T, in the order of the grid
        (only those the grid asks for). The rows are in the order stress,
        kappa_0, magnitude, distance. NaN where a BT15 table does not reach
        the magnitude and distance.
    """
    label = details.measure.where(details.measure != "psa", details.period.map(psa_column))
    value = np.where(
        details.measure == "pgv",
        details.pgv,
        np.where(details.measure == "pga", details.pga, details.psa),
    )
    # Each measure visits the scenarios in the same order
    scenario = details.groupby(label, sort=False).cumcount()
    first = label == label.iloc[0]
    table = details.loc[first, list(SCENARIO_COLUMNS)].reset_index(drop=True)
    motions = pd.DataFrame({"scenario": scenario, "label": label, "value": value}).pivot(
        index="scenario", columns="label", values="value"
    )
    motions = motions[list(dict.fromkeys(label))]
    motions.columns.name = None
    return pd.DataFrame(pd.concat([table, motions.reset_index(drop=True)], axis=1))


@dataclass
class _RVState:
    """What every task of :func:`rv_loop_details` shares (sent once to each worker)."""

    params: Params
    grid: ScenarioGrid
    fortran_compat: bool
    runs: dict[tuple[float | None, float | None], Params] = field(default_factory=dict)
    correction: OscillatorCorrection | None = None


def _rv_rows(
    state: _RVState, task: tuple[float | None, float | None, int, float]
) -> list[dict[str, Any]]:
    """Return the rows of one stress, kappa_0, measure and magnitude, over the distances."""
    stress, kappa0, index, magnitude = task
    grid = state.grid
    fortran_compat = state.fortran_compat
    run = state.runs.get((stress, kappa0))
    if run is None:
        run = params_for(state.params, kappa0=kappa0, stress=stress, fortran_compat=fortran_compat)
        state.runs[(stress, kappa0)] = run
    if state.correction is None:
        state.correction = oscillator_correction(state.params.rv)
    correction = state.correction
    measure, response = _responses(grid)[index]
    rows: list[dict[str, Any]] = []
    by_rjb = grid.distance_metric is DistanceMetric.RJB
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
        rrup = math.sqrt(rjb**2 + h**2) if by_rjb else distance
        result = rv_motion(
            run,
            magnitude,
            rrup,
            response,
            correction=correction,
            fortran_compat=fortran_compat,
        )
        terms = result.terms
        stress_value = terms.source.stress
        row: dict[str, Any] = {
            "measure": measure,
            "period": math.nan,
            "damping": math.nan,
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
            "fup": result.fup,
            "dur_source": result.dur_source,
            "dur_path": result.dur_path,
            "dur_ex": result.dur_ex,
            "dur_rms": result.dur_rms,
            "m0": result.moments.m0,
            "m1": result.moments.m1,
            "m2": result.moments.m2,
            "m4": result.moments.m4,
            "bandwidth": result.moments.bandwidth,
            "n_zero_crossings": result.n_zero_crossings,
            "n_extrema": result.n_extrema,
            "rms": result.rms,
            "peak_factor": result.peak_factor,
            "peak_factor_steps": (
                math.nan if result.peak_factor_steps is None else float(result.peak_factor_steps)
            ),
            "peak_factor_cl": result.peak_factor_cl,
            "peak_motion_factor": result.peak_motion_factor,
        }
        peak = result.peak
        if isinstance(response, Oscillator):
            row["period"] = response.period
            row["damping"] = response.damping
            row["psv"] = peak
            row["psa"] = 2.0 * math.pi / response.period * peak
            row["sd"] = response.period / (2.0 * math.pi) * peak
        elif measure == "pgv":
            row["pgv"] = peak
        else:
            row["pga"] = peak
        rows.append(row)
    return rows


def rv_loop(
    params: Params,
    grid: ScenarioGrid,
    *,
    fortran_compat: bool = False,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Random-vibration peak motions, one row per scenario of a grid.

    Parameters and options as for :func:`rv_loop_details`.

    Returns
    -------
    pandas.DataFrame
        The table of :func:`rv_table`: the scenario, then ``pgv`` (cm/s),
        ``pga`` (cm/s^2) and ``psa(T)`` (cm/s^2) for each period.
    """
    return rv_table(rv_loop_details(params, grid, fortran_compat=fortran_compat, workers=workers))


def rv_loop_details(
    params: Params,
    grid: ScenarioGrid,
    *,
    fortran_compat: bool = False,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Random-vibration peak motions and intermediate quantities for a grid.

    Parameters
    ----------
    params : Params
        Model parameters.
    grid : ScenarioGrid
        Magnitudes, distances, periods, damping, whether to compute PGV and
        PGA, the Rjb-to-Rrup conversion and optional kappa_0 and stress
        values.
    fortran_compat : bool
        Make kappa independent of magnitude even when the grid does not
        give kappa_0 (see :func:`~pysmsim.drivers.loop_fas.params_for`),
        compute fup from kappa_0, and compute the DK80 peak factor with
        SMSIM's sum (see :func:`~pysmsim.rv.gm_rv.rv_motion`).
    workers : int or None
        Number of processes to share the scenarios: 1 (the default)
        computes them in this process, None or 0 uses one per CPU. The
        results are the same; starting the processes takes a second or two,
        so more than one pays off only for large grids.

    Returns
    -------
    pandas.DataFrame
        One row per scenario and measure, with the columns
        :data:`RV_DETAIL_COLUMNS`.
    """
    state = _RVState(params=params, grid=grid, fortran_compat=fortran_compat)
    tasks = [
        (stress, kappa0, index, magnitude)
        for stress in grid.stress or (None,)
        for kappa0 in grid.kappa0 or (None,)
        for index in range(len(_responses(grid)))
        for magnitude in grid.magnitudes
    ]
    rows = [row for block in map_tasks(_rv_rows, state, tasks, workers) for row in block]
    table = pd.DataFrame(rows).reindex(columns=list(RV_DETAIL_COLUMNS))
    numeric = [c for c in RV_DETAIL_COLUMNS if c != "measure"]
    table[numeric] = table[numeric].astype(np.float64)
    return table
