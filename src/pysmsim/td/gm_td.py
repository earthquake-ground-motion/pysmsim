"""Time-domain simulations of one scenario (Fortran ``gm_td``).

:func:`td_motion` simulates ``params.td.nsims`` acceleration series for one
magnitude and distance, and returns the peak ground acceleration,
velocity and displacement, the Arias intensity, the 5-75% and 5-95%
significant durations and the oscillator responses of each simulation::

    from pysmsim.td.gm_td import td_motion
    result = td_motion(params, magnitude=6.0, rrup=20.0, periods=[0.1, 1.0])
    result.pga.mean(), result.psa[:, 1]          # one value per simulation
    result.statistics("pga")                     # SMSIM's averages

As in SMSIM, the random-number generator starts again from
``params.td.seed`` for every scenario, so all scenarios of a run share the
same noise (each uses as many values as its motion has points).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pysmsim.params import Params
from pysmsim.spectrum.fas import ScenarioTerms, scenario_terms
from pysmsim.td.acc_ts import acc_ts, target_spectrum
from pysmsim.td.intensity import SERIES_MEASURES, series_measures
from pysmsim.td.noise import white_noise
from pysmsim.td.response import oscillator_displacement
from pysmsim.td.window import TimeGrid, time_grid, window

__all__ = ["MEASURES", "Statistics", "TDResult", "statistics", "td_motion"]

FloatArray = npt.NDArray[np.float64]

MEASURES = SERIES_MEASURES
"""The measures of every simulation besides the oscillator responses."""

_CHUNK_POINTS = 1 << 21
"""Simulations are computed in groups of at most this many points."""


@dataclass(frozen=True, slots=True)
class Statistics:
    """Averages over the simulations (Fortran ``avg_std``).

    Attributes
    ----------
    mean : float
        Arithmetic mean.
    std : float
        Standard deviation (with n - 1).
    gmean : float
        Geometric mean.
    gfactor : float
        Geometric standard deviation as a factor: 10**std(log10 x).
    """

    mean: float
    std: float
    gmean: float
    gfactor: float


def statistics(values: npt.ArrayLike) -> Statistics:
    """Arithmetic and geometric means and spreads of positive values."""
    x = np.asarray(values, dtype=np.float64)
    ddof = 1 if x.size > 1 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log10(x)
    return Statistics(
        mean=float(x.mean()),
        std=float(x.std(ddof=ddof)),
        gmean=float(10.0 ** logs.mean()),
        gfactor=float(10.0 ** logs.std(ddof=ddof)),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TDResult:
    """The simulations of one scenario.

    Attributes
    ----------
    terms : ScenarioTerms
        M0, corner frequencies, Rps and the other frequency-independent terms.
    grid : TimeGrid
        Length and timing of the series.
    dur_source, dur_path : float
        Source and path durations (s).
    periods : ndarray
        Oscillator periods (s).
    damping : float
        Oscillator damping.
    pga, pgv, pgd : ndarray
        Peak acceleration (cm/s^2), velocity (cm/s) and displacement (cm)
        of each simulation.
    arias : ndarray
        Arias intensity (cm/s).
    dur_75_05, dur_95_05 : ndarray
        Significant durations (s).
    sd : ndarray
        Peak oscillator displacement (cm), shape ``(nsims, nperiods)``.
    acc : ndarray or None
        The acceleration series, shape ``(nsims, npts)``, when kept.
    """

    terms: ScenarioTerms
    grid: TimeGrid
    dur_source: float
    dur_path: float
    periods: FloatArray
    damping: float
    pga: FloatArray
    pgv: FloatArray
    pgd: FloatArray
    arias: FloatArray
    dur_75_05: FloatArray
    dur_95_05: FloatArray
    sd: FloatArray
    acc: FloatArray | None = None

    @property
    def nsims(self) -> int:
        """Number of simulations."""
        return int(self.pga.size)

    @property
    def psv(self) -> FloatArray:
        """Pseudo-velocity 2*pi/T * SD (cm/s), shape ``(nsims, nperiods)``."""
        return self.sd * (2.0 * np.pi / self.periods)

    @property
    def psa(self) -> FloatArray:
        """Pseudo-acceleration (2*pi/T)**2 * SD (cm/s^2), shape ``(nsims, nperiods)``."""
        return self.sd * (2.0 * np.pi / self.periods) ** 2

    def statistics(self, measure: str) -> Statistics | list[Statistics]:
        """Return the averages of a measure.

        ``measure`` is one of :data:`MEASURES`, or ``"psa"``, ``"psv"`` or
        ``"sd"`` (then a list, one per period).
        """
        if measure in ("psa", "psv", "sd"):
            values: FloatArray = getattr(self, measure)
            return [statistics(values[:, k]) for k in range(values.shape[1])]
        if measure not in MEASURES:
            raise ValueError(f"unknown measure {measure!r}")
        return statistics(getattr(self, measure))


def td_motion(
    params: Params,
    magnitude: float,
    rrup: float,
    periods: Sequence[float] | npt.ArrayLike = (),
    damping: float = 0.05,
    *,
    nsims: int | None = None,
    rng: np.random.Generator | None = None,
    keep_series: bool = False,
) -> TDResult:
    """Simulate the ground motion of one scenario (Fortran ``gm_td``).

    Parameters
    ----------
    params : Params
        Model parameters; ``params.td`` gives the window, the timing, the
        noise, the seed and the number of simulations.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    periods : sequence of float
        Oscillator periods (s); none by default.
    damping : float
        Oscillator damping (fraction of critical).
    nsims : int, optional
        Number of simulations (default ``params.td.nsims``).
    rng : numpy.random.Generator, optional
        Source of the noise (default: a new generator seeded with
        ``params.td.seed``).
    keep_series : bool
        Keep the acceleration series in the result (memory: nsims * npts
        values).

    Returns
    -------
    TDResult
    """
    td = params.td
    nsims = td.nsims if nsims is None else nsims
    if nsims < 1:
        raise ValueError("nsims must be at least 1")
    if rng is None:
        rng = np.random.default_rng(td.seed)
    periods_arr = np.atleast_1d(np.asarray(periods, dtype=np.float64))
    if np.any(periods_arr <= 0):
        raise ValueError("oscillator periods must be positive")

    terms = scenario_terms(params, magnitude, rrup)
    dur_source = params.source.duration(terms.source)
    dur_path = params.path_duration.duration(terms.rps)
    grid = time_grid(td, params.filters.low_cut, dur_source, dur_path)
    w = window(grid, td)
    spectrum = target_spectrum(grid, params, magnitude, rrup)
    dt = grid.dt

    out = {name: np.empty(nsims) for name in MEASURES}
    sd = np.empty((nsims, periods_arr.size))
    kept = np.empty((nsims, grid.npts)) if keep_series else None
    chunk = max(1, _CHUNK_POINTS // grid.npts)
    for first in range(0, nsims, chunk):
        rows = slice(first, min(first + chunk, nsims))
        count = rows.stop - first
        noise = white_noise(rng, (count, grid.nmotion), td.noise)
        acc = acc_ts(noise, grid, w, spectrum)
        for name, values in series_measures(acc, dt).items():
            out[name][rows] = values
        if periods_arr.size:
            sd[rows] = oscillator_displacement(acc, periods_arr, damping, dt)
        if kept is not None:
            kept[rows] = acc
    if not math.isfinite(float(out["pga"].sum())):  # pragma: no cover - a bad spectrum
        raise ValueError("the simulated motions are not finite")
    return TDResult(
        terms=terms,
        grid=grid,
        dur_source=dur_source,
        dur_path=dur_path,
        periods=periods_arr,
        damping=damping,
        sd=sd,
        acc=kept,
        **out,
    )
