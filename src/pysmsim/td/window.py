"""The time grid and the window of a time-domain simulation.

:func:`time_grid` sets the length of the series and where the motion
starts and stops (Fortran ``get_npts_for_td_calcs``); :func:`window` is
the box or Saragoni-Hart exponential window applied to the noise (Fortran
``wind_box`` and ``wind_exp``)::

    from pysmsim.td.window import time_grid, window
    grid = time_grid(params.td, params.filters.low_cut, dur_source, dur_path)
    w = window(grid, params.td)        # npts values, zero outside the motion

Indices are 0-based: the motion occupies ``nstart`` to ``nstop``
inclusive (Fortran ``nstart - 1`` to ``nstop - 1``). ``grid.times()``
gives the time of each point (s), for plotting a series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pysmsim.params import TDConfig, WindowShape
from pysmsim.spectrum.filters import LowCutFilter

__all__ = ["TimeGrid", "exponential_window_coefficients", "time_grid", "window"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class TimeGrid:
    """Length and timing of the series of one scenario.

    Attributes
    ----------
    dt : float
        Time step (s).
    npts : int
        Number of points, a power of two (Fortran ``npw2``).
    nstart, nstop : int
        First and last index of the windowed noise (0-based, inclusive).
    ntaper : int
        Box window: points in each cosine taper (0 for the exponential
        window).
    dur_ex : float
        Excitation duration, source plus path (s).
    te, tw : float
        Exponential window: te = f_tb2te * dur_ex, where the window falls to
        eta_w of its peak, and the window length tw = f_te_xtnd * te (s);
        0 for the box window.
    tshift : float
        Time before the motion (s), raised to the low-cut pad when longer.
    tpad : float
        Low-cut filter pad, 1.5 * (order/2) / fcut (s), or 0.
    """

    dt: float
    npts: int
    nstart: int
    nstop: int
    ntaper: int
    dur_ex: float
    te: float
    tw: float
    tshift: float
    tpad: float

    @property
    def nmotion(self) -> int:
        """Number of points of the windowed noise."""
        return self.nstop - self.nstart + 1

    def times(self) -> FloatArray:
        """Return the time of each point of the series (s), from 0.

        ``npts`` values ``0, dt, 2*dt, ...``, the times of the acceleration
        series of :func:`pysmsim.td.acc_ts.acc_ts`; the motion starts at
        ``times()[nstart]``, about ``tshift``::

            import matplotlib.pyplot as plt
            result = td_motion(params, 6.0, 20.0, nsims=1, keep_series=True)
            plt.plot(result.grid.times(), result.acc[0])
        """
        return np.arange(self.npts, dtype=np.float64) * self.dt


def _power_of_two_at_least(x: float) -> int:
    n = 1
    while n < x:
        n *= 2
    return n


def time_grid(td: TDConfig, low_cut: LowCutFilter, dur_source: float, dur_path: float) -> TimeGrid:
    """Return the time grid of a scenario (Fortran ``get_npts_for_td_calcs``).

    Parameters
    ----------
    td : TDConfig
        Window and timing settings.
    low_cut : LowCutFilter
        The low-cut filter, which sets the pad before the motion.
    dur_source, dur_path : float
        Source and path durations (s).

    Returns
    -------
    TimeGrid

    Raises
    ------
    ValueError
        If the motion is shorter than one time step.

    Notes
    -----
    As in SMSIM, the number of points of the motion is truncated
    (``nmotion = int(duration/dt)``); the box window adds ``int(taper *
    nmotion)`` points for its tapers. The series then holds ``tshift``,
    ``dur_fctr`` times the motion and the filter pad, rounded up to a power
    of two.
    """
    dt = td.dt
    dur_ex = dur_source + dur_path
    if td.window is WindowShape.BOX:
        if dur_ex < dt:
            raise ValueError(f"the excitation duration {dur_ex:g} s is shorter than dt")
        nmotion = int(dur_ex / dt)
        ntaper = int(td.taper * nmotion)
        nmotion += ntaper
        te = tw = 0.0
    else:
        te = td.f_tb2te * dur_ex
        tw = td.f_te_xtnd * te
        if tw < dt:
            raise ValueError(f"the window length {tw:g} s is shorter than dt")
        nmotion = int(tw / dt)
        ntaper = 0
    tpad = 0.0 if low_cut.fcut == 0.0 else 1.5 * (low_cut.order // 2) / low_cut.fcut
    tshift = max(td.tshift, tpad)
    nstart = int(tshift / dt)  # Fortran nstart = tshift/dt + 1, 1-based
    nstop = nstart + nmotion
    tsimdur = (nstart + 1) * dt + td.dur_fctr * nmotion * dt + tpad
    npts = _power_of_two_at_least(tsimdur / dt + 1.0)
    return TimeGrid(
        dt=dt,
        npts=npts,
        nstart=nstart,
        nstop=nstop,
        ntaper=ntaper,
        dur_ex=dur_ex,
        te=te,
        tw=tw,
        tshift=tshift,
        tpad=tpad,
    )


def exponential_window_coefficients(eps_w: float, eta_w: float) -> tuple[float, float, float]:
    """Return a, b, c of the window a * (t/te)**b * exp(-c * t/te).

    The window peaks at t = eps_w * te with the value 1 and is eta_w at te
    (Boore, 1983, p. 1869).
    """
    b = -eps_w * math.log(eta_w) / (1.0 + eps_w * (math.log(eps_w) - 1.0))
    c = b / eps_w
    a = (math.e / eps_w) ** b
    return a, b, c


def window(grid: TimeGrid, td: TDConfig) -> FloatArray:
    """Return the window on the whole series (Fortran ``wind_box``, ``wind_exp``).

    Parameters
    ----------
    grid : TimeGrid
        The time grid of the scenario.
    td : TDConfig
        Window settings.

    Returns
    -------
    ndarray
        ``grid.npts`` values, zero outside ``nstart`` to ``nstop``.
    """
    w = np.zeros(grid.npts)
    i = np.arange(grid.nstart, grid.nstop + 1, dtype=np.float64)
    if td.window is WindowShape.BOX:
        values = np.ones_like(i)
        if grid.ntaper > 0:
            mid = (grid.nstop + grid.nstart) / 2.0
            half = (grid.nstop - grid.nstart - grid.ntaper) / 2.0
            taper = (i < grid.nstart + grid.ntaper) | (i > grid.nstop - grid.ntaper)
            values[taper] = 0.5 * (
                1.0 - np.sin(np.pi * (np.abs(i[taper] - mid) - half) / grid.ntaper)
            )
    else:
        a, b, c = exponential_window_coefficients(td.eps_w, td.eta_w)
        x = (i - grid.nstart) * grid.dt / grid.te
        values = a * x**b * np.exp(-c * x)
    w[grid.nstart : grid.nstop + 1] = values
    return w
