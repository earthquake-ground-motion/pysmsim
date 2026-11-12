"""Velocity, displacement, Arias intensity and significant durations.

The functions work along the last axis, so they take one series or a
stack of them (one per row), as SMSIM's ``acc2vd``, ``accsqint`` and
``locate`` do for one::

    from pysmsim.td.intensity import arias_intensity, cumulative_squared, ...
    vel, dis = velocity_displacement(acc, dt)
    a2 = cumulative_squared(acc, dt)
    arias = arias_intensity(a2[..., -1])
    d575 = significant_duration(a2, dt, 0.05, 0.75)

:func:`series_measures` gives all the measures of a stack of series at once
(PGA, PGV, PGD, Arias intensity and the 5-75% and 5-95% durations) without
storing the velocity, displacement or cumulative integral. It is a Numba
kernel, parallel over the series, that does the same arithmetic in the same
order as the functions above, so it gives identical results;
:func:`series_measures_numpy` is its NumPy twin.
"""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

__all__ = [
    "SERIES_MEASURES",
    "G",
    "arias_intensity",
    "cumulative_squared",
    "integrate",
    "series_measures",
    "series_measures_numpy",
    "significant_duration",
    "velocity_displacement",
]

FloatArray = npt.NDArray[np.float64]

G = 981.0
"""Acceleration of gravity (cm/s^2), as in SMSIM's Arias intensity."""


def integrate(y: FloatArray, dt: float) -> FloatArray:
    """Return the cumulative trapezoidal integral, 0 at the first point.

    Fortran ``Trapezoidal_Integration`` with a zero initial value.
    """
    out = np.zeros_like(y)
    np.cumsum((y[..., :-1] + y[..., 1:]) * (0.5 * dt), axis=-1, out=out[..., 1:])
    return out


def velocity_displacement(acc: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
    """Velocity and displacement, from rest (Fortran ``acc2vd`` without detrending)."""
    vel = integrate(acc, dt)
    return vel, integrate(vel, dt)


def cumulative_squared(acc: FloatArray, dt: float) -> FloatArray:
    """Cumulative integral of acc**2, with acc linear between samples (Fortran ``accsqint``)."""
    a1 = acc[..., :-1]
    a2 = acc[..., 1:]
    out = np.zeros_like(acc)
    np.cumsum((a1 * a1 + a1 * a2 + a2 * a2) * (dt / 3.0), axis=-1, out=out[..., 1:])
    return out


def arias_intensity(integral_of_squared: npt.ArrayLike) -> FloatArray:
    """Arias intensity pi/(2g) * integral of acc**2 (cm/s)."""
    return np.asarray(integral_of_squared) * (math.pi / (2.0 * G))


def significant_duration(
    cumulative: FloatArray, dt: float, start: float, stop: float
) -> FloatArray:
    """Time between ``start`` and ``stop`` fractions of the total of a cumulative integral.

    Each fraction is located as Numerical Recipes ``locate`` does in SMSIM:
    at the last sample where the cumulative integral does not exceed it.
    """
    total = cumulative[..., -1:]
    j_start = np.count_nonzero(cumulative <= start * total, axis=-1)
    j_stop = np.count_nonzero(cumulative <= stop * total, axis=-1)
    return np.asarray((j_stop - j_start) * dt, dtype=np.float64)


SERIES_MEASURES = ("pga", "pgv", "pgd", "arias", "dur_75_05", "dur_95_05")
"""The measures of :func:`series_measures`, in order."""


@numba.njit(cache=True, parallel=True)
def _series_measures(acc: FloatArray, dt: float) -> FloatArray:  # pragma: no cover - compiled
    nseries, n = acc.shape
    out = np.empty((nseries, 6))
    half = 0.5 * dt
    third = dt / 3.0
    for s in numba.prange(nseries):
        # Running trapezoidal integrals, as the cumulative sums of integrate
        # and cumulative_squared add their terms
        a_prev = acc[s, 0]
        vel = 0.0
        dis = 0.0
        sq = 0.0
        pga = abs(a_prev)
        pgv = 0.0
        pgd = 0.0
        for i in range(1, n):
            a = acc[s, i]
            vel_next = vel + (a_prev + a) * half
            dis = dis + (vel + vel_next) * half
            sq = sq + (a_prev * a_prev + a_prev * a + a * a) * third
            vel = vel_next
            pga = max(pga, abs(a))
            pgv = max(pgv, abs(vel))
            pgd = max(pgd, abs(dis))
            a_prev = a
        # Count the samples where the integral of acc**2 does not exceed each
        # fraction of its total, as significant_duration does (the first
        # sample is 0)
        c05 = 0.05 * sq
        c75 = 0.75 * sq
        c95 = 0.95 * sq
        n05 = 1 if c05 >= 0.0 else 0
        n75 = 1 if c75 >= 0.0 else 0
        n95 = 1 if c95 >= 0.0 else 0
        cum = 0.0
        a_prev = acc[s, 0]
        for i in range(1, n):
            a = acc[s, i]
            cum = cum + (a_prev * a_prev + a_prev * a + a * a) * third
            n05 += cum <= c05
            n75 += cum <= c75
            n95 += cum <= c95
            a_prev = a
        out[s, 0] = pga
        out[s, 1] = pgv
        out[s, 2] = pgd
        out[s, 3] = sq
        out[s, 4] = (n75 - n05) * dt
        out[s, 5] = (n95 - n05) * dt
    return out


def series_measures(acc: npt.ArrayLike, dt: float) -> dict[str, FloatArray]:
    """PGA, PGV, PGD, Arias intensity and significant durations of series.

    Parameters
    ----------
    acc : array_like
        Acceleration (cm/s^2): one series, or one per row.
    dt : float
        Time step (s).

    Returns
    -------
    dict
        For each name in :data:`SERIES_MEASURES`: PGA (cm/s^2), PGV (cm/s)
        and PGD (cm) from rest, the Arias intensity (cm/s) and the 5-75% and
        5-95% significant durations (s); a number per series (an array of
        shape ``acc.shape[:-1]``).
    """
    a = np.ascontiguousarray(acc, dtype=np.float64)
    if a.shape[-1] < 2:
        raise ValueError("a series needs at least two points")
    values = _series_measures(a.reshape(-1, a.shape[-1]), float(dt))
    values[:, 3] *= math.pi / (2.0 * G)
    shape = a.shape[:-1]
    return {name: values[:, k].reshape(shape) for k, name in enumerate(SERIES_MEASURES)}


def series_measures_numpy(acc: npt.ArrayLike, dt: float) -> dict[str, FloatArray]:
    """Return what :func:`series_measures` does, with the NumPy functions above."""
    a = np.asarray(acc, dtype=np.float64)
    vel, dis = velocity_displacement(a, dt)
    a2 = cumulative_squared(a, dt)
    return {
        "pga": np.abs(a).max(axis=-1),
        "pgv": np.abs(vel).max(axis=-1),
        "pgd": np.abs(dis).max(axis=-1),
        "arias": arias_intensity(a2[..., -1]),
        "dur_75_05": significant_duration(a2, dt, 0.05, 0.75),
        "dur_95_05": significant_duration(a2, dt, 0.05, 0.95),
    }
