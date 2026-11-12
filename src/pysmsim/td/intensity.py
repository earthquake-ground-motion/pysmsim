"""Velocity, displacement, Arias intensity and significant durations.

The functions work along the last axis, so they take one series or a
stack of them (one per row), as SMSIM's ``acc2vd``, ``accsqint`` and
``locate`` do for one::

    from pysmsim.td.intensity import arias_intensity, cumulative_squared, ...
    vel, dis = velocity_displacement(acc, dt)
    a2 = cumulative_squared(acc, dt)
    arias = arias_intensity(a2[..., -1])
    d575 = significant_duration(a2, dt, 0.05, 0.75)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "G",
    "arias_intensity",
    "cumulative_squared",
    "integrate",
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
