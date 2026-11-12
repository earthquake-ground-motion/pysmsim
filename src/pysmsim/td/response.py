"""Peak oscillator responses of acceleration series (Fortran ``rdrvaa``).

:func:`oscillator_displacement` gives the peak relative displacement of
single-degree-of-freedom oscillators, by the exact solution for an
acceleration that is linear between samples (Nigam and Jennings, 1969),
as SMSIM's ``rs_calc`` does without interpolation::

    from pysmsim.td.response import oscillator_displacement
    sd = oscillator_displacement(acc, periods, damping=0.05, dt=0.005)
    psa = (2 * np.pi / periods) ** 2 * sd

The work is a Numba kernel, parallel over the series;
:func:`oscillator_displacement_numpy` is its NumPy twin.
"""

from __future__ import annotations

import math

import numba
import numpy as np
import numpy.typing as npt

__all__ = ["oscillator_displacement", "oscillator_displacement_numpy"]

FloatArray = npt.NDArray[np.float64]


def _coefficients(omega: float, damping: float, dt: float) -> tuple[float, ...]:
    """Return the coefficients a11, a12, a21, a22, b11, b12, b21, b22 of ``rdrvaa``."""
    d2 = math.sqrt(1.0 - damping * damping)
    bom = damping * omega
    omd = omega * d2
    om2 = omega * omega
    c1 = 1.0 / om2
    omt = omega * dt
    omdt = omd * dt
    c2 = 2.0 * damping / (om2 * omt)
    c3 = c1 + c2
    c4 = 1.0 / (omega * omt)
    ee = math.exp(-damping * omt)
    ss = math.sin(omdt) * ee
    cc = math.cos(omdt) * ee
    s1 = ss / omd
    s2 = s1 * bom
    s3 = s2 + cc
    s4 = c4 * (1.0 - s3)
    s5 = s1 * c4 + c2
    return (s3, s1, -om2 * s1, cc - s2, s3 * c3 - s5, -c2 * s3 + s5 - c1, -s1 + s4, -s4)


_coefficients_jit = numba.njit(cache=True)(_coefficients)


@numba.njit(cache=True, parallel=True)
def _peak_displacement(
    acc: FloatArray, omegas: FloatArray, damping: float, dt: float
) -> FloatArray:
    nseries, npts = acc.shape
    nper = omegas.shape[0]
    out = np.zeros((nseries, nper))
    for k in range(nper):
        a11, a12, a21, a22, b11, b12, b21, b22 = _coefficients_jit(omegas[k], damping, dt)
        for s in numba.prange(nseries):
            y = 0.0
            ydot = 0.0
            peak = 0.0
            for i in range(npts - 1):
                y1 = a11 * y + a12 * ydot + b11 * acc[s, i] + b12 * acc[s, i + 1]
                ydot = a21 * y + a22 * ydot + b21 * acc[s, i] + b22 * acc[s, i + 1]
                y = y1
                if abs(y) > peak:
                    peak = abs(y)
            out[s, k] = peak
    return out


def _check(acc: npt.ArrayLike, periods: npt.ArrayLike) -> tuple[FloatArray, FloatArray, bool]:
    a = np.ascontiguousarray(acc, dtype=np.float64)
    single = a.ndim == 1
    a = a.reshape(1, -1) if single else a.reshape(-1, a.shape[-1])
    omegas = 2.0 * np.pi / np.atleast_1d(np.asarray(periods, dtype=np.float64))
    return a, omegas, single


def oscillator_displacement(
    acc: npt.ArrayLike, periods: npt.ArrayLike, damping: float, dt: float
) -> FloatArray:
    """Peak relative displacement of oscillators (cm), Fortran ``rdrvaa``.

    Parameters
    ----------
    acc : array_like
        Acceleration (cm/s^2): one series, or one per row.
    periods : array_like
        Oscillator periods (s).
    damping : float
        Fraction of critical damping.
    dt : float
        Time step (s).

    Returns
    -------
    ndarray
        Shape ``(nperiods,)`` for one series, else ``(nseries, nperiods)``.
    """
    a, omegas, single = _check(acc, periods)
    out = _peak_displacement(a, omegas, float(damping), float(dt))
    return out[0] if single else out


def oscillator_displacement_numpy(
    acc: npt.ArrayLike, periods: npt.ArrayLike, damping: float, dt: float
) -> FloatArray:
    """Return what :func:`oscillator_displacement` does, with NumPy.

    The same recurrence, vectorised over the series and periods.
    """
    a, omegas, single = _check(acc, periods)
    coef = np.array([_coefficients(float(w), damping, dt) for w in omegas]).T[:, np.newaxis, :]
    a11, a12, a21, a22, b11, b12, b21, b22 = coef
    y = np.zeros((a.shape[0], omegas.size))
    ydot = np.zeros_like(y)
    peak = np.zeros_like(y)
    for i in range(a.shape[1] - 1):
        u0 = a[:, i : i + 1]
        u1 = a[:, i + 1 : i + 2]
        y, ydot = (
            a11 * y + a12 * ydot + b11 * u0 + b12 * u1,
            a21 * y + a22 * ydot + b21 * u0 + b22 * u1,
        )
        np.maximum(peak, np.abs(y), out=peak)
    return peak[0] if single else peak
