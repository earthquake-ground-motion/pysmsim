"""Ratios of peak to RMS motion (peak factors) for random vibration.

SMSIM's ``get_motion`` computes several peak factors from the number of
zero crossings or extrema in the excitation duration and the bandwidth of
the spectrum. The one it reports is :func:`der_kiureghian_1980` (DK80
eq. 2); the others are kept for comparison::

    from pysmsim.rv.peak_factors import der_kiureghian_1980
    pf = der_kiureghian_1980(n_zero_crossings=40.0, bandwidth=0.6)

:func:`zero_crossings` and :func:`extrema` give the numbers of zero
crossings and extrema with SMSIM's lower limits.

References
----------
Cartwright, D. E. and M. S. Longuet-Higgins (1956). The statistical
distribution of the maxima of a random function. Proc. R. Soc. Lond. A
237, 212-232.

Der Kiureghian, A. (1980). Structural response to stationary excitation.
J. Eng. Mech. Div. ASCE 106, 1195-1213.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

__all__ = [
    "EULER_GAMMA",
    "cartwright_longuet_higgins",
    "cartwright_longuet_higgins_asymptotic",
    "der_kiureghian_1980",
    "der_kiureghian_1980_asymptotic",
    "der_kiureghian_1980_cdf",
    "der_kiureghian_1980_fortran",
    "der_kiureghian_1980_pdf",
    "extrema",
    "zero_crossings",
]

EULER_GAMMA = 0.5772
"""Euler's constant, to the four decimals SMSIM uses."""

_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)


def zero_crossings(dominant_frequency: float, duration: float) -> float:
    """Return the number of zero crossings, 2*f20*duration, at least 1.33 (``anz``).

    The lower limit is Toro's (EPRI, 1985), as in SMSIM.
    """
    n = 2.0 * dominant_frequency * duration
    return 1.33 if n <= 1.33 else n


def extrema(extrema_frequency: float, duration: float) -> float:
    """Return the number of extrema, 2*f42*duration, or 1.002 if not above 1 (``ane``)."""
    n = 2.0 * extrema_frequency * duration
    return 1.002 if n <= 1.0 else n


def cartwright_longuet_higgins(n_extrema: float, xi: float, zup: float = 10.0) -> float:
    """Peak factor by eq. 6.8 of Cartwright and Longuet-Higgins (1956).

    sqrt(2) * integral from 0 to zup of 1 - (1 - xi*exp(-z**2))**n_extrema dz
    (Fortran ``cl68_numrcl_int``).

    Parameters
    ----------
    n_extrema : float
        Number of extrema (:func:`extrema`).
    xi : float
        Irregularity factor m2/sqrt(m0*m4).
    zup : float
        Upper limit of the integral (the integrand is negligible beyond
        about 5).
    """

    def integrand(z: float) -> float:
        # 1 - (1 - xi*exp(-z**2))**n, written as in SMSIM (after C. Mueller)
        return -math.expm1(n_extrema * math.log1p(-xi * math.exp(-z * z)))

    value, _ = quad(integrand, 0.0, zup, epsabs=0.0, epsrel=1.0e-10, limit=200)
    return math.sqrt(2.0) * float(value)


def cartwright_longuet_higgins_asymptotic(n_zero_crossings: float, terms: int = 2) -> float:
    """Asymptotic peak factor: sqrt(2*ln(n)), plus 0.5772/sqrt(2*ln(n)).

    Parameters
    ----------
    n_zero_crossings : float
        Number of zero crossings (:func:`zero_crossings`).
    terms : int
        1 or 2 terms of the expansion.
    """
    if terms not in (1, 2):
        raise ValueError("terms must be 1 or 2")
    first = math.sqrt(2.0 * math.log(n_zero_crossings))
    return first if terms == 1 else first + EULER_GAMMA / first


def der_kiureghian_1980_asymptotic(
    n_zero_crossings: float, bandwidth: float, damping: float | None = None
) -> float:
    """Der Kiureghian's (1980) asymptotic peak factor (Fortran ``pk_rms_dk_2``).

    sqrt(2*ln(n')) + 0.5772/sqrt(2*ln(n')), with the number of zero
    crossings n reduced for narrow-band motion: n' = max(2.1, 2*q*n) for
    q <= 0.1, (1.63*q**0.45 - 0.38)*n for q <= 0.69, and n beyond. For an
    oscillator, q is at least its damping.

    Parameters
    ----------
    n_zero_crossings : float
        Number of zero crossings n.
    bandwidth : float
        Bandwidth q (:attr:`~pysmsim.rv.moments.SpectralMoments.bandwidth`).
    damping : float, optional
        Damping of the oscillator, if the motion is an oscillator response.

    Returns
    -------
    float
        The peak factor, or NaN when n' is not above 1.
    """
    q = bandwidth if damping is None else max(bandwidth, damping)
    if q <= 0.10:
        n = max(2.1, 2.0 * q * n_zero_crossings)
    elif q <= 0.69:
        n = (1.63 * q**0.45 - 0.38) * n_zero_crossings
    else:
        n = n_zero_crossings
    if n <= 1.0:
        return math.nan
    first = math.sqrt(2.0 * math.log(n))
    return first + EULER_GAMMA / first


def der_kiureghian_1980_cdf(a: float, n_zero_crossings: float, bandwidth: float) -> float:
    """Probability that the peak factor is below a (eq. 2 of Der Kiureghian, 1980).

    (1 - exp(-a**2/2)) * exp(-n * (1 - exp(-sqrt(pi/2)*q**1.2*a)) / (exp(a**2/2) - 1)).
    """
    if a <= 0.0:
        return 0.0
    c = -math.expm1(-0.5 * a * a)
    b = -math.expm1(-_SQRT_HALF_PI * bandwidth**1.2 * a)
    return c * math.exp(-n_zero_crossings * b * math.exp(-0.5 * a * a) / c)


def der_kiureghian_1980_pdf(a: np.ndarray, n_zero_crossings: float, bandwidth: float) -> np.ndarray:
    """Probability density of the peak factor (derivative of eq. 2; ``DK80eq2PDF``).

    Parameters
    ----------
    a : ndarray
        Peak factors, positive.
    n_zero_crossings : float
        Number of zero crossings n.
    bandwidth : float
        Bandwidth q.
    """
    a = np.asarray(a, dtype=np.float64)
    qe = bandwidth**1.2
    e1 = np.exp(-0.5 * a * a)
    e2 = np.exp(-_SQRT_HALF_PI * qe * a)
    big_a = 1.0 - e1
    big_b = 1.0 - e2
    big_c = big_a / e1
    big_e = np.exp(-n_zero_crossings * big_b / big_c)
    d_a = a * e1
    d_b = _SQRT_HALF_PI * qe * e2
    d_c = a / e1
    d_e = -n_zero_crossings * big_e * (big_c * d_b - big_b * d_c) / big_c**2
    return np.asarray(big_a * d_e + big_e * d_a, dtype=np.float64)


def der_kiureghian_1980(n_zero_crossings: float, bandwidth: float) -> float:
    """Mean peak factor from eq. 2 of Der Kiureghian (1980).

    The mean of the distribution :func:`der_kiureghian_1980_cdf`,
    integral from 0 to infinity of 1 - F(a) da. This is the peak factor
    SMSIM reports (as ``pk_rms_dk80_eq2``); :func:`der_kiureghian_1980_fortran`
    is SMSIM's own approximation of the integral.

    Parameters
    ----------
    n_zero_crossings : float
        Number of zero crossings n (:func:`zero_crossings`).
    bandwidth : float
        Bandwidth q (:attr:`~pysmsim.rv.moments.SpectralMoments.bandwidth`).
    """
    # The distribution's mass lies near sqrt(2*ln(n)); split there
    centre = math.sqrt(2.0 * math.log(max(n_zero_crossings, 1.0)) + 1.0)
    total = 0.0
    for lo, hi in ((0.0, centre), (centre, centre + 10.0)):
        value, _ = quad(
            lambda a: 1.0 - der_kiureghian_1980_cdf(a, n_zero_crossings, bandwidth),
            lo,
            hi,
            epsabs=1.0e-13,
            epsrel=1.0e-11,
            limit=200,
        )
        total += value
    return total


def der_kiureghian_1980_fortran(
    n_zero_crossings: float, bandwidth: float, tol: float
) -> tuple[float, int]:
    """SMSIM's approximation of :func:`der_kiureghian_1980` (``PF_DK80eq2PDF``).

    The mean is a sum of a * pdf(a) * da over a = 2*da ... (n - 2)*da, with
    da = 6/n, for n = 10, 20, 40, ... It stops when the relative change from
    the previous n is below ``tol``, without taking the absolute value, so
    that a sum that decreases stops at once. The integral stops at a = 6.
    The result usually agrees with :func:`der_kiureghian_1980` to about
    1e-4, but can differ by more than 1e-3 when there are very many zero
    crossings.

    Parameters
    ----------
    n_zero_crossings : float
        Number of zero crossings n.
    bandwidth : float
        Bandwidth q.
    tol : float
        Relative change that stops the doubling (Fortran ``eps_int``).

    Returns
    -------
    tuple of float and int
        The peak factor and the final n (``na_dk80_eq2``).
    """
    a_max = 6.0

    def riemann_sum(n: int) -> float:
        da = a_max / n
        a = np.arange(2, n - 1) * da
        return float(np.sum(a * der_kiureghian_1980_pdf(a, n_zero_crossings, bandwidth)) * da)

    n = 10
    previous = riemann_sum(n)
    while True:
        n *= 2
        current = riemann_sum(n)
        if (current - previous) / previous < tol:
            return current, n
        if n > 10_000_000:
            raise RuntimeError("the DK80 peak-factor sum did not converge")
        previous = current
