"""Spectral moments of a Fourier amplitude spectrum (Fortran ``amom_rv``).

The random-vibration method uses the moments

    m_k = 2 * integral from 0 to fup of A(f)**2 * (2*pi*f)**k df

of the Fourier amplitude spectrum A(f), for k = 0, 1, 2 and 4. SMSIM
integrates each one with Numerical Recipes' ``odeint`` up to the frequency
``fup`` given by :func:`upper_frequency`. Here all four come from one
adaptive Gauss-Kronrod integration (:func:`integrate`), which evaluates the
spectrum on arrays of frequencies, so that each step is one call of the
compiled spectrum kernels::

    from pysmsim.rv.moments import spectral_moments, upper_frequency
    fup = upper_frequency(fmax=0.0, kappa=0.035, amp_cutoff=1e-3)
    moments = spectral_moments(lambda f: fourier_spectrum(f, params, 6.0, 20.0), fup)
    moments.m0, moments.dominant_frequency, moments.bandwidth

:class:`SpectralMoments` gives the quantities derived from the moments that
the peak factors use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "MOMENT_ORDERS",
    "SpectralMoments",
    "integrate",
    "spectral_moments",
    "upper_frequency",
]

FloatArray = npt.NDArray[np.float64]

MOMENT_ORDERS = (0, 1, 2, 4)
"""The orders k of the moments that the random-vibration method uses."""

# Nodes and weights of the 15-point Kronrod rule and the embedded 7-point
# Gauss rule on [-1, 1] (Piessens et al., 1983, QUADPACK). The Gauss nodes
# are the Kronrod nodes with odd index.
_XGK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_WGK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_WG = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)
NODES = np.concatenate([-_XGK[:-1], _XGK[::-1]])
"""The 15 Kronrod nodes on [-1, 1], in increasing order."""
KRONROD_WEIGHTS = np.concatenate([_WGK[:-1], _WGK[::-1]])
"""The weights of the 15-point Kronrod rule, in the order of :data:`NODES`."""
GAUSS_WEIGHTS = np.zeros(15)
"""The weights of the 7-point Gauss rule on :data:`NODES` (0 off its nodes)."""
GAUSS_WEIGHTS[1:7:2] = _WG[:3]
GAUSS_WEIGHTS[7] = _WG[3]
GAUSS_WEIGHTS[9:15:2] = _WG[2::-1]

_MAX_INTERVALS = 20_000


def integrate(
    func: Callable[[FloatArray], FloatArray],
    edges: npt.ArrayLike,
    *,
    rtol: float = 1.0e-6,
) -> FloatArray:
    """Integrate a vector-valued function by adaptive Gauss-Kronrod quadrature.

    The interval is split at ``edges``; each piece is integrated with the
    15-point Kronrod rule, whose difference from the embedded 7-point Gauss
    rule estimates its error. The pieces with the largest errors are halved
    until the summed error of every component is below ``rtol`` times the
    size of its integral. Each round evaluates ``func`` once, on the nodes of
    all the pieces that changed.

    Parameters
    ----------
    func : callable
        Takes a one-dimensional array of n points and returns an array of
        shape (m, n): m functions evaluated at the points.
    edges : array_like
        Increasing points that split the interval of integration, the first
        and last being its ends.
    rtol : float
        Relative accuracy wanted for each of the m integrals.

    Returns
    -------
    ndarray
        The m integrals.

    Raises
    ------
    RuntimeError
        If the accuracy is not reached with 20,000 pieces.
    """
    x = np.asarray(edges, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("edges must be at least two increasing points")
    lo, hi = x[:-1], x[1:]
    done_value: FloatArray | None = None
    done_error: FloatArray | None = None
    while True:
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        points = centre[:, None] + half[:, None] * NODES
        values = np.asarray(func(points.ravel()), dtype=np.float64)
        values = values.reshape(values.shape[0], lo.size, NODES.size)
        kronrod = half * (values @ KRONROD_WEIGHTS)
        gauss = half * (values @ GAUSS_WEIGHTS)
        error = np.abs(kronrod - gauss)
        value = kronrod.sum(axis=1)
        total_error = error.sum(axis=1)
        if done_value is not None and done_error is not None:
            value = value + done_value
            total_error = total_error + done_error
        scale = np.abs(value)
        scale[scale == 0.0] = 1.0
        if np.all(total_error <= rtol * scale):
            return np.asarray(value, dtype=np.float64)
        # Halve the pieces whose error is above their share of the target;
        # keep the others
        share = rtol * scale[:, None] / lo.size
        split = np.any(error > share, axis=0)
        if not split.any():
            split = np.any(error >= error.max(axis=1, keepdims=True), axis=0)
        keep = ~split
        kept_value = kronrod[:, keep].sum(axis=1)
        kept_error = error[:, keep].sum(axis=1)
        done_value = kept_value if done_value is None else done_value + kept_value
        done_error = kept_error if done_error is None else done_error + kept_error
        mid = centre[split]
        lo, hi = np.concatenate([lo[split], mid]), np.concatenate([mid, hi[split]])
        if lo.size > _MAX_INTERVALS:
            raise RuntimeError("adaptive integration did not reach the requested accuracy")


def upper_frequency(fmax: float, kappa: float, amp_cutoff: float) -> float:
    """Upper limit fup of the moment integrals (Fortran ``get_fup``).

    fup is where the kappa filter exp(-pi*kappa*f), or the fmax filter
    1/sqrt(1 + (f/fmax)**8), falls to ``amp_cutoff``; with both, the
    smaller of the two.

    Parameters
    ----------
    fmax : float
        fmax (Hz); 0 means no fmax filter.
    kappa : float
        Kappa (s); 0 means no kappa filter.
    amp_cutoff : float
        Amplitude of the filter at fup, between 0 and 1.

    Raises
    ------
    ValueError
        If there is neither a kappa nor an fmax filter (SMSIM then
        integrates up to fup = 0, which gives no motion), or if kappa is
        negative.
    """
    if not 0 < amp_cutoff < 1:
        raise ValueError(f"amp_cutoff must be between 0 and 1, got {amp_cutoff}")
    if kappa < 0:
        raise ValueError(f"kappa must not be negative, got {kappa}")
    limits = []
    if fmax > 0:
        limits.append(fmax / amp_cutoff**0.25)
    if kappa > 0:
        limits.append(-math.log(amp_cutoff) / (math.pi * kappa))
    if not limits:
        raise ValueError("the moments need a kappa or an fmax filter to set fup")
    return float(min(limits))


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectralMoments:
    """The spectral moments m0, m1, m2 and m4 and what follows from them.

    For an oscillator they are the moments of the pseudo-velocity
    response, as in SMSIM.
    """

    m0: float
    m1: float
    m2: float
    m4: float

    @property
    def dominant_frequency(self) -> float:
        """sqrt(m2/m0)/(2*pi) (Hz), the rate of zero crossings (``freq20``)."""
        return math.sqrt(self.m2 / self.m0) / (2.0 * math.pi)

    @property
    def extrema_frequency(self) -> float:
        """sqrt(m4/m2)/(2*pi) (Hz), the rate of extrema (``freq42``)."""
        return math.sqrt(self.m4 / self.m2) / (2.0 * math.pi)

    @property
    def bandwidth(self) -> float:
        """sqrt(1 - m1**2/(m0*m2)), Der Kiureghian's (1980) q (``deltay``).

        Set to 0 when rounding makes 1 - m1**2/(m0*m2) negative, as in
        SMSIM.
        """
        return math.sqrt(max(1.0 - self.m1 * self.m1 / (self.m0 * self.m2), 0.0))

    @property
    def xi(self) -> float:
        """m2/sqrt(m0*m4), the irregularity factor (Fortran ``xi``)."""
        return self.m2 / math.sqrt(self.m0 * self.m4)

    @property
    def epsilon(self) -> float:
        """sqrt(1 - xi**2), Cartwright and Longuet-Higgins' (1956) epsilon.

        Set to 0 when rounding makes 1 - xi**2 negative, as in SMSIM.
        """
        return math.sqrt(max(1.0 - self.xi * self.xi, 0.0))


def _edges(fup: float, breakpoints: Sequence[float]) -> FloatArray:
    # 0, then 40 log-spaced points from fup*1e-5 to fup, and the breakpoints
    grid = np.concatenate(([0.0], np.geomspace(fup * 1.0e-5, fup, 41)))
    extra = [b for b in breakpoints if 0.0 < b < fup]
    return np.unique(np.concatenate([grid, extra]))


def spectral_moments(
    spectrum: Callable[[FloatArray], FloatArray],
    fup: float,
    *,
    rtol: float = 1.0e-5,
    breakpoints: Sequence[float] = (),
) -> SpectralMoments:
    """Spectral moments m0, m1, m2 and m4 of a spectrum (Fortran ``amom_rv``).

    Parameters
    ----------
    spectrum : callable
        The Fourier amplitude spectrum A(f): takes a one-dimensional array
        of frequencies (Hz) and returns the amplitudes.
    fup : float
        Upper limit of the integrals (Hz), usually from
        :func:`upper_frequency`.
    rtol : float
        Relative accuracy of each moment (Fortran ``eps_int``).
    breakpoints : sequence of float
        Frequencies where the spectrum changes quickly, such as an
        oscillator's natural frequency; the integration starts with a split
        there.

    Returns
    -------
    SpectralMoments
    """
    if fup <= 0:
        raise ValueError(f"fup must be positive, got {fup}")
    orders = np.array(MOMENT_ORDERS, dtype=np.float64)

    def integrand(f: FloatArray) -> FloatArray:
        a = np.asarray(spectrum(f), dtype=np.float64)
        return np.asarray((a * a) * (2.0 * math.pi * f) ** orders[:, None], dtype=np.float64)

    m0, m1, m2, m4 = 2.0 * integrate(integrand, _edges(fup, breakpoints), rtol=rtol)
    return SpectralMoments(m0=float(m0), m1=float(m1), m2=float(m2), m4=float(m4))
