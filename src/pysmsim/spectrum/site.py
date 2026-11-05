"""Crustal amplification and high-frequency diminution at the site.

The site terms of SMSIM's ``spect_amp`` are the crustal amplification
(``crustal_amp_factor``) and the kappa and fmax filters (the site part of
``dimin``). :class:`Site` holds their parameters::

    from pysmsim.spectrum.site import Site
    site = Site(
        amp_frequencies=(0.1, 1.0, 10.0),
        amplifications=(1.0, 1.5, 3.0),
        kappa0=0.035,
    )
    site.amplification(frequencies)       # crustal amplification
    site.kappa(6.0)                       # kappa (s) at M 6
    site.diminution(frequencies, 6.0)     # kappa and fmax filters

Both factors are computed by compiled kernels (:class:`pysmsim._jit.Kernel`),
with their plain NumPy twins kept for the tests. The anelastic (Q) part of
``dimin`` is in :mod:`pysmsim.spectrum.attenuation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import Kernel, kernel

__all__ = ["Site"]

FloatArray = npt.NDArray[np.float64]


def _crustal_amplification(f: FloatArray, famp: FloatArray, amp: FloatArray) -> FloatArray:
    # Linear in frequency, logarithmic in amplitude (crustal_amp_factor with
    # interp_type 2, the type spect_amp uses); constant beyond the table
    out = np.empty_like(f)
    n = famp.size
    for i in range(f.size):
        fi = f[i]
        if fi <= famp[0]:
            out[i] = amp[0]
        elif fi >= famp[n - 1]:
            out[i] = amp[n - 1]
        else:
            j = np.searchsorted(famp, fi, side="right") - 1
            slope = np.log10(amp[j + 1] / amp[j]) / (famp[j + 1] - famp[j])
            out[i] = amp[j] * 10.0 ** (slope * (fi - famp[j]))
    return out


def _site_diminution(f: FloatArray, kappa: float, fmax: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        d = np.exp(-np.pi * kappa * f[i])
        if fmax != 0.0:
            d = d / np.sqrt(1.0 + (f[i] / fmax) ** 8)
        out[i] = d
    return out


_CRUSTAL_AMPLIFICATION_KERNEL = kernel(_crustal_amplification)
_SITE_DIMINUTION_KERNEL = kernel(_site_diminution)


@dataclass(frozen=True, slots=True, kw_only=True)
class Site:
    """Crustal amplification and high-frequency diminution.

    Fortran: ``namps, famp, amp`` and ``fm, akappa, dkappadmag, amagkref``.

    Parameters
    ----------
    amp_frequencies, amplifications : tuple of float
        Crustal amplification table. Between the frequencies it is
        interpolated linearly in frequency and logarithmically in amplitude;
        below the first and above the last frequency it is constant.
    kappa0 : float
        Kappa (s) at ``kappa_magnitude_ref`` (Fortran ``akappa``).
    dkappa_dmag : float
        Change of kappa per magnitude unit (``dkappadmag``).
    kappa_magnitude_ref : float
        Reference magnitude for kappa (``amagkref``).
    fmax : float
        fmax (Hz) (Fortran ``fm``); 0 means no fmax filter.
    """

    amplification_kernel: ClassVar[Kernel] = _CRUSTAL_AMPLIFICATION_KERNEL
    """The kernel that computes the crustal amplification."""
    diminution_kernel: ClassVar[Kernel] = _SITE_DIMINUTION_KERNEL
    """The kernel that computes the kappa and fmax filters."""

    amp_frequencies: tuple[float, ...]
    amplifications: tuple[float, ...]
    kappa0: float = 0.0
    dkappa_dmag: float = 0.0
    kappa_magnitude_ref: float = 0.0
    fmax: float = 0.0

    def __post_init__(self) -> None:
        if len(self.amp_frequencies) != len(self.amplifications):
            raise ValueError("amp_frequencies and amplifications must have the same length")
        if not self.amp_frequencies:
            raise ValueError("site amplification needs at least one point")
        if any(b <= a for a, b in pairwise(self.amp_frequencies)):
            raise ValueError(
                f"amp_frequencies must be strictly increasing, got {self.amp_frequencies}"
            )
        if any(a <= 0 for a in self.amplifications):
            raise ValueError("amplifications must be positive")
        if self.fmax < 0:
            raise ValueError(f"fmax must not be negative, got {self.fmax}")

    def amplification(self, f: npt.ArrayLike) -> FloatArray:
        """Crustal amplification (Fortran ``crustal_amp_factor``).

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.

        Returns
        -------
        ndarray
            The amplification, with the shape of ``f``.
        """
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        famp = np.asarray(self.amp_frequencies, dtype=np.float64)
        amp = np.asarray(self.amplifications, dtype=np.float64)
        return self.amplification_kernel.compiled(flat, famp, amp).reshape(freqs.shape)

    def kappa(self, magnitude: float) -> float:
        """Kappa (s) at a moment magnitude (Fortran ``kappa_f``).

        kappa = kappa0 + dkappa_dmag * (M - kappa_magnitude_ref). It can be
        negative when ``dkappa_dmag`` is not 0, as in SMSIM.
        """
        return self.kappa0 + self.dkappa_dmag * (magnitude - self.kappa_magnitude_ref)

    def diminution(self, f: npt.ArrayLike, magnitude: float) -> FloatArray:
        """Kappa and fmax filters: exp(-pi*kappa*f) / sqrt(1 + (f/fmax)**8).

        The site part of Fortran ``dimin``; without fmax (``fmax`` = 0) the
        second factor is left out.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.
        magnitude : float
            Moment magnitude (kappa may depend on it).

        Returns
        -------
        ndarray
            The factor, with the shape of ``f``.
        """
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        kappa = self.kappa(magnitude)
        return self.diminution_kernel.compiled(flat, kappa, self.fmax).reshape(freqs.shape)
