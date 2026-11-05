"""Low-cut and high-cut filters applied to the spectrum.

SMSIM's ``spect_amp`` multiplies the spectrum by an acausal Butterworth
low-cut filter (``buttrlcf``) and a cosine-tapered high-cut filter
(``high_cut_filter``). :class:`LowCutFilter` holds the low-cut settings;
each high-cut shape is a class derived from :class:`HighCutFilter` that
holds only its own parameters::

    from pysmsim.spectrum.filters import Filters, LowCutFilter, RaisedCosineHighCut
    filters = Filters(
        low_cut=LowCutFilter(fcut=0.05, order=4),
        high_cut=RaisedCosineHighCut(f1=25.0, f2=50.0),
    )
    filters.factor(frequencies)            # both filters
    filters.high_cut.factor(frequencies)   # the high-cut filter alone

The filters are computed by compiled kernels (:class:`pysmsim._jit.Kernel`),
with their plain NumPy twins kept for the tests. The high-cut shapes are
picked by name in a TOML run file (``model = "raised_cosine"``);
:data:`HIGH_CUT_MODELS` maps each name to its class. The Fortran
``itype_hcfilt`` of each shape is given in its docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import Kernel, kernel

__all__ = [
    "HIGH_CUT_MODELS",
    "Filters",
    "HighCutFilter",
    "LowCutFilter",
    "NoHighCut",
    "QuarterCosineHighCut",
    "RaisedCosineHighCut",
]

FloatArray = npt.NDArray[np.float64]


def _apply(k: Kernel, f: npt.ArrayLike, *args: float) -> FloatArray:
    """Run a kernel on frequencies of any shape."""
    freqs = np.asarray(f, dtype=np.float64)
    flat = np.ascontiguousarray(freqs.ravel())
    return k.compiled(flat, *args).reshape(freqs.shape)


# --- Low-cut filter -------------------------------------------------------


def _butterworth_low_cut(f: FloatArray, fcut: float, order: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        if fcut == 0.0:
            out[i] = 1.0
        elif f[i] == 0.0:
            out[i] = 0.0
        else:
            out[i] = 1.0 / (1.0 + (fcut / f[i]) ** (2.0 * order))
    return out


_LOW_CUT_KERNEL = kernel(_butterworth_low_cut)


@dataclass(frozen=True, slots=True, kw_only=True)
class LowCutFilter:
    """Acausal Butterworth low-cut filter (Fortran ``buttrlcf``).

    1/(1 + (fcut/f)**(2*order)): two passes of a Butterworth filter of the
    given order, as in accelerogram processing, so the filter is 1/2 at
    ``fcut``. It is 0 at f = 0.

    Parameters
    ----------
    fcut : float
        Corner frequency (Hz); 0 means no filter.
    order : int
        Order of each of the two passes (Fortran ``norder = nslope/2``): an
        even number of at least 2, so that ``nslope`` is 4, 8, 12, ...
    """

    kernel: ClassVar[Kernel] = _LOW_CUT_KERNEL
    """The kernel that computes the filter."""

    fcut: float = 0.0
    order: int = 4

    def __post_init__(self) -> None:
        if self.fcut < 0:
            raise ValueError(f"fcut must not be negative, got {self.fcut}")
        if self.order < 2 or self.order % 2:
            raise ValueError("low-cut order must be an even number of at least 2")

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the filter on frequencies (Hz) of any shape."""
        return _apply(self.kernel, f, self.fcut, float(self.order))


# --- High-cut filters -----------------------------------------------------


class HighCutFilter(ABC):
    """Base class of the high-cut filter shapes.

    A subclass sets :attr:`name` and implements :meth:`factor`.
    """

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the shape in a run file."""

    @abstractmethod
    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the filter on frequencies (Hz) of any shape."""


@dataclass(frozen=True, slots=True, kw_only=True)
class NoHighCut(HighCutFilter):
    """No high-cut filter (Fortran ``itype_hcfilt`` 0)."""

    name: ClassVar[str] = "none"

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the filter on frequencies (Hz) of any shape: 1."""
        return np.ones(np.shape(f))


def _raised_cosine(f: FloatArray, f1: float, f2: float, eta: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        if f[i] <= f1:
            out[i] = 1.0
        elif f[i] >= f2:
            out[i] = 0.0
        else:
            out[i] = (0.5 * (1.0 + np.cos(np.pi * (f[i] - f1) / (f2 - f1)))) ** eta
    return out


def _quarter_cosine(f: FloatArray, f1: float, f2: float, eta: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        if f[i] <= f1:
            out[i] = 1.0
        elif f[i] >= f2:
            out[i] = 0.0
        else:
            out[i] = np.cos(0.5 * np.pi * (f[i] - f1) / (f2 - f1)) ** eta
    return out


_RAISED_COSINE_KERNEL = kernel(_raised_cosine)
_QUARTER_COSINE_KERNEL = kernel(_quarter_cosine)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CosineTaper(HighCutFilter):
    """Base class of the cosine tapers from 1 at f1 to 0 at f2.

    Parameters
    ----------
    f1, f2 : float
        Frequencies (Hz) where the taper starts and ends, 0 < f1 < f2
        (Fortran ``fhc1, fhc2``).
    eta : float
        Power of the cosine (``eta_hcfilt``), positive.
    """

    kernel: ClassVar[Kernel]
    """The kernel that computes the taper; it takes f1, f2 and eta."""

    f1: float
    f2: float
    eta: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.f1 < self.f2:
            raise ValueError(f"high-cut filter needs 0 < f1 < f2, got {self.f1} and {self.f2}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the filter on frequencies (Hz) of any shape."""
        return _apply(self.kernel, f, self.f1, self.f2, self.eta)


@dataclass(frozen=True, slots=True, kw_only=True)
class RaisedCosineHighCut(_CosineTaper):
    """Raised half cycle of a cosine (Fortran ``itype_hcfilt`` 1).

    (0.5*(1 + cos(pi*(f - f1)/(f2 - f1))))**eta between f1 and f2; 1 below
    f1 and 0 above f2.
    """

    name: ClassVar[str] = "raised_cosine"
    kernel: ClassVar[Kernel] = _RAISED_COSINE_KERNEL


@dataclass(frozen=True, slots=True, kw_only=True)
class QuarterCosineHighCut(_CosineTaper):
    """Quarter cycle of a cosine (Fortran ``itype_hcfilt`` 2).

    cos(0.5*pi*(f - f1)/(f2 - f1))**eta between f1 and f2; 1 below f1 and 0
    above f2.
    """

    name: ClassVar[str] = "quarter_cosine"
    kernel: ClassVar[Kernel] = _QUARTER_COSINE_KERNEL


HIGH_CUT_MODELS: dict[str, type[HighCutFilter]] = {
    cls.name: cls for cls in (NoHighCut, RaisedCosineHighCut, QuarterCosineHighCut)
}
"""The high-cut filter classes by name."""


# --- Both -----------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Filters:
    """Low-cut and high-cut filters applied to the spectrum."""

    low_cut: LowCutFilter = field(default_factory=LowCutFilter)
    high_cut: HighCutFilter = field(default_factory=NoHighCut)

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return both filters on frequencies (Hz) of any shape."""
        return self.low_cut.factor(f) * self.high_cut.factor(f)
