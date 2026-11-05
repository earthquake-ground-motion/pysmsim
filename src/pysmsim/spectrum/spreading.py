"""Geometrical spreading.

The geometrical spreading G is evaluated at the point-source distance Rps
(see :mod:`pysmsim.spectrum.finite_fault`); the Stafford et al. (2022)
optimal model also uses the rupture distance. SMSIM multiplies G by the
radiation constant, which contains 1/r_ref, so G here is dimensionless, as
in the Fortran (``gsprd``).

Each spreading model is a class derived from :class:`GeometricalSpreading`
that holds only its own parameters. Every model has
:meth:`~GeometricalSpreading.factor`, G on an array of frequencies::

    from pysmsim.spectrum.spreading import Hinged, SpreadingSegment
    spreading = Hinged(
        segments=(SpreadingSegment(r_start=1.0, a=-1.0), SpreadingSegment(r_start=40.0, a=-0.5))
    )
    spreading.factor(frequencies, rps=50.0, rrup=50.0, magnitude=6.0)

Most models do not depend on frequency; those are derived from
:class:`FrequencyIndependentSpreading` and also give the single value with
:meth:`~FrequencyIndependentSpreading.value`. The frequency-dependent ones
(Fortran ``gsprd_freq``) are derived from :class:`FrequencyDependentSpreading`
and are computed by a compiled kernel (:class:`pysmsim._jit.Kernel`), with
its plain NumPy twin kept for the tests.

The models are picked by name in a TOML run file (``model = "hinged"``);
:data:`SPREADING_MODELS` maps each name to its class. The Fortran
``i_gsprd_option`` of each model is given in its docstring.

References
----------
Atkinson, G. M. and D. M. Boore (2014). The attenuation of Fourier
amplitudes for rock sites in eastern North America. Bull. Seismol. Soc. Am.
104, 513-528.

Boore, D. M. A smooth transition between two geometrical spreading
functions. Note in the SMSIM distribution
(``geometrical_spreading_with_smooth_transition_between_linear_segments.v3.pdf``).

The other frequency-dependent models (Atkinson's November 2011 and 2012
proposals and Boore's January 2012 trial) are described only in comments in
SMSIM's ``gsprd_freq``; pysmsim follows that code.
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import Kernel, kernel

__all__ = [
    "SPREADING_MODELS",
    "Atkinson2011",
    "Atkinson2012",
    "AtkinsonBoore2014",
    "AtkinsonBoore2014R10",
    "BooreTrial",
    "FrequencyDependentSpreading",
    "FrequencyIndependentSpreading",
    "GeometricalSpreading",
    "Hinged",
    "SpreadingSegment",
    "StaffordGeneralized",
    "StaffordOptimal",
]

FloatArray = npt.NDArray[np.float64]

# SMSIM's value of pi/2 in gsprd_freq (options 4 and 5)
_SMSIM_HALF_PI = 1.570796


class GeometricalSpreading(ABC):
    """Base class of the geometrical spreading models.

    A subclass sets :attr:`name` and implements :meth:`factor`.
    """

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the model in a run file."""
    frequency_dependent: ClassVar[bool]
    """Whether G depends on frequency."""

    @abstractmethod
    def factor(self, f: npt.ArrayLike, rps: float, rrup: float, magnitude: float) -> FloatArray:
        """Geometrical spreading G on an array of frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.
        rps : float
            Point-source distance (km).
        rrup : float
            Rupture distance (km); used only by :class:`StaffordOptimal`.
        magnitude : float
            Moment magnitude.

        Returns
        -------
        ndarray
            G, with the shape of ``f``.
        """

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant; 1 km unless the model sets it."""
        return 1.0


class FrequencyIndependentSpreading(GeometricalSpreading):
    """Base class of the models in which G does not depend on frequency."""

    __slots__ = ()

    frequency_dependent: ClassVar[bool] = False

    @abstractmethod
    def value(self, rps: float, rrup: float, magnitude: float) -> float:
        """Geometrical spreading G (Fortran ``gsprd``)."""

    def factor(self, f: npt.ArrayLike, rps: float, rrup: float, magnitude: float) -> FloatArray:
        """Geometrical spreading G on an array of frequencies (all equal)."""
        return np.full(np.shape(f), self.value(rps, rrup, magnitude))


class FrequencyDependentSpreading(GeometricalSpreading):
    """Base class of the models in which G depends on frequency.

    A subclass sets :attr:`kernel` and implements :meth:`kernel_args`.
    """

    __slots__ = ()

    frequency_dependent: ClassVar[bool] = True
    kernel: ClassVar[Kernel]
    """The kernel that computes G; it takes the frequencies and then
    :meth:`kernel_args`."""

    @abstractmethod
    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """Return the numbers that :attr:`kernel` takes after the frequencies."""

    def factor(self, f: npt.ArrayLike, rps: float, rrup: float, magnitude: float) -> FloatArray:
        """Geometrical spreading G on an array of frequencies (Fortran ``gsprd_freq``)."""
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        args = self.kernel_args(rps, rrup, magnitude)
        return self.kernel.compiled(flat, *args).reshape(freqs.shape)


# --- Frequency-independent models ------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SpreadingSegment:
    """One segment of the hinged spreading model.

    Beyond ``r_start``, G falls as R**slope with slope = a + b*(M - m_ref).
    Fortran: ``rlow, a_s, b_s, m_s``.
    """

    r_start: float
    a: float
    b: float = 0.0
    m_ref: float = 0.0

    def __post_init__(self) -> None:
        if self.r_start <= 0:
            raise ValueError(f"r_start must be positive, got {self.r_start}")

    def slope(self, magnitude: float) -> float:
        """Slope of log G against log R at a magnitude."""
        return self.a + self.b * (magnitude - self.m_ref)


@dataclass(frozen=True, slots=True, kw_only=True)
class Hinged(FrequencyIndependentSpreading):
    """Hinged line segments in log G against log R (Fortran option 0).

    G = r_ref/R_1 at the start of the first segment and is continuous at
    each hinge. The first segment's slope is also used at distances below
    its start.

    Parameters
    ----------
    segments : tuple of SpreadingSegment
        The segments, in order of increasing ``r_start``.
    r_ref : float
        Reference distance (km), usually 1.
    """

    name: ClassVar[str] = "hinged"

    segments: tuple[SpreadingSegment, ...]
    r_ref: float = 1.0

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("hinged spreading needs at least one segment")
        starts = [s.r_start for s in self.segments]
        if any(b <= a for a, b in pairwise(starts)):
            raise ValueError(f"segment r_start must be strictly increasing, got {starts}")
        if self.r_ref <= 0:
            raise ValueError(f"r_ref must be positive, got {self.r_ref}")

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def value(self, rps: float, rrup: float, magnitude: float) -> float:
        """Geometrical spreading G at Rps."""
        segs = self.segments
        g = self.r_ref / segs[0].r_start
        starts = [s.r_start for s in segs]
        # Segment that contains rps, as SMSIM's locate: r_start <= rps < next r_start
        j = min(max(bisect.bisect_right(starts, rps) - 1, 0), len(segs) - 1)
        for i in range(j):
            g *= (segs[i + 1].r_start / segs[i].r_start) ** segs[i].slope(magnitude)
        return float(g * (rps / segs[j].r_start) ** segs[j].slope(magnitude))


def _check_ratio(g_ratio: float) -> None:
    if g_ratio <= 0 or g_ratio == 1:
        raise ValueError(f"g_ratio must be positive and not 1, got {g_ratio}")


@dataclass(frozen=True, slots=True, kw_only=True)
class StaffordGeneralized(FrequencyIndependentSpreading):
    """Two slopes with a smooth transition (Fortran option 6).

    ln G = gamma1*ln(x) + ((gamma2 - gamma1)/xi)
    * ln((x**xi + xt**xi)/(1 + xt**xi)), with x = Rps/r_ref,
    xt = r_t/r_ref and xi = (gamma2 - gamma1)*ln(2)/ln(g_ratio).

    Parameters
    ----------
    gamma1, gamma2 : float
        Slopes of ln G against ln R near and far from the source.
    r_t : float
        Transition distance (km).
    g_ratio : float
        G at r_t divided by the near-source line there (Fortran ``grat``).
    r_ref : float
        Reference distance (km), usually 1.
    """

    name: ClassVar[str] = "stafford_generalized"

    gamma1: float
    r_t: float
    gamma2: float
    g_ratio: float
    r_ref: float = 1.0

    def __post_init__(self) -> None:
        _check_ratio(self.g_ratio)
        if self.r_t <= 0 or self.r_ref <= 0:
            raise ValueError("r_t and r_ref must be positive")

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def value(self, rps: float, rrup: float, magnitude: float) -> float:
        """Geometrical spreading G at Rps."""
        g1, g2, r_ref = self.gamma1, self.gamma2, self.r_ref
        xi = (g2 - g1) * math.log(2.0) / math.log(self.g_ratio)
        xt = (self.r_t / r_ref) ** xi
        return math.exp(
            g1 * math.log(rps / r_ref)
            + (g2 - g1) * (1.0 / xi) * math.log(((rps / r_ref) ** xi + xt) / (1.0 + xt))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StaffordOptimal(FrequencyIndependentSpreading):
    """Stafford et al. (2022) optimal model (Fortran option 7).

    ln G = gamma1*ln(Rps) + ((gamma2 - gamma1)/2)
    * ln((Rrup**2 + r_t**2)/(r_ref**2 + r_t**2)). The first term uses Rps
    and the second Rrup (the params-file comments say Rjb, but the code
    uses Rrup).

    Parameters
    ----------
    gamma1, gamma2 : float
        Slopes of ln G against ln R near and far from the source.
    r_t : float
        Transition distance (km).
    r_ref : float
        Reference distance (km), r0 of Stafford et al.; usually 1.
    """

    name: ClassVar[str] = "stafford_optimal"

    gamma1: float
    r_t: float
    gamma2: float
    r_ref: float = 1.0

    def __post_init__(self) -> None:
        if self.r_ref <= 0:
            raise ValueError(f"r_ref must be positive, got {self.r_ref}")

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def value(self, rps: float, rrup: float, magnitude: float) -> float:
        """Geometrical spreading G at Rps and Rrup."""
        g1, g2, rt = self.gamma1, self.gamma2, self.r_t
        return math.exp(
            g1 * math.log(rps)
            + 0.5 * (g2 - g1) * math.log((rrup**2.0 + rt**2.0) / (self.r_ref**2.0 + rt**2.0))
        )


# --- Frequency-dependent models --------------------------------------------------
# Each kernel loops over the frequencies, following the Fortran branch by
# branch; r is Rps.


def _atkinson_2011(f: FloatArray, r: float) -> FloatArray:
    out = np.empty_like(f)
    r1 = min(r, 60.0)
    r2 = max(r / 60.0, 1.0)
    for i in range(f.size):
        fi = f[i]
        if fi <= 1.0:
            b = -0.6
        elif fi >= 3.162:
            b = -0.1
        else:
            b = -0.6 + np.log10(fi)
        delta = 0.7 - min(0.7, 0.7 * np.exp(b * (r - 1.0)))
        out[i] = 10.0 ** (-1.6 * np.log10(r1) - 0.5 * np.log10(r2) + delta)
    return out


def _boore_trial(
    f: FloatArray,
    r: float,
    r1: float,
    slope_lf: float,
    slope_hf: float,
    slope_far: float,
    ft1: float,
    ft2: float,
) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        fi = f[i]
        # Slope within r1 (Fortran gsprd_pwr_dmb)
        if fi <= ft1:
            near = slope_lf
        elif fi <= ft2:
            near = slope_lf - ((slope_lf - slope_hf) / np.log(ft2 / ft1)) * np.log(fi / ft1)
        else:
            near = slope_hf
        if r <= 1.0:
            out[i] = 1.0 / r
        elif r <= r1:
            out[i] = r**near
        else:
            out[i] = r1**near * (r / r1) ** slope_far
    return out


def _atkinson_2012(f: FloatArray, r: float, r_ref: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        rt = min(70.0, 70.0 - min(57.2 * np.log10(f[i] / 2.0), 40.0))
        if r < rt:
            out[i] = (r_ref / r) ** 1.3
        else:
            out[i] = (r_ref / rt) ** 1.3 * (rt / r) ** 0.5
    return out


def _atkinson_boore_2014(
    f: FloatArray, r: float, h: float, r_ref: float, half_pi: float
) -> FloatArray:
    out = np.empty_like(f)
    r_hinge = 50.0
    if r <= h:
        clf = 0.2 * np.cos(half_pi * (r - h) / (1.0 - h))
    else:
        clf = 0.2 * np.cos(half_pi * (min(r, r_hinge) - h) / (r_hinge - h))
    for i in range(f.size):
        if r <= r_hinge:
            tc = max(1.0 - 1.429 * np.log10(max(f[i], 1.0)), 0.0)
            out[i] = 10.0 ** (tc * clf) * (r_ref / r) ** 1.3
        else:
            out[i] = (r_ref / r_hinge) ** 1.3 * (r_hinge / r) ** 0.5
    return out


def _atkinson_boore_2014_r10(
    f: FloatArray, r: float, h: float, r_ref: float, half_pi: float
) -> FloatArray:
    out = np.empty_like(f)
    r_hinge = 50.0
    if r <= h:
        clf = 0.2 * np.cos(half_pi * (r - h) / (1.0 - h))
    else:
        clf = 0.2 * np.cos(half_pi * (min(r, r_hinge) - h) / (r_hinge - h))
    for i in range(f.size):
        if r <= r_hinge:
            tc = max(1.0 - 1.429 * np.log10(max(f[i], 1.0)), 0.0)
            if r < 10.0:
                out[i] = 10.0 ** (tc * clf) * (r_ref / r) ** 1.0
            else:
                out[i] = 10.0 ** (1.3 - 1.0) * 10.0 ** (tc * clf) * (r_ref / r) ** 1.3
        else:
            out[i] = 10.0 ** (1.3 - 1.0) * (r_ref / r_hinge) ** 1.3 * (r_hinge / r) ** 0.5
    return out


_ATKINSON_2011_KERNEL = kernel(_atkinson_2011)
_BOORE_TRIAL_KERNEL = kernel(_boore_trial)
_ATKINSON_2012_KERNEL = kernel(_atkinson_2012)
_ATKINSON_BOORE_2014_KERNEL = kernel(_atkinson_boore_2014)
_ATKINSON_BOORE_2014_R10_KERNEL = kernel(_atkinson_boore_2014_r10)


@dataclass(frozen=True, slots=True, kw_only=True)
class Atkinson2011(FrequencyDependentSpreading):
    """Atkinson's November 2011 ENA proposal (Fortran option 1).

    log10 G = -1.6*log10(min(R, 60)) - 0.5*log10(max(R/60, 1)) + delta(f, R).
    The model has no parameters; r_ref is 1 km.
    """

    name: ClassVar[str] = "atkinson_2011"
    kernel: ClassVar[Kernel] = _ATKINSON_2011_KERNEL

    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """(Rps,)."""
        return (rps,)


@dataclass(frozen=True, slots=True, kw_only=True)
class BooreTrial(FrequencyDependentSpreading):
    """Boore's January 2012 trial ENA spreading (Fortran option 2).

    G = 1/R within 1 km, R**p(f) out to r1 and R**slope_far beyond it. p(f)
    is slope_lf below ft1, slope_hf above ft2 and linear in log f between.
    r_ref is fixed at 1 km.

    Parameters
    ----------
    r1 : float
        Distance (km) where the far slope starts (Fortran ``r1_dmb_gsprd``).
    slope_lf, slope_hf : float
        Slopes within r1 at low and high frequencies
        (``pgsprd_r_le_r1_lf``, ``pgsprd_r_le_r1_hf``).
    slope_far : float
        Slope beyond r1 (``pgsprd_r_gt_r1``).
    ft1, ft2 : float
        Frequencies (Hz) of the transition between the near slopes
        (``ft1_dmb_gsprd``, ``ft2_dmb_gsprd``).

    Notes
    -----
    SMSIM sets r_ref to 1 inside this model, which also changes the r_ref
    used afterwards in the radiation constant if the params file gave
    another value. pysmsim uses r_ref = 1 km throughout.
    """

    name: ClassVar[str] = "boore_trial"
    kernel: ClassVar[Kernel] = _BOORE_TRIAL_KERNEL

    r1: float
    slope_lf: float
    slope_hf: float
    slope_far: float
    ft1: float
    ft2: float

    def __post_init__(self) -> None:
        if self.r1 <= 1.0:
            raise ValueError(f"r1 must be greater than 1 km, got {self.r1}")
        if not 0 < self.ft1 < self.ft2:
            raise ValueError("boore_trial spreading needs 0 < ft1 < ft2")

    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """(Rps, r1, slope_lf, slope_hf, slope_far, ft1, ft2)."""
        return (rps, self.r1, self.slope_lf, self.slope_hf, self.slope_far, self.ft1, self.ft2)


@dataclass(frozen=True, slots=True, kw_only=True)
class Atkinson2012(FrequencyDependentSpreading):
    """Atkinson's 2012 NGA-East proposal (Fortran option 3).

    G = (r_ref/R)**1.3 within the transition distance
    rt(f) = min(70, 70 - min(57.2*log10(f/2), 40)) km, and R**-0.5 beyond.

    Parameters
    ----------
    r_ref : float
        Reference distance (km), usually 1.
    """

    name: ClassVar[str] = "atkinson_2012"
    kernel: ClassVar[Kernel] = _ATKINSON_2012_KERNEL

    r_ref: float = 1.0

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """(Rps, r_ref)."""
        return (rps, self.r_ref)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtkinsonBoore2014(FrequencyDependentSpreading):
    """Atkinson and Boore (2014), BSSA 104, 513-528 (Fortran option 4).

    G = 10**(tc(f)*clf(R)) * (r_ref/R)**1.3 within 50 km and R**-0.5 beyond,
    where tc = max(1 - 1.429*log10(max(f, 1)), 0) and clf is a cosine taper
    in distance that crosses zero at h.

    Parameters
    ----------
    h : float
        Distance (km) where clf = 0 (Fortran ``r1_dmb_gsprd``); not 1 and
        less than 50.
    r_ref : float
        Reference distance (km), usually 1.
    fortran_compat : bool
        SMSIM uses 1.570796 for pi/2 in clf; pysmsim uses pi/2 unless this
        is set. The difference in G is below 1e-6.
    """

    name: ClassVar[str] = "atkinson_boore_2014"
    kernel: ClassVar[Kernel] = _ATKINSON_BOORE_2014_KERNEL

    h: float
    r_ref: float = 1.0
    fortran_compat: bool = False

    def __post_init__(self) -> None:
        if self.h == 1.0 or not 0 < self.h < 50.0:
            raise ValueError(f"h must be between 0 and 50 km and not 1, got {self.h}")

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """(Rps, h, r_ref, pi/2)."""
        half_pi = _SMSIM_HALF_PI if self.fortran_compat else 0.5 * math.pi
        return (rps, self.h, self.r_ref, half_pi)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtkinsonBoore2014R10(FrequencyDependentSpreading):
    """Atkinson and Boore (2014) with 1/R within 10 km (Fortran option 5, AB14Rlt10mod).

    As :class:`AtkinsonBoore2014`, but G falls as 1/R within 10 km; beyond
    10 km G is scaled by 10**0.3 so that it is continuous there.

    Parameters
    ----------
    h : float
        Distance (km) where clf = 0 (Fortran ``r1_dmb_gsprd``); not 1 and
        less than 50.
    r_ref : float
        Reference distance (km), usually 1.
    fortran_compat : bool
        SMSIM uses 1.570796 for pi/2; pysmsim uses pi/2 unless this is set.
    """

    name: ClassVar[str] = "atkinson_boore_2014_r10"
    kernel: ClassVar[Kernel] = _ATKINSON_BOORE_2014_R10_KERNEL

    h: float
    r_ref: float = 1.0
    fortran_compat: bool = False

    def __post_init__(self) -> None:
        if self.h == 1.0 or not 0 < self.h < 50.0:
            raise ValueError(f"h must be between 0 and 50 km and not 1, got {self.h}")

    def reference_distance(self) -> float:
        """Distance r_ref (km) in the radiation constant."""
        return self.r_ref

    def kernel_args(self, rps: float, rrup: float, magnitude: float) -> tuple[float, ...]:
        """(Rps, h, r_ref, pi/2)."""
        half_pi = _SMSIM_HALF_PI if self.fortran_compat else 0.5 * math.pi
        return (rps, self.h, self.r_ref, half_pi)


SPREADING_MODELS: dict[str, type[GeometricalSpreading]] = {
    cls.name: cls
    for cls in (
        Hinged,
        Atkinson2011,
        BooreTrial,
        Atkinson2012,
        AtkinsonBoore2014,
        AtkinsonBoore2014R10,
        StaffordGeneralized,
        StaffordOptimal,
    )
}
"""The spreading classes by name, in the order of the Fortran options."""
