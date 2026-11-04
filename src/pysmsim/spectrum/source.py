"""Source spectrum: seismic moment, corner frequencies, shape and duration.

The source factor of the Fourier displacement spectrum is M0 * S(f), where
S(f) is the source shape (1 at f = 0). Each source model is a class derived
from :class:`SourceSpectrum` that holds only the parameters it uses and
provides, for one magnitude:

- :meth:`~SourceSpectrum.terms`: M0, stress, the corner frequencies fa and
  fb, and the weight eps of the additive two-corner models, as a
  :class:`SourceTerms` (Fortran ``spect_scale``);
- :meth:`~SourceSpectrum.shape`: S(f) on an array of frequencies (Fortran
  ``spect_shape``);
- :meth:`~SourceSpectrum.duration`: the source duration used by the RV and
  TD calculations (Fortran ``dursource``).

For example::

    from pysmsim.spectrum.source import SingleCorner
    from pysmsim.spectrum.stress import LogLinearStress
    source = SingleCorner(stress=LogLinearStress(stress_ref=100.0))
    terms = source.terms(6.0, shear_velocity=3.5)
    s = source.shape(frequencies, terms)

The models are picked by name in a TOML run file (``model =
"single_corner"``); :data:`SOURCE_MODELS` maps each name to its class. The
Fortran ``numsource`` of each model is given in its docstring.

Each shape is computed by a compiled kernel (:class:`ShapeKernel`) that
takes the frequencies and the numbers from :meth:`~SourceSpectrum.shape_args`;
the plain NumPy version of each kernel is kept as its twin for the tests.

References
----------
Boore, D. M. (2003). Simulation of ground motion using the stochastic
method. Pure Appl. Geophys. 160, 635-676 (Table 2: the source models).

Boore, D. M., C. Di Alessandro and N. A. Abrahamson (2014). A generalization
of the double-corner-frequency source spectral model and its use in the
SCEC BBP validation exercise. Bull. Seismol. Soc. Am. 104, 2387-2398
(equations 3, 7, 14 and 16: the generalised two-corner models).

Ji, C. and R. J. Archuleta (2020): the JA19_2S model. The coefficients are
those hard-wired in SMSIM's ``spect_scale``; no copy of the paper is in the
SMSIM distribution, so they have not been checked against it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import jit
from pysmsim.spectrum.stress import StressModel

__all__ = [
    "BRUNE_CONSTANT",
    "SOURCE_MODELS",
    "Atkinson1993",
    "Atkinson2005",
    "AtkinsonBoore1998California",
    "AtkinsonSilva1997",
    "AtkinsonSilva2000",
    "BoatwrightChoy",
    "EmpiricalAdditiveTwoCorner",
    "Haddon1996",
    "JiArchuleta2020",
    "Joyner1984",
    "JoynerENA",
    "ShapeKernel",
    "SingleCorner",
    "SourceSpectrum",
    "SourceTerms",
    "TwoCornerAdditive",
    "TwoCornerMultiplicative",
    "WeightedDuration",
    "corner_frequency",
    "seismic_moment",
]

FloatArray = npt.NDArray[np.float64]

BRUNE_CONSTANT = 4.906e6
"""fc = BRUNE_CONSTANT * beta * (stress / M0)**(1/3), with fc in Hz, beta in
km/s, stress in bars and M0 in dyne-cm (Boore et al., 2014, eq. 3)."""


def seismic_moment(magnitude: float) -> float:
    """Seismic moment (dyne-cm) of a moment magnitude.

    M0 = 10**(1.5*M + 16.05), as in SMSIM.
    """
    return float(10.0 ** (1.5 * magnitude + 16.05))


def corner_frequency(m0: float, stress: float, shear_velocity: float) -> float:
    """Single-corner frequency (Hz) for a seismic moment and stress.

    Parameters
    ----------
    m0 : float
        Seismic moment (dyne-cm).
    stress : float
        Stress parameter (bars).
    shear_velocity : float
        Shear-wave velocity near the source (km/s).

    Returns
    -------
    float
        fc = 4.906e6 * beta * (stress / M0)**(1/3) (Boore et al., 2014, eq. 3).
    """
    return float(BRUNE_CONSTANT * shear_velocity * (stress / m0) ** (1.0 / 3.0))


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceTerms:
    """Source quantities that depend on magnitude but not on frequency.

    Attributes
    ----------
    magnitude : float
        Moment magnitude.
    m0 : float
        Seismic moment (dyne-cm).
    stress : float or None
        Stress parameter (bars); None for the models whose corner
        frequencies are fixed functions of magnitude. For Ji and Archuleta
        (2020) it is the equivalent single-corner stress.
    fa, fb : float
        Corner frequencies (Hz). For the single-corner model fb = fa.
    eps : float
        Relative weight of the fb term in the additive two-corner models
        (Fortran ``am0b_m0``); 0 for the models that do not use it.
    """

    magnitude: float
    m0: float
    stress: float | None
    fa: float
    fb: float
    eps: float = 0.0


# --- Shape kernels ------------------------------------------------------------
# Each kernel takes one-dimensional frequencies and plain numbers, so that
# Numba can compile it and a compiled RV integrand can call it.


@dataclass(frozen=True, slots=True)
class ShapeKernel:
    """A source-shape kernel, compiled and as its NumPy twin.

    Both are called as ``kernel(f, *source.shape_args(terms))``, with ``f``
    a one-dimensional array.
    """

    numpy: Callable[..., FloatArray]
    """The plain NumPy function."""
    compiled: Callable[..., FloatArray]
    """The same function compiled by Numba."""


def _kernel(func: Callable[..., FloatArray]) -> ShapeKernel:
    return ShapeKernel(numpy=func, compiled=jit(func))


def _single_corner_shape(f: FloatArray, fa: float, pf_a: float, pd_a: float) -> FloatArray:
    """1/(1 + (f/fa)**pf_a)**pd_a."""
    return 1.0 / (1.0 + (f / fa) ** pf_a) ** pd_a


def _joyner_shape(f: FloatArray, fa: float, fb: float) -> FloatArray:
    """1/((1 + (f/fa)**2)**(3/4) * (1 + (f/fb)**2)**(1/4))."""
    return 1.0 / ((1.0 + (f / fa) ** 2) ** 0.75 * (1.0 + (f / fb) ** 2) ** 0.25)


def _additive_omega_squared_shape(f: FloatArray, fa: float, fb: float, eps: float) -> FloatArray:
    """(1 - eps)/(1 + (f/fa)**2) + eps/(1 + (f/fb)**2)."""
    return (1.0 - eps) / (1.0 + (f / fa) ** 2) + eps / (1.0 + (f / fb) ** 2)


def _haddon_shape(f: FloatArray, fa: float, fb: float) -> FloatArray:
    """1/((1 + (f/fa)**8)**(1/8) * (1 + (f/fb)**8)**(1/8))."""
    return 1.0 / ((1.0 + (f / fa) ** 8) ** 0.125 * (1.0 + (f / fb) ** 8) ** 0.125)


def _boatwright_choy_shape(f: FloatArray, fa: float, fb: float) -> FloatArray:
    """1 below fa and fa/f above it, times a single corner at fb."""
    return fa / np.maximum(f, fa) / np.sqrt(1.0 + (f / fb) ** 2)


def _multiplicative_shape(
    f: FloatArray, fa: float, fb: float, pf_a: float, pd_a: float, pf_b: float, pd_b: float
) -> FloatArray:
    """1/((1 + (f/fa)**pf_a)**pd_a * (1 + (f/fb)**pf_b)**pd_b)."""
    return 1.0 / ((1.0 + (f / fa) ** pf_a) ** pd_a * (1.0 + (f / fb) ** pf_b) ** pd_b)


def _additive_shape(
    f: FloatArray,
    fa: float,
    fb: float,
    eps: float,
    pf_a: float,
    pd_a: float,
    pf_b: float,
    pd_b: float,
) -> FloatArray:
    """(1 - eps)/(1 + (f/fa)**pf_a)**pd_a + eps/(1 + (f/fb)**pf_b)**pd_b."""
    return (1.0 - eps) / (1.0 + (f / fa) ** pf_a) ** pd_a + eps / (1.0 + (f / fb) ** pf_b) ** pd_b


def _ji_archuleta_shape(f: FloatArray, fa: float, fb: float) -> FloatArray:
    """1/((1 + (f/fa)**4)**(1/4) * (1 + (f/fb)**4)**(1/4))."""
    return 1.0 / ((1.0 + (f / fa) ** 4) ** 0.25 * (1.0 + (f / fb) ** 4) ** 0.25)


_SINGLE_CORNER_KERNEL = _kernel(_single_corner_shape)
_JOYNER_KERNEL = _kernel(_joyner_shape)
_ADDITIVE_OMEGA_SQUARED_KERNEL = _kernel(_additive_omega_squared_shape)
_HADDON_KERNEL = _kernel(_haddon_shape)
_BOATWRIGHT_CHOY_KERNEL = _kernel(_boatwright_choy_shape)
_MULTIPLICATIVE_KERNEL = _kernel(_multiplicative_shape)
_ADDITIVE_KERNEL = _kernel(_additive_shape)
_JI_ARCHULETA_KERNEL = _kernel(_ji_archuleta_shape)


# --- Base classes -------------------------------------------------------------


class SourceSpectrum(ABC):
    """Base class of the source models.

    A subclass sets :attr:`name` and :attr:`kernel` and implements
    :meth:`terms`, :meth:`shape_args` and :meth:`duration`; :meth:`shape`
    is the same for every model.
    """

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the model in a run file."""
    kernel: ClassVar[ShapeKernel]
    """The kernel that computes the shape."""

    @abstractmethod
    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, stress and corner frequencies at a magnitude.

        Parameters
        ----------
        magnitude : float
            Moment magnitude.
        shear_velocity : float
            Shear-wave velocity near the source (km/s).

        Returns
        -------
        SourceTerms
        """

    @abstractmethod
    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """Return the numbers that :attr:`kernel` takes after the frequencies."""

    @abstractmethod
    def duration(self, terms: SourceTerms) -> float:
        """Source duration (s) (Fortran ``dursource``)."""

    def shape(self, f: npt.ArrayLike, terms: SourceTerms) -> FloatArray:
        """Source shape S(f) on an array of frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.
        terms : SourceTerms
            From :meth:`terms`.

        Returns
        -------
        ndarray
            S(f), with the shape of ``f``; S(0) = 1.
        """
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        return self.kernel.compiled(flat, *self.shape_args(terms)).reshape(freqs.shape)


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightedDuration(SourceSpectrum):
    """Base class of the models whose duration is a weighted sum of 1/fa and 1/fb.

    That is every source model except Ji and Archuleta (2020).

    Parameters
    ----------
    duration_weight_fa, duration_weight_fb : float
        Source duration = w_fa/fa + w_fb/fb (Fortran ``w_fa, w_fb``).
    """

    duration_weight_fa: float = 0.5
    duration_weight_fb: float = 0.5

    def duration(self, terms: SourceTerms) -> float:
        """Source duration (s): w_fa/fa + w_fb/fb."""
        return self.duration_weight_fa / terms.fa + self.duration_weight_fb / terms.fb


# --- Models with a stress parameter -------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleCorner(WeightedDuration):
    """Single corner frequency (Fortran source 1).

    S(f) = 1/(1 + (f/fc)**pf_a)**pd_a, with fc from the stress (Boore et al.,
    2014, eq. 3). The defaults give the omega-squared (Brune) model.

    Parameters
    ----------
    stress : StressModel
        Stress parameter as a function of magnitude.
    pf_a, pd_a : float
        Exponents of the shape.
    """

    name: ClassVar[str] = "single_corner"
    kernel: ClassVar[ShapeKernel] = _SINGLE_CORNER_KERNEL

    stress: StressModel
    pf_a: float = 2.0
    pd_a: float = 1.0

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, stress and corner frequency at a magnitude (fb = fa)."""
        m0 = seismic_moment(magnitude)
        stress = self.stress(magnitude)
        fa = corner_frequency(m0, stress, shear_velocity)
        return SourceTerms(magnitude=magnitude, m0=m0, stress=stress, fa=fa, fb=fa)

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, pf_a, pd_a)."""
        return (terms.fa, self.pf_a, self.pd_a)


@dataclass(frozen=True, slots=True, kw_only=True)
class Joyner1984(WeightedDuration):
    """Joyner (1984), BSSA 74, 1167-1188 (Fortran source 2).

    S(f) = 1/((1 + (f/fa)**2)**(3/4) * (1 + (f/fb)**2)**(1/4)). Below the
    hinge magnitude fb/fa is fixed; above it, fb stays at its hinge value
    and fa falls as M0**(-1/2).

    Parameters
    ----------
    stress : StressModel
        Stress parameter as a function of magnitude.
    magnitude_hinge : float
        Magnitude at which the scaling changes (Fortran ``amagc``).
    fb_over_fa : float
        Ratio of the corner frequencies below the hinge (``fbdfa``).

    Notes
    -----
    SMSIM computes the stress of this model with the log-linear formula
    whatever stress model the params file names. That gives zero stress
    (and NaN corner frequencies) for Hoerl's model; pysmsim always uses the
    given stress model.
    """

    name: ClassVar[str] = "joyner_1984"
    kernel: ClassVar[ShapeKernel] = _JOYNER_KERNEL

    stress: StressModel
    magnitude_hinge: float
    fb_over_fa: float = 4.0

    def __post_init__(self) -> None:
        if self.fb_over_fa <= 0:
            raise ValueError(f"fb_over_fa must be positive, got {self.fb_over_fa}")

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, stress and corner frequencies at a magnitude."""
        m0 = seismic_moment(magnitude)
        stress = self.stress(magnitude)
        m0c = seismic_moment(self.magnitude_hinge)
        ratio = self.fb_over_fa
        cb = BRUNE_CONSTANT * shear_velocity
        fb = cb * ratio**0.75 * (stress / min(m0, m0c)) ** (1.0 / 3.0)
        if m0 < m0c:
            fa = fb / ratio
        else:
            fa = cb * stress ** (1.0 / 3.0) * m0c ** (1.0 / 6.0) * ratio**-0.25 * m0**-0.5
        return SourceTerms(magnitude=magnitude, m0=m0, stress=stress, fa=fa, fb=fb)

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb)."""
        return (terms.fa, terms.fb)


def _log_linear(coefficients: tuple[float, float, float], magnitude: float) -> float:
    """10**(c1 + c2*(M - M_ref)) for coefficients (c1, c2, M_ref)."""
    c1, c2, m_ref = coefficients
    return float(10.0 ** (c1 + c2 * (magnitude - m_ref)))


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoCornerMultiplicative(WeightedDuration):
    """Generalised multiplicative two-corner model (Fortran source 11).

    S(f) = 1/((1 + (f/fa)**pf_a)**pd_a * (1 + (f/fb)**pf_b)**pd_b) (Boore et
    al., 2014). fa is a given function of magnitude, and fb is set so that
    the high-frequency level equals that of a single-corner model with the
    same stress (eq. 7).

    Parameters
    ----------
    stress : StressModel
        Stress parameter as a function of magnitude.
    log10_fa : tuple of float
        (c1, c2, M_ref) in log10(fa) = c1 + c2*(M - M_ref) (Fortran ``c1_fa,
        c2_fa, amagc4fa``).
    pf_a, pd_a, pf_b, pd_b : float
        Exponents of the shape. The defaults give an omega-squared fall-off
        above fb.
    """

    name: ClassVar[str] = "two_corner_multiplicative"
    kernel: ClassVar[ShapeKernel] = _MULTIPLICATIVE_KERNEL

    stress: StressModel
    log10_fa: tuple[float, float, float]
    pf_a: float = 2.0
    pd_a: float = 0.5
    pf_b: float = 2.0
    pd_b: float = 0.5

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, stress and corner frequencies at a magnitude."""
        m0 = seismic_moment(magnitude)
        stress = self.stress(magnitude)
        fc = corner_frequency(m0, stress, shear_velocity)
        fa = _log_linear(self.log10_fa, magnitude)
        fb = (fc**2 / fa ** (self.pf_a * self.pd_a)) ** (1.0 / (self.pf_b * self.pd_b))
        return SourceTerms(magnitude=magnitude, m0=m0, stress=stress, fa=fa, fb=fb)

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb, pf_a, pd_a, pf_b, pd_b)."""
        return (terms.fa, terms.fb, self.pf_a, self.pd_a, self.pf_b, self.pd_b)


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoCornerAdditive(WeightedDuration):
    """Generalised additive two-corner model (Fortran source 12).

    S(f) = (1 - eps)/(1 + (f/fa)**pf_a)**pd_a + eps/(1 + (f/fb)**pf_b)**pd_b
    (Boore et al., 2014). fa and eps are given functions of magnitude, and
    fb is set so that the high-frequency level equals that of a
    single-corner model with the same stress (eqs. 14-17). Below the
    magnitude where eps = 1 the model is single-corner.

    Parameters
    ----------
    stress : StressModel
        Stress parameter as a function of magnitude.
    log10_fa : tuple of float
        (c1, c2, M_ref) in log10(fa) = c1 + c2*(M - M_ref) (Fortran ``c1_fa,
        c2_fa, amagc4fa``).
    log10_eps : tuple of float
        (c1, c2, M_ref) in log10(eps) = c1 + c2*(M - M_ref) (``c1_eps,
        c2_eps, amagc4eps``).
    pf_a, pd_a, pf_b, pd_b : float
        Exponents of the shape. The defaults give omega-squared terms.
    fortran_compat : bool
        Reproduce SMSIM 2023-03-03 exactly. SMSIM finds the magnitude where
        eps = 1 as -c1/c2, ignoring M_ref in log10_eps. By default pysmsim
        uses M_ref - c1/c2 (Boore et al., 2014, eq. 16).
    """

    name: ClassVar[str] = "two_corner_additive"
    kernel: ClassVar[ShapeKernel] = _ADDITIVE_KERNEL

    stress: StressModel
    log10_fa: tuple[float, float, float]
    log10_eps: tuple[float, float, float]
    pf_a: float = 2.0
    pd_a: float = 1.0
    pf_b: float = 2.0
    pd_b: float = 1.0
    fortran_compat: bool = False

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, stress, corner frequencies and eps at a magnitude.

        Raises
        ------
        ValueError
            If the stress is too small for the given fa and eps, so that fb
            is undefined.
        """
        m = magnitude
        m0 = seismic_moment(m)
        stress = self.stress(m)
        fc = corner_frequency(m0, stress, shear_velocity)
        c1_eps, c2_eps, m_ref_eps = self.log10_eps

        # Magnitude at which eps = 1; below it the model is single-corner
        if c2_eps == 0.0:
            m_eps1 = -9.9
        elif self.fortran_compat:
            m_eps1 = -c1_eps / c2_eps
        else:
            m_eps1 = m_ref_eps - c1_eps / c2_eps
        if m < m_eps1:
            return SourceTerms(magnitude=m, m0=m0, stress=stress, fa=fc, fb=fc, eps=1.0)

        fa = _log_linear(self.log10_fa, m)
        eps = _log_linear(self.log10_eps, m)
        if eps < 1.0:
            stress_min = (math.sqrt(1.0 - eps) / (BRUNE_CONSTANT * shear_velocity) * fa) ** 3 * m0
            if stress < stress_min:
                raise ValueError(
                    f"{self.name} source: stress {stress:g} bars is below the minimum "
                    f"{stress_min:g} bars for fa = {fa:g} Hz and eps = {eps:g} at M {m:g}, "
                    "so fb is undefined"
                )
        fb = math.sqrt((fc**2 - (1.0 - eps) * fa**2) / eps)
        return SourceTerms(magnitude=m, m0=m0, stress=stress, fa=fa, fb=fb, eps=eps)

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb, eps, pf_a, pd_a, pf_b, pd_b)."""
        return (terms.fa, terms.fb, terms.eps, self.pf_a, self.pd_a, self.pf_b, self.pd_b)


# --- Models with corner frequencies fixed by magnitude ------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _FixedCorners(WeightedDuration):
    """Base class of the models whose corners are fixed functions of magnitude."""

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb)."""
        return (terms.fa, terms.fb)

    @staticmethod
    def _terms(magnitude: float, fa: float, fb: float, eps: float = 0.0) -> SourceTerms:
        return SourceTerms(
            magnitude=magnitude, m0=seismic_moment(magnitude), stress=None, fa=fa, fb=fb, eps=eps
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EmpiricalAdditiveTwoCorner(_FixedCorners):
    """Base class of the empirical additive two-corner models.

    S(f) = (1 - eps)/(1 + (f/fa)**2) + eps/(1 + (f/fb)**2), with log10(fa),
    log10(fb) and log10(eps) linear in magnitude. At and below
    :attr:`m_single` the model is single-corner (eps = 1), with fc from
    :attr:`log10_fc_small`.

    A subclass only sets the coefficients, as class constants. The models
    have no parameters besides the duration weights.
    """

    kernel: ClassVar[ShapeKernel] = _ADDITIVE_OMEGA_SQUARED_KERNEL

    m_single: ClassVar[float] = -math.inf
    """Single-corner at or below this magnitude."""
    log10_fa: ClassVar[tuple[float, float]]
    """(a, b) in log10(fa) = a + b*M."""
    log10_fb: ClassVar[tuple[float, float]]
    """(a, b) in log10(fb) = a + b*M."""
    log10_eps: ClassVar[tuple[float, float]]
    """(a, b) in log10(eps) = a + b*M."""
    log10_fc_small: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.0)
    """(a, b, M_ref) in log10(fc) = a + b*(M - M_ref), at and below m_single."""

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, corner frequencies and eps at a magnitude; there is no stress."""
        m = magnitude
        if m <= self.m_single:
            fc = _log_linear(self.log10_fc_small, m)
            return self._terms(m, fc, fc, 1.0)
        return self._terms(
            m,
            10.0 ** (self.log10_fa[0] + self.log10_fa[1] * m),
            10.0 ** (self.log10_fb[0] + self.log10_fb[1] * m),
            10.0 ** (self.log10_eps[0] + self.log10_eps[1] * m),
        )

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb, eps)."""
        return (terms.fa, terms.fb, terms.eps)


@dataclass(frozen=True, slots=True, kw_only=True)
class Atkinson1993(EmpiricalAdditiveTwoCorner):
    """Atkinson (1993), BSSA 83, 1778-1798 (Fortran source 3)."""

    name: ClassVar[str] = "atkinson_1993"
    m_single: ClassVar[float] = 4.0
    log10_fa: ClassVar[tuple[float, float]] = (2.41, -0.533)
    log10_fb: ClassVar[tuple[float, float]] = (1.43, -0.188)
    log10_eps: ClassVar[tuple[float, float]] = (2.52, -0.637)
    log10_fc_small: ClassVar[tuple[float, float, float]] = (2.678, -0.5, 0.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtkinsonSilva1997(EmpiricalAdditiveTwoCorner):
    """Atkinson and Silva (1997), BSSA 87, 97-113 (Fortran source 4)."""

    name: ClassVar[str] = "atkinson_silva_1997"
    m_single: ClassVar[float] = 4.6
    log10_fa: ClassVar[tuple[float, float]] = (2.181, -0.496)
    log10_fb: ClassVar[tuple[float, float]] = (1.778, -0.302)
    log10_eps: ClassVar[tuple[float, float]] = (3.440, -0.746)
    log10_fc_small: ClassVar[tuple[float, float, float]] = (2.689, -0.5, 0.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtkinsonBoore1998California(EmpiricalAdditiveTwoCorner):
    """Atkinson and Boore (1998), BSSA 88, 917-934, California model (Fortran source 6)."""

    name: ClassVar[str] = "atkinson_boore_1998_california"
    m_single: ClassVar[float] = 4.8
    log10_fa: ClassVar[tuple[float, float]] = (2.181, -0.496)
    log10_fb: ClassVar[tuple[float, float]] = (1.308, -0.227)
    log10_eps: ClassVar[tuple[float, float]] = (3.223, -0.670)
    log10_fc_small: ClassVar[tuple[float, float, float]] = (2.617, -0.5, 0.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class AtkinsonSilva2000(EmpiricalAdditiveTwoCorner):
    """Atkinson and Silva (2000), BSSA 90, 255-274 (Fortran source 9)."""

    name: ClassVar[str] = "atkinson_silva_2000"
    m_single: ClassVar[float] = 2.4
    log10_fa: ClassVar[tuple[float, float]] = (2.181, -0.496)
    log10_fb: ClassVar[tuple[float, float]] = (2.41, -0.408)
    log10_eps: ClassVar[tuple[float, float]] = (0.605, -0.255)
    log10_fc_small: ClassVar[tuple[float, float, float]] = (1.431, -0.5, 2.4)


@dataclass(frozen=True, slots=True, kw_only=True)
class Atkinson2005(EmpiricalAdditiveTwoCorner):
    """Atkinson (2005) (Fortran source 10); two corners at all magnitudes."""

    name: ClassVar[str] = "atkinson_2005"
    log10_fa: ClassVar[tuple[float, float]] = (2.41, -0.533)
    log10_fb: ClassVar[tuple[float, float]] = (2.06, -0.345)
    log10_eps: ClassVar[tuple[float, float]] = (1.2, -0.3)


@dataclass(frozen=True, slots=True, kw_only=True)
class Haddon1996(_FixedCorners):
    """Haddon (1996), BSSA 86, 1300-1313 (Fortran source 5).

    S(f) = 1/((1 + (f/fa)**8)**(1/8) * (1 + (f/fb)**8)**(1/8)), with
    log10(fa) = 0.3 - 0.5*(M - 4) and log10(fb) = 1.4 - 0.5*(M - 4).
    """

    name: ClassVar[str] = "haddon_1996"
    kernel: ClassVar[ShapeKernel] = _HADDON_KERNEL

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0 and corner frequencies at a magnitude; there is no stress."""
        m = magnitude
        return self._terms(m, 10.0 ** (0.3 - 0.5 * (m - 4.0)), 10.0 ** (1.4 - 0.5 * (m - 4.0)))


@dataclass(frozen=True, slots=True, kw_only=True)
class BoatwrightChoy(_FixedCorners):
    """Boatwright and Choy, as used by Boore and Atkinson (1989) (Fortran source 7).

    S(f) is 1 below fa and fa/f above it, times 1/sqrt(1 + (f/fb)**2).
    Above M 5.3, log10(fa) = 3.409 - 0.681*M and log10(fb) = 1.495 - 0.319*M;
    at and below it, fa = fb = 0.634 * 10**(0.5*(5.3 - M)).
    """

    name: ClassVar[str] = "boatwright_choy"
    kernel: ClassVar[ShapeKernel] = _BOATWRIGHT_CHOY_KERNEL

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0 and corner frequencies at a magnitude; there is no stress."""
        m = magnitude
        if m <= 5.3:
            # 0.634 = 10**((log10(fa) + log10(fb))/2) at M 5.3
            fc = 0.634 * 10.0 ** (0.5 * (5.3 - m))
            return self._terms(m, fc, fc)
        return self._terms(m, 10.0 ** (3.409 - 0.681 * m), 10.0 ** (1.495 - 0.319 * m))


@dataclass(frozen=True, slots=True, kw_only=True)
class JoynerENA(_FixedCorners):
    """Joyner's ENA two-corner model for the SSHAC workshop (Fortran source 8).

    The shape of :class:`Joyner1984`, with log10(fa) = 2.312 - 0.5*M and
    log10(fb) = 3.609 - 0.5*M.
    """

    name: ClassVar[str] = "joyner_ena"
    kernel: ClassVar[ShapeKernel] = _JOYNER_KERNEL

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0 and corner frequencies at a magnitude; there is no stress."""
        m = magnitude
        return self._terms(m, 10.0 ** (2.312 - 0.5 * m), 10.0 ** (3.609 - 0.5 * m))


# Ji and Archuleta (2020) JA19_2S: log10(fa) = a + b*M, with a hinge at
# M 5.3, and log10(fb) = a + b*M
_JA19_FA_SMALL = (1.474, -0.415)
_JA19_FA_LARGE = (2.375, -0.585)
_JA19_FB = (3.250, -0.500)
_JA19_M_HINGE = 5.3


@dataclass(frozen=True, slots=True, kw_only=True)
class JiArchuleta2020(SourceSpectrum):
    """Ji and Archuleta (2020) model JA19_2S (Fortran source 13).

    S(f) = 1/((1 + (f/fa)**4)**(1/4) * (1 + (f/fb)**4)**(1/4)). The corner
    frequencies are fixed functions of magnitude, and the source duration is
    their empirical 10**(0.488*M - 2.244) s. The model has no parameters.
    """

    name: ClassVar[str] = "ji_archuleta_2020"
    kernel: ClassVar[ShapeKernel] = _JI_ARCHULETA_KERNEL

    def terms(self, magnitude: float, shear_velocity: float) -> SourceTerms:
        """M0, corner frequencies and equivalent single-corner stress at a magnitude."""
        m = magnitude
        a, b = _JA19_FA_SMALL if m <= _JA19_M_HINGE else _JA19_FA_LARGE
        fa = 10.0 ** (a + b * m)
        fb = 10.0 ** (_JA19_FB[0] + _JA19_FB[1] * m)
        # Equivalent single-corner stress: fc**2 = fa*fb in eq. 3
        log10_xi = math.log10(BRUNE_CONSTANT * shear_velocity) - 16.05 / 3.0 - 0.5 * m
        stress = 10.0 ** (-3.0 * log10_xi + 1.5 * (a + _JA19_FB[0] + (b + _JA19_FB[1]) * m))
        return SourceTerms(magnitude=m, m0=seismic_moment(m), stress=stress, fa=fa, fb=fb)

    def shape_args(self, terms: SourceTerms) -> tuple[float, ...]:
        """(fa, fb)."""
        return (terms.fa, terms.fb)

    def duration(self, terms: SourceTerms) -> float:
        """Source duration (s): 10**(0.488*M - 2.244)."""
        return float(10.0 ** (0.488 * terms.magnitude - 2.244))


SOURCE_MODELS: dict[str, type[SourceSpectrum]] = {
    cls.name: cls
    for cls in (
        SingleCorner,
        Joyner1984,
        Atkinson1993,
        AtkinsonSilva1997,
        Haddon1996,
        AtkinsonBoore1998California,
        BoatwrightChoy,
        JoynerENA,
        AtkinsonSilva2000,
        Atkinson2005,
        TwoCornerMultiplicative,
        TwoCornerAdditive,
        JiArchuleta2020,
    )
}
"""The source model classes by name, in the order of the Fortran numbers."""
