"""Finite-fault factor h and the point-source distance Rps.

SMSIM computes the spectrum of a point source at the distance Rps, which
combines the rupture distance with a magnitude-dependent finite-fault
factor h: Rps = sqrt(R**2 + h**2), or R + h (Fortran ``rmod_calc``; Boore
and Thompson, 2015). Each form of h(M) is a class derived from
:class:`FiniteFaultFactor` that holds only its own coefficients::

    from pysmsim.spectrum.finite_fault import TwoLinesSmoothed
    ff = TwoLinesSmoothed(c1=-1.72, c2=0.43, c3=-0.405, c4=0.235, delta_m=2.0)
    ff.h(6.0)          # finite-fault factor (km)
    ff.rps(20.0, 6.0)  # point-source distance (km) for Rrup = 20 km

The models are picked by name in a TOML run file (``model =
"two_lines_smoothed"``); :data:`FINITE_FAULT_MODELS` maps each name to its
class. The Fortran ``nlines_f_ff`` of each model is given in its docstring.

References
----------
Boore, D. M. and E. M. Thompson (2015). Revisions to some parameters used
in stochastic-method simulations of ground motion. Bull. Seismol. Soc. Am.
105, 1029-1041.

Boore, D. M. Representing the Boore and Thompson (2015) finite-fault factor
by a single equation. Note in the SMSIM distribution
(``generalize_boore-thompson_2015_finite-fault_factor.v3.pdf``).

Stafford, P. J., et al. (2022), eq. 27, as coded in SMSIM's ``rmod_calc``
(the paper is not in the SMSIM distribution).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

__all__ = [
    "FINITE_FAULT_MODELS",
    "FiniteFaultFactor",
    "Generalized",
    "NoFiniteFault",
    "OneLine",
    "RpsCombination",
    "Stafford2022FiniteFault",
    "TwoLines",
    "TwoLinesSmoothed",
]


class RpsCombination(StrEnum):
    """How the finite-fault factor h is combined with distance (``iflag_f_ff``)."""

    QUADRATURE = "quadrature"
    """1: Rps = sqrt(R**2 + h**2)."""
    SUM = "sum"
    """2: Rps = R + h."""


class FiniteFaultFactor(ABC):
    """Base class of the finite-fault factor models.

    A subclass sets :attr:`name` and implements :meth:`h`.
    """

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the model in a run file."""

    @abstractmethod
    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""

    @abstractmethod
    def rps(self, distance: float, magnitude: float) -> float:
        """Point-source distance Rps (km) (Fortran ``rmod_calc``).

        Parameters
        ----------
        distance : float
            Rupture distance (km).
        magnitude : float
            Moment magnitude.
        """


@dataclass(frozen=True, slots=True, kw_only=True)
class NoFiniteFault(FiniteFaultFactor):
    """No finite-fault factor: Rps = R (Fortran ``iflag_f_ff`` 0)."""

    name: ClassVar[str] = "none"

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km): 0."""
        return 0.0

    def rps(self, distance: float, magnitude: float) -> float:
        """Point-source distance Rps (km): the rupture distance."""
        return distance


@dataclass(frozen=True, slots=True, kw_only=True)
class _CombinedWithDistance(FiniteFaultFactor):
    """Base class of the models with an h; combines h with distance.

    Parameters
    ----------
    combination : RpsCombination
        How h is combined with distance (Fortran ``iflag_f_ff``).
    """

    combination: RpsCombination = RpsCombination.QUADRATURE

    def rps(self, distance: float, magnitude: float) -> float:
        """Point-source distance Rps (km): sqrt(R**2 + h**2) or R + h."""
        h = self.h(magnitude)
        if self.combination is RpsCombination.SUM:
            return distance + h
        return math.sqrt(distance**2 + h**2)


@dataclass(frozen=True, slots=True, kw_only=True)
class OneLine(_CombinedWithDistance):
    """log10(h) = c1 + c2*M (Fortran ``nlines_f_ff`` 1).

    For example Atkinson and Silva (2000): c1 = -0.05, c2 = 0.15, or Yenier
    and Atkinson (2014): c1 = -1.72, c2 = 0.43.
    """

    name: ClassVar[str] = "one_line"

    c1: float
    c2: float

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""
        return float(10.0 ** (self.c1 + self.c2 * magnitude))


def _hinge(c1: float, c2: float, c3: float, c4: float) -> float:
    """Magnitude where log10(h) = c1 + c2*M meets c3 + c4*M."""
    if c4 == c2:
        raise ValueError("the two lines of the finite-fault factor are parallel (c2 = c4)")
    return (c1 - c3) / (c4 - c2)


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoLines(_CombinedWithDistance):
    """Two lines in log10(h) (Fortran ``nlines_f_ff`` 2).

    log10(h) = c1 + c2*M up to the magnitude Mh where the lines meet, and
    c3 + c4*M above it.
    """

    name: ClassVar[str] = "two_lines"

    c1: float
    c2: float
    c3: float
    c4: float

    def __post_init__(self) -> None:
        _hinge(self.c1, self.c2, self.c3, self.c4)

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""
        if magnitude <= _hinge(self.c1, self.c2, self.c3, self.c4):
            return float(10.0 ** (self.c1 + self.c2 * magnitude))
        return float(10.0 ** (self.c3 + self.c4 * magnitude))


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoLinesSmoothed(_CombinedWithDistance):
    """Two lines in log10(h) joined by a cubic (Fortran ``nlines_f_ff`` 3; BT15).

    log10(h) = c1 + c2*M below Mh - delta_m/2 and c3 + c4*M above
    Mh + delta_m/2, where Mh is the magnitude where the lines meet. Between
    them a cubic in M matches the values and slopes of both lines (Boore and
    Thompson, 2015).

    Parameters
    ----------
    c1, c2, c3, c4 : float
        Coefficients of the two lines.
    delta_m : float
        Width of the transition (magnitude units), positive.
    """

    name: ClassVar[str] = "two_lines_smoothed"

    c1: float
    c2: float
    c3: float
    c4: float
    delta_m: float

    def __post_init__(self) -> None:
        _hinge(self.c1, self.c2, self.c3, self.c4)
        if self.delta_m <= 0:
            raise ValueError(f"delta_m must be positive, got {self.delta_m}")

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""
        c1, c2, c3, c4, dm = self.c1, self.c2, self.c3, self.c4, self.delta_m
        mh = _hinge(c1, c2, c3, c4)
        m1 = mh - 0.5 * dm
        m2 = mh + 0.5 * dm
        if magnitude <= m1:
            return float(10.0 ** (c1 + c2 * magnitude))
        if magnitude >= m2:
            return float(10.0 ** (c3 + c4 * magnitude))
        # Cubic coefficients, as in SMSIM's get_params
        c1mod = c1 + c2 * m1
        c3mod = c3 + c4 * m2
        delta_a = c3mod - c1mod
        delta_b = c4 - c2
        a2 = 2.0 * delta_a / dm**2 - 0.5 * (3.0 * c2 + c4) / dm
        a3 = 0.5 * (delta_b - 2.0 * (delta_a / dm - c2)) / dm**2
        x = magnitude - m1
        return float(10.0 ** (c1mod + c2 * x + a2 * x**2 + a3 * x**3))


@dataclass(frozen=True, slots=True, kw_only=True)
class Generalized(_CombinedWithDistance):
    """One-equation form of the smoothed two lines (Fortran ``nlines_f_ff`` 999).

    log10(h) = log10(ht) + c2*(M - Mt) + ((c4 - c2)/xi)
    * log10((10**(xi*(M - Mt)) + 1)/2), where Mt is the magnitude where the
    lines meet, ht = h_ratio * 10**(c1 + c2*Mt) and
    xi = (c4 - c2)*log10(2)/log10(h_ratio).

    Parameters
    ----------
    c1, c2, c3, c4 : float
        Coefficients of the two lines.
    h_ratio : float
        h at Mt divided by the value of the lines there (Fortran ``hrat``,
        given in place of DeltaM). Positive and not 1; below 1 when the
        second line is flatter, as in BT15 (Boore suggests 0.8939).
    """

    name: ClassVar[str] = "generalized"

    c1: float
    c2: float
    c3: float
    c4: float
    h_ratio: float

    def __post_init__(self) -> None:
        _hinge(self.c1, self.c2, self.c3, self.c4)
        if self.h_ratio <= 0 or self.h_ratio == 1:
            raise ValueError(f"h_ratio must be positive and not 1, got {self.h_ratio}")

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""
        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
        mt = _hinge(c1, c2, c3, c4)
        ht = self.h_ratio * 10.0 ** (c1 + c2 * mt)
        xi = (c4 - c2) * math.log10(2.0) / math.log10(self.h_ratio)
        dm = magnitude - mt
        return float(
            10.0
            ** (
                math.log10(ht)
                + c2 * dm
                + (c4 - c2) * (1.0 / xi) * math.log10((10.0 ** (xi * dm) + 1.0) / 2.0)
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stafford2022FiniteFault(_CombinedWithDistance):
    """Stafford et al. (2022), eq. 27 (Fortran ``nlines_f_ff`` 1000).

    ln(h) = h_alpha + h_beta*M + ((h_beta - h_gamma)/h_delta)
    * ln(1 + exp(-h_delta*(M - h_epsilon))).

    In a params file the coefficients are c1, c2, c3, c4 and DeltaM, in
    that order. Stafford et al. use Rps = Rrup + h (``combination =
    "sum"``).
    """

    name: ClassVar[str] = "stafford_2022"

    h_alpha: float
    h_beta: float
    h_gamma: float
    h_delta: float
    h_epsilon: float

    def __post_init__(self) -> None:
        if self.h_delta == 0:
            raise ValueError("h_delta must not be 0")

    def h(self, magnitude: float) -> float:
        """Finite-fault factor h (km) at a moment magnitude."""
        ln_h = (
            self.h_alpha
            + self.h_beta * magnitude
            + ((self.h_beta - self.h_gamma) / self.h_delta)
            * math.log(1.0 + math.exp(-self.h_delta * (magnitude - self.h_epsilon)))
        )
        return math.exp(ln_h)


FINITE_FAULT_MODELS: dict[str, type[FiniteFaultFactor]] = {
    cls.name: cls
    for cls in (
        NoFiniteFault,
        OneLine,
        TwoLines,
        TwoLinesSmoothed,
        Generalized,
        Stafford2022FiniteFault,
    )
}
"""The finite-fault factor classes by name."""
