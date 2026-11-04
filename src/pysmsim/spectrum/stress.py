"""Stress parameter as a function of magnitude.

Each stress model is a small frozen dataclass holding only its own
coefficients. Calling it on a magnitude gives the stress parameter in bars::

    from pysmsim.spectrum.stress import LogLinearStress
    stress = LogLinearStress(stress_ref=100.0, dlog10_stress_dm=0.1)
    stress(7.0)  # 100 * 10**(0.1 * (7.0 - 6.5))

The models are picked by name in a TOML run file (``model = "log_linear"``);
:data:`STRESS_MODELS` maps each name to its class. The Fortran
``istress_model`` number of each model is given in its docstring.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "STRESS_MODELS",
    "HoerlStress",
    "LogLinearStress",
    "Stafford2022Stress",
    "StressModel",
]


class StressModel(ABC):
    """Base class of the stress models."""

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the model in a run file."""

    @abstractmethod
    def __call__(self, magnitude: float) -> float:
        """Stress parameter (bars) at a moment magnitude."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LogLinearStress(StressModel):
    """log10(stress) linear in magnitude (Fortran ``istress_model`` 1).

    stress = stress_ref * 10**(dlog10_stress_dm * (M - magnitude_ref)).

    Parameters
    ----------
    stress_ref : float
        Stress (bars) at ``magnitude_ref`` (Fortran ``stressc``).
    dlog10_stress_dm : float
        Slope of log10(stress) with magnitude (``dlsdm``).
    magnitude_ref : float
        Reference magnitude (``amagc``).
    """

    name: ClassVar[str] = "log_linear"

    stress_ref: float
    dlog10_stress_dm: float = 0.0
    magnitude_ref: float = 6.5

    def __post_init__(self) -> None:
        if self.stress_ref <= 0:
            raise ValueError(f"stress_ref must be positive, got {self.stress_ref}")

    def __call__(self, magnitude: float) -> float:
        """Stress parameter (bars) at a moment magnitude."""
        return float(
            self.stress_ref * 10.0 ** (self.dlog10_stress_dm * (magnitude - self.magnitude_ref))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HoerlStress(StressModel):
    """Hoerl's equation (Fortran ``istress_model`` 2).

    stress = c1 * M**c2 * exp(c3 * M), with the coefficients from Fortran
    ``scoeff01-03``.
    """

    name: ClassVar[str] = "hoerl"

    c1: float
    c2: float
    c3: float

    def __call__(self, magnitude: float) -> float:
        """Stress parameter (bars) at a moment magnitude."""
        return float(self.c1 * magnitude**self.c2 * math.exp(self.c3 * magnitude))


@dataclass(frozen=True, slots=True, kw_only=True)
class Stafford2022Stress(StressModel):
    """Stafford et al. (2022) (Fortran ``istress_model`` 3).

    ln(stress) = c1 + c2 * min(M - 5, 0), with the coefficients from Fortran
    ``scoeff01-02``.

    Stafford et al. (2022) add (c3 + c4/cosh(2*max(M - 4.5, 0))) * DeltaZtor.
    SMSIM sets DeltaZtor to 0, so that term (``scoeff03-04``) is left out
    here.
    """

    name: ClassVar[str] = "stafford_2022"

    c1: float
    c2: float

    def __call__(self, magnitude: float) -> float:
        """Stress parameter (bars) at a moment magnitude."""
        return math.exp(self.c1 + self.c2 * min(magnitude - 5.0, 0.0))


STRESS_MODELS: dict[str, type[StressModel]] = {
    cls.name: cls for cls in (LogLinearStress, HoerlStress, Stafford2022Stress)
}
"""The stress model classes by name."""
