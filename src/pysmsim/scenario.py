"""Earthquake scenarios and the grid of scenarios a run loops over.

A :class:`Scenario` is one magnitude and distance. A :class:`ScenarioGrid`
holds what an SMSIM loop-driver control file (``.ctl``) specifies: the
magnitudes, distances and oscillator periods to loop over, how Rjb is
converted to Rrup, and optional replacements for kappa_0 and the stress
parameter. Both are frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "DepthFunction",
    "DistanceMetric",
    "Mechanism",
    "OutputFormat",
    "OutputSettings",
    "RjbToRrup",
    "Scenario",
    "ScenarioGrid",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Scenario:
    """One earthquake scenario.

    Parameters
    ----------
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km). This is the distance R that the finite-fault
        factor turns into the point-source distance Rps.
    rjb : float or None
        Joyner-Boore distance (km), when the scenario was specified by Rjb.
        It is carried through to the output; the spectrum uses only Rrup.
    """

    magnitude: float
    rrup: float
    rjb: float | None = None

    def __post_init__(self) -> None:
        if self.rrup <= 0:
            raise ValueError(f"rrup must be positive, got {self.rrup}")
        if self.rjb is not None and not 0 <= self.rjb <= self.rrup:
            raise ValueError(f"rjb must be between 0 and rrup, got {self.rjb}")


class DistanceMetric(StrEnum):
    """Distance measure of the loop distances."""

    RJB = "rjb"
    """Joyner-Boore distance, converted to Rrup as set by ``rjb_to_rrup``."""
    RRUP = "rrup"
    """Rupture distance, used as it is."""


class RjbToRrup(StrEnum):
    """Depth h used in Rrup = sqrt(Rjb**2 + h**2) (Fortran ``case_h4rjb2rrup``)."""

    NONE = "none"
    """0: h = 0, so Rrup = Rjb."""
    MAGNITUDE_FUNCTION = "magnitude_function"
    """1: h(M) from :class:`DepthFunction`."""
    CY14_ZTOR = "cy14_ztor"
    """2: h = the Chiou and Youngs (2014) Ztor(M) for the mechanism."""


class Mechanism(StrEnum):
    """Fault mechanism, for the CY14 Ztor(M) relation (Fortran ``mech``)."""

    STRIKE_SLIP = "strike_slip"
    """1"""
    NORMAL = "normal"
    """2 (the same Ztor(M) as strike-slip)."""
    REVERSE = "reverse"
    """3"""


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthFunction:
    """Depth h(M) for the ``magnitude_function`` Rjb-to-Rrup conversion.

    h = h1 for M <= m1, h = h2 for M >= m2, and
    h = h1 + c1*(M-m1) + c2*(M-m1)**2 + c3*(M-m1)**3 in between.
    """

    m1: float = 3.75
    h1: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    m2: float = 7.5
    h2: float = 0.0

    def depth(self, magnitude: float) -> float:
        """Depth h (km) at a moment magnitude (case 1 of Fortran ``h4rjb2rrup_f``)."""
        if magnitude <= self.m1:
            return self.h1
        if magnitude >= self.m2:
            return self.h2
        dm = magnitude - self.m1
        return self.h1 + self.c1 * dm + self.c2 * dm**2 + self.c3 * dm**3


class OutputFormat(StrEnum):
    """File format of the results table."""

    CSV = "csv"
    PARQUET = "parquet"
    COL = "col"
    """SMSIM column layout, for comparison with the Fortran."""
    HDF5 = "hdf5"
    """One HDF5 file with the scenario, FAS and RV tables (:mod:`pysmsim.io.hdf5`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputSettings:
    """Where and how a run writes its results."""

    path: str = "results.csv"
    format: OutputFormat = OutputFormat.CSV


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioGrid:
    """The magnitudes, distances and ground-motion measures of a run.

    Every combination of magnitude, distance, period, kappa_0 and stress is
    computed (the loop order of the Fortran drivers only changes the order
    of the output rows).

    Parameters
    ----------
    magnitudes : tuple of float
        Moment magnitudes.
    distances : tuple of float
        Distances (km), of the type given by ``distance_metric``.
    distance_metric : DistanceMetric
        Whether ``distances`` are Rjb or Rrup.
    rjb_to_rrup : RjbToRrup
        Conversion of Rjb to Rrup (for ``distance_metric = rjb``).
    mechanism : Mechanism
        Fault mechanism for the ``cy14_ztor`` conversion.
    depth_function : DepthFunction
        Depth for the ``magnitude_function`` conversion.
    periods : tuple of float
        Oscillator periods (s) for response spectra.
    pga, pgv : bool
        Whether to compute PGA and PGV.
    damping : float
        Oscillator damping, as a fraction of critical.
    frequencies : tuple of float
        Frequencies (Hz) at which a run also gives the Fourier acceleration
        spectrum (see :func:`pysmsim.drivers.run.run`); none by default.
    kappa0 : tuple of float or None
        If given, loop over these kappa_0 values instead of the one in the
        parameters (and make kappa independent of magnitude, as the Fortran
        does).
    stress : tuple of float or None
        If given, loop over these stress parameters (bars) instead of the
        stress model in the parameters.
    """

    magnitudes: tuple[float, ...]
    distances: tuple[float, ...]
    distance_metric: DistanceMetric = DistanceMetric.RJB
    rjb_to_rrup: RjbToRrup = RjbToRrup.CY14_ZTOR
    mechanism: Mechanism = Mechanism.STRIKE_SLIP
    depth_function: DepthFunction = field(default_factory=DepthFunction)
    periods: tuple[float, ...] = ()
    pga: bool = True
    pgv: bool = True
    damping: float = 0.05
    frequencies: tuple[float, ...] = ()
    kappa0: tuple[float, ...] | None = None
    stress: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.magnitudes or not self.distances:
            raise ValueError("a scenario grid needs at least one magnitude and one distance")
        if any(r < 0 for r in self.distances):
            raise ValueError("distances must not be negative")
        if any(p <= 0 for p in self.periods):
            raise ValueError("periods must be positive (use pga and pgv for peak motions)")
        if not (self.periods or self.pga or self.pgv):
            raise ValueError("a scenario grid needs at least one period, pga or pgv")
        if not 0 < self.damping < 1:
            raise ValueError(f"damping must be between 0 and 1, got {self.damping}")
        if any(f < 0 for f in self.frequencies):
            raise ValueError("frequencies must not be negative")
