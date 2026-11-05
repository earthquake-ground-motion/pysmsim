"""Model parameters for SMSIM simulations.

Each class here holds one group of settings from an SMSIM ``.params`` file
(the ``smsim.fi`` common blocks in the Fortran). The classes are frozen
dataclasses: they hold settings only, never values derived from a magnitude
or distance, so one ``Params`` object can be shared by every scenario.

Options are chosen by name. The source spectrum, finite-fault factor,
geometrical spreading, Q and high-cut filter are each one of the classes in
the modules of :mod:`pysmsim.spectrum` (for example
:class:`~pysmsim.spectrum.source.SingleCorner`), each holding only its own
parameters; the other options are string enums, each listing the Fortran
code it replaces. The site terms and the filters are
:class:`~pysmsim.spectrum.site.Site` and
:class:`~pysmsim.spectrum.filters.Filters`. Units follow SMSIM: density in g/cm^3, shear
velocity in km/s, distance in km, stress in bars, frequency in Hz.

The objects can be built directly, for example in a notebook::

    from pysmsim.params import Medium
    medium = Medium(density=2.8, shear_velocity=3.6)

or read from a TOML run file with :func:`pysmsim.io.config.load_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

from pysmsim.spectrum.attenuation import QModel
from pysmsim.spectrum.filters import Filters
from pysmsim.spectrum.finite_fault import FiniteFaultFactor, NoFiniteFault
from pysmsim.spectrum.site import Site
from pysmsim.spectrum.source import SourceSpectrum
from pysmsim.spectrum.spreading import GeometricalSpreading

__all__ = [
    "Medium",
    "NoiseDistribution",
    "OscillatorCorrection",
    "Params",
    "PathDuration",
    "RVConfig",
    "TDConfig",
    "WindowShape",
]


def _check_increasing(name: str, values: tuple[float, ...]) -> None:
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing, got {values}")


def _check_same_length(names: str, *arrays: tuple[float, ...]) -> None:
    if len({len(a) for a in arrays}) != 1:
        raise ValueError(f"{names} must have the same length")


@dataclass(frozen=True, slots=True, kw_only=True)
class Medium:
    """Properties of the source region and the radiation constant.

    Fortran: ``rho, beta, prtitn, rtp, fs``.

    Parameters
    ----------
    density : float
        Density near the source (g/cm^3), ``rho``.
    shear_velocity : float
        Shear-wave velocity near the source (km/s), ``beta``.
    partition : float
        Partition of energy into two horizontal components, ``prtitn``
        (usually 1/sqrt(2)).
    radiation : float
        Average radiation pattern, ``rtp`` (usually 0.55).
    free_surface : float
        Free-surface amplification, ``fs`` (usually 2).
    """

    density: float
    shear_velocity: float
    partition: float = 0.707
    radiation: float = 0.55
    free_surface: float = 2.0

    def __post_init__(self) -> None:
        if self.density <= 0 or self.shear_velocity <= 0:
            raise ValueError("density and shear_velocity must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PathDuration:
    """Path duration as a function of distance (``nknots, rdur, dur, slast``).

    Linear between the knots, ``slope_beyond`` (s/km) past the last one.
    """

    distances: tuple[float, ...]
    durations: tuple[float, ...]
    slope_beyond: float = 0.0

    def __post_init__(self) -> None:
        _check_same_length("distances and durations", self.distances, self.durations)
        if not self.distances:
            raise ValueError("path duration needs at least one knot")
        _check_increasing("path-duration distances", self.distances)


class OscillatorCorrection(StrEnum):
    """RMS-duration correction for oscillators (Fortran ``osc_crrctn``)."""

    NONE = "none"
    """0: no correction."""
    BJ84 = "bj84"
    """1: Boore and Joyner (1984)."""
    LP99 = "lp99"
    """2: Liu and Pezeshk (1999)."""
    BT15_WNA = "bt15_wna"
    """3: Boore and Thompson (2015), first table (WNA)."""
    BT15_ENA = "bt15_ena"
    """4: Boore and Thompson (2015), second table (ENA)."""
    BT15_AVERAGE = "bt15_average"
    """5: average of the two BT15 tables."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RVConfig:
    """Random-vibration settings (``zup, eps_int, amp_cutoff, osc_crrctn``).

    Parameters
    ----------
    zup : float
        Upper limit for the Cartwright and Longuet-Higgins peak-factor
        integration.
    eps_int : float
        Relative accuracy of the integrations.
    amp_cutoff : float
        Sets the upper integration frequency fup, where the kappa (or fmax)
        diminution falls to this value (Fortran ``get_fup``).
    osc_correction : OscillatorCorrection
        RMS-duration correction.
    bt15_tables : tuple of str
        Names of the two BT15 coefficient tables (bundled package data).
    """

    zup: float = 10.0
    eps_int: float = 1.0e-5
    amp_cutoff: float = 1.0e-3
    osc_correction: OscillatorCorrection = OscillatorCorrection.BT15_WNA
    bt15_tables: tuple[str, str] = ("bt15_wna_acr", "bt15_ena_scr")


class WindowShape(StrEnum):
    """Time-domain window (Fortran ``indxwind``)."""

    BOX = "box"
    """0: box window."""
    EXPONENTIAL = "exponential"
    """1: Saragoni-Hart exponential window."""


class NoiseDistribution(StrEnum):
    """Distribution of the time-domain noise (Fortran ``iran_type``)."""

    GAUSSIAN = "gaussian"
    """0: normal distribution."""
    UNIFORM = "uniform"
    """1: uniform distribution."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TDConfig:
    """Time-domain settings (window and timing lines of the params file).

    Parameters
    ----------
    window : WindowShape
        Window shape.
    taper : float
        Box window only: fraction of the motion duration added for the
        tapers (half at each end).
    eps_w : float
        Exponential window: fraction of te at which the window peaks.
    eta_w : float
        Exponential window: amplitude at te, as a fraction of the peak.
    f_tb2te : float
        Exponential window: te = f_tb2te * (source + path duration).
    f_te_xtnd : float
        Exponential window: window length tw = f_te_xtnd * te.
    dur_fctr : float
        The series is long enough for dur_fctr times the motion duration
        after ``tshift`` (then rounded up to a power of two).
    dt : float
        Time step (s).
    tshift : float
        Time before the motion starts (s); raised to the filter pad length
        when a low-cut filter is used.
    seed : int
        Random seed.
    nsims : int
        Number of simulations.
    noise : NoiseDistribution
        Noise distribution.
    """

    window: WindowShape = WindowShape.EXPONENTIAL
    taper: float = 0.05
    eps_w: float = 0.2
    eta_w: float = 0.05
    f_tb2te: float = 2.0
    f_te_xtnd: float = 1.0
    dur_fctr: float = 1.3
    dt: float = 0.005
    tshift: float = 50.0
    seed: int = 123
    nsims: int = 800
    noise: NoiseDistribution = NoiseDistribution.GAUSSIAN

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.nsims < 1:
            raise ValueError("dt must be positive and nsims at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class Params:
    """All model parameters: the content of one SMSIM ``.params`` file."""

    title: str
    medium: Medium
    source: SourceSpectrum
    finite_fault: FiniteFaultFactor = field(default_factory=NoFiniteFault)
    spreading: GeometricalSpreading
    q: QModel
    path_duration: PathDuration
    site: Site
    filters: Filters = field(default_factory=Filters)
    rv: RVConfig = field(default_factory=RVConfig)
    td: TDConfig = field(default_factory=TDConfig)
