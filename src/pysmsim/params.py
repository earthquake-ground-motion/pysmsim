"""Model parameters for SMSIM simulations.

Each class here holds one group of settings from an SMSIM ``.params`` file
(the ``smsim.fi`` common blocks in the Fortran). The classes are frozen
dataclasses: they hold settings only, never values derived from a magnitude
or distance, so one ``Params`` object can be shared by every scenario.

Options are chosen by name. The source spectrum is one of the classes in
:mod:`pysmsim.spectrum.source` (for example
:class:`~pysmsim.spectrum.source.SingleCorner`), each holding only its own
parameters; the other options are string enums, each listing the Fortran
code it replaces. Units follow SMSIM: density in g/cm^3, shear
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

from pysmsim.spectrum.source import SourceSpectrum

__all__ = [
    "Filters",
    "FiniteFault",
    "FiniteFaultModel",
    "HighCutFilter",
    "HighCutModel",
    "LowCutFilter",
    "Medium",
    "NoiseDistribution",
    "OscillatorCorrection",
    "Params",
    "PathDuration",
    "Q",
    "QModel",
    "RVConfig",
    "RpsCombination",
    "Site",
    "Spreading",
    "SpreadingModel",
    "SpreadingSegment",
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


class RpsCombination(StrEnum):
    """How the finite-fault factor h is combined with distance (``iflag_f_ff``)."""

    QUADRATURE = "quadrature"
    """1: Rps = sqrt(R**2 + h**2)."""
    SUM = "sum"
    """2: Rps = R + h."""


class FiniteFaultModel(StrEnum):
    """Magnitude dependence of the finite-fault factor h (``nlines_f_ff``)."""

    NONE = "none"
    """0: no finite-fault factor; Rps = R."""
    ONE_LINE = "one_line"
    """1: log10(h) = c1 + c2*M."""
    TWO_LINES = "two_lines"
    """2: two lines, log10(h) = c1 + c2*M and c3 + c4*M, meeting at Mh."""
    TWO_LINES_SMOOTHED = "two_lines_smoothed"
    """3: two lines joined by a cubic over delta_m magnitude units (BT15)."""
    GENERALIZED = "generalized"
    """999: one-equation approximation of the smoothed lines; delta_m is hrat."""
    STAFFORD_2022 = "stafford_2022"
    """1000: Stafford et al. (2022), eq. 27, in natural logs; c1-c4 and delta_m
    are h_alpha, h_beta, h_gamma, h_delta and h_epsilon."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteFault:
    """Finite-fault factor h, which turns R into the point-source distance Rps.

    Fortran: ``iflag_f_ff, nlines_f_ff, c1, c2, c3, c4, DeltaM`` (see
    ``rmod_calc``; Boore and Thompson, 2015).

    Parameters
    ----------
    model : FiniteFaultModel
        Functional form of h(M).
    combination : RpsCombination
        How h combines with distance.
    c1, c2, c3, c4 : float
        Coefficients of the lines (log10) or of the Stafford et al. model.
    delta_m : float
        Width of the transition (``two_lines_smoothed``), hrat
        (``generalized``) or h_epsilon (``stafford_2022``).
    """

    model: FiniteFaultModel = FiniteFaultModel.NONE
    combination: RpsCombination = RpsCombination.QUADRATURE
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    delta_m: float = 0.0


class SpreadingModel(StrEnum):
    """Geometrical-spreading model (Fortran ``i_gsprd_option``)."""

    HINGED = "hinged"
    """0: hinged line segments, slopes optionally magnitude dependent."""
    ATKINSON_2011 = "atkinson_2011"
    """1: Atkinson's November 2011 ENA proposal (frequency dependent)."""
    BOORE_TRIAL = "boore_trial"
    """2: Boore's trial ENA spreading #1 (frequency dependent)."""
    ATKINSON_2012 = "atkinson_2012"
    """3: Atkinson (2012) NGA-East report (frequency dependent)."""
    ATKINSON_BOORE_2014 = "atkinson_boore_2014"
    """4: Atkinson and Boore (2014), BSSA 104, 513-528 (frequency dependent)."""
    ATKINSON_BOORE_2014_R10 = "atkinson_boore_2014_r10"
    """5: AB14Rlt10mod, 1/R within 10 km (frequency dependent)."""
    STAFFORD_GENERALIZED = "stafford_generalized"
    """6: Stafford's two segments with a smooth transition."""
    STAFFORD_OPTIMAL = "stafford_optimal"
    """7: Stafford et al. (2022) optimal model (Rps in the first term and Rrup
    in the second; the params-file comments say Rjb, but the code uses Rrup)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SpreadingSegment:
    """One segment of the hinged spreading model.

    The slope beyond ``r_start`` is ``a + b*(M - m_ref)``. Fortran:
    ``rlow, a_s, b_s, m_s``.
    """

    r_start: float
    a: float
    b: float = 0.0
    m_ref: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Spreading:
    """Geometrical spreading.

    Parameters
    ----------
    model : SpreadingModel
        Spreading model.
    r_ref : float
        Reference distance (km), usually 1 (``r_ref``).
    segments : tuple of SpreadingSegment
        ``hinged`` only: the segments, in order of increasing ``r_start``.
    gamma1, r_t, gamma2 : float
        ``stafford_generalized`` and ``stafford_optimal``: near slope,
        transition distance and far slope (``g1_gsprd, rt_gsprd, g2_gsprd``).
    g_ratio : float
        ``stafford_generalized`` only: actual spreading divided by the near
        spreading at r_t (``grat``).
    frequency_dependent : tuple of float
        Models 1-5: ``r1, p_lf, p_hf, p_far, ft1, ft2`` (``r1_dmb_gsprd``
        etc.); only models 2 and 4 use them.
    """

    model: SpreadingModel = SpreadingModel.HINGED
    r_ref: float = 1.0
    segments: tuple[SpreadingSegment, ...] = ()
    gamma1: float = 0.0
    r_t: float = 0.0
    gamma2: float = 0.0
    g_ratio: float = 1.0
    frequency_dependent: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.model is SpreadingModel.HINGED:
            if not self.segments:
                raise ValueError("hinged spreading needs at least one segment")
            _check_increasing("segment r_start", tuple(s.r_start for s in self.segments))
        if len(self.frequency_dependent) != 6:
            raise ValueError("frequency_dependent needs 6 values")


class QModel(StrEnum):
    """Form of Q(f) (the sign of the first Fortran Q parameter)."""

    PIECEWISE = "piecewise"
    """fr1 > 0: power laws below ft1 and above ft2, joined between them."""
    STAFFORD_2022 = "stafford_2022"
    """fr1 < 0: Q = q0 * f**eta(M), eta = eta_a + eta_b*tanh(M - eta_g)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Q:
    """Anelastic attenuation.

    Fortran: ``fr1, qr1, s1, ft1, ft2, fr2, qr2, s2, c_q``.

    Parameters
    ----------
    model : QModel
        Form of Q(f).
    c_q : float
        Velocity used in the Q operator (km/s).
    fr1, qr1, s1, ft1, ft2, fr2, qr2, s2 : float
        ``piecewise`` only: Q = qr1*(f/fr1)**s1 for f <= ft1 and
        qr2*(f/fr2)**s2 for f >= ft2, joined by a power law between.
    q0, eta_a, eta_b, eta_g : float
        ``stafford_2022`` only.
    """

    model: QModel = QModel.PIECEWISE
    c_q: float = 3.5
    fr1: float = 1.0
    qr1: float = 0.0
    s1: float = 0.0
    ft1: float = 1.0
    ft2: float = 1.0
    fr2: float = 1.0
    qr2: float = 0.0
    s2: float = 0.0
    q0: float = 0.0
    eta_a: float = 0.0
    eta_b: float = 0.0
    eta_g: float = 0.0

    def __post_init__(self) -> None:
        if self.model is QModel.PIECEWISE and (self.fr1 <= 0 or self.fr2 <= 0):
            raise ValueError("piecewise Q needs positive fr1 and fr2")


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Site:
    """Crustal amplification and high-frequency diminution.

    Fortran: ``namps, famp, amp`` and ``fm, akappa, dkappadmag, amagkref``.

    Parameters
    ----------
    amp_frequencies, amplifications : tuple of float
        Crustal amplification, interpolated linearly in frequency and
        logarithmically in amplitude.
    kappa0 : float
        Kappa (s) at ``kappa_magnitude_ref``.
    dkappa_dmag : float
        Change of kappa per magnitude unit.
    kappa_magnitude_ref : float
        Reference magnitude for kappa.
    fmax : float
        fmax (Hz); 0 means not used.
    """

    amp_frequencies: tuple[float, ...]
    amplifications: tuple[float, ...]
    kappa0: float = 0.0
    dkappa_dmag: float = 0.0
    kappa_magnitude_ref: float = 0.0
    fmax: float = 0.0

    def __post_init__(self) -> None:
        _check_same_length(
            "amp_frequencies and amplifications", self.amp_frequencies, self.amplifications
        )
        if not self.amp_frequencies:
            raise ValueError("site amplification needs at least one point")
        _check_increasing("amp_frequencies", self.amp_frequencies)
        if any(a <= 0 for a in self.amplifications):
            raise ValueError("amplifications must be positive")


class HighCutModel(StrEnum):
    """High-cut filter shape (Fortran ``itype_hcfilt``)."""

    NONE = "none"
    """0: no filter."""
    RAISED_COSINE = "raised_cosine"
    """1: raised half cycle of a cosine."""
    QUARTER_COSINE = "quarter_cosine"
    """2: quarter cycle of a cosine."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LowCutFilter:
    """Acausal Butterworth low-cut filter (``fcut, nslope``).

    ``fcut`` = 0 means no filter. ``order`` is the order of each of the two
    passes (Fortran ``norder = nslope/2``), so the filter is
    1/(1 + (fcut/f)**(2*order)).
    """

    fcut: float = 0.0
    order: int = 4

    def __post_init__(self) -> None:
        if self.order < 2 or self.order % 2:
            raise ValueError("low-cut order must be an even number of at least 2")


@dataclass(frozen=True, slots=True, kw_only=True)
class HighCutFilter:
    """High-cut filter between ``f1`` and ``f2`` (``fhc1, fhc2, eta_hcfilt``)."""

    model: HighCutModel = HighCutModel.NONE
    f1: float = 0.0
    f2: float = 0.0
    eta: float = 1.0

    def __post_init__(self) -> None:
        if self.model is not HighCutModel.NONE and not 0 < self.f1 < self.f2:
            raise ValueError("high-cut filter needs 0 < f1 < f2")


@dataclass(frozen=True, slots=True, kw_only=True)
class Filters:
    """Low-cut and high-cut filters applied to the spectrum."""

    low_cut: LowCutFilter = field(default_factory=LowCutFilter)
    high_cut: HighCutFilter = field(default_factory=HighCutFilter)


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
    finite_fault: FiniteFault
    spreading: Spreading
    q: Q
    path_duration: PathDuration
    site: Site
    filters: Filters = field(default_factory=Filters)
    rv: RVConfig = field(default_factory=RVConfig)
    td: TDConfig = field(default_factory=TDConfig)
