"""Durations for random vibration: excitation and RMS durations.

The excitation duration Dex is the source plus the path duration (Fortran
``dursource`` and ``durpath``). The number of zero crossings and extrema
is counted over Dex. The RMS motion is sqrt(m0/Drms), and for an
oscillator Drms is longer than Dex, because the oscillator keeps ringing
after the excitation ends. SMSIM offers several corrections for this
(``osc_crrctn``); each is a class derived from :class:`OscillatorCorrection`::

    from pysmsim.rv.duration import BooreJoyner1984
    correction = BooreJoyner1984()
    correction.rms_duration(dur_ex, period=1.0, damping=0.05, magnitude=6.0, rps=20.0,
                            bandwidth=0.5)

The Boore and Thompson (2015) corrections interpolate tables of
coefficients in magnitude and distance (:class:`BT15Table`); the tables
also hold their factors for the peak ground acceleration and velocity
(:meth:`OscillatorCorrection.peak_motion_factor`).
:func:`oscillator_correction` builds the correction an
:class:`~pysmsim.params.RVConfig` names.

References
----------
Boore, D. M. and W. B. Joyner (1984). A note on the use of random vibration
theory to predict peak amplitudes of transient signals. Bull. Seismol. Soc.
Am. 74, 2035-2039.

Liu, L. and S. Pezeshk (1999). An improvement on the estimation of
pseudoresponse spectral velocity using RVT method. Bull. Seismol. Soc. Am.
89, 1384-1389.

Boore, D. M. and E. M. Thompson (2015). Revisions to some parameters used
in stochastic-method simulations of ground motion. Bull. Seismol. Soc. Am.
105, 1029-1041.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim.spectrum.response import MotionQuantity

if TYPE_CHECKING:
    from pysmsim.params import RVConfig

__all__ = [
    "OSCILLATOR_CORRECTIONS",
    "BT15Table",
    "BooreJoyner1984",
    "BooreThompson2015",
    "BooreThompson2015Average",
    "LiuPezeshk1999",
    "NoCorrection",
    "OscillatorCorrection",
    "oscillator_correction",
    "oscillator_decay_time",
]

FloatArray = npt.NDArray[np.float64]


def oscillator_decay_time(period: float, damping: float) -> float:
    """Time (s) for an oscillator's motion to decay by 1/e: T/(2*pi*damping)."""
    return period / (2.0 * math.pi * damping)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BT15Table:
    """A table of Boore and Thompson (2015) coefficients (a ``.pars`` file).

    For each magnitude M and point-source distance Rps of a grid, the
    coefficients c1 to c7 of Drms/Dex as a function of T/Dex, and the ratios
    of time-domain to random-vibration PGA and PGV.

    Parameters
    ----------
    name : str
        Name of the table.
    magnitudes : ndarray
        The magnitudes of the grid, increasing (nm).
    log_distances : ndarray
        log10 of the distances (km) of the grid, increasing (nr).
    coefficients : ndarray
        c1 to c7, shape (nm, nr, 7).
    td_rv_pga, td_rv_pgv : ndarray
        TD/RV ratios for PGA and PGV, shape (nm, nr).
    """

    name: str
    magnitudes: FloatArray
    log_distances: FloatArray
    coefficients: FloatArray
    td_rv_pga: FloatArray
    td_rv_pgv: FloatArray

    def __post_init__(self) -> None:
        nm, nr = self.magnitudes.size, self.log_distances.size
        if nm < 2 or nr < 2:
            raise ValueError("a BT15 table needs at least two magnitudes and two distances")
        if np.any(np.diff(self.magnitudes) <= 0) or np.any(np.diff(self.log_distances) <= 0):
            raise ValueError("BT15 table magnitudes and distances must be increasing")
        if self.coefficients.shape != (nm, nr, 7):
            raise ValueError(f"BT15 coefficients must have shape ({nm}, {nr}, 7)")
        if self.td_rv_pga.shape != (nm, nr) or self.td_rv_pgv.shape != (nm, nr):
            raise ValueError(f"BT15 TD/RV ratios must have shape ({nm}, {nr})")

    def _cell(self, magnitude: float, rps: float) -> tuple[int, int, float, float] | None:
        # The grid cell around (M, log10 Rps) and the fractions t, u across
        # it (Fortran locate); None outside the grid, edges included
        logr = math.log10(rps)
        m, lr = self.magnitudes, self.log_distances
        if not (m[0] <= magnitude <= m[-1] and lr[0] <= logr <= lr[-1]):
            return None
        j = min(int(np.searchsorted(m, magnitude, side="right")) - 1, m.size - 2)
        k = min(int(np.searchsorted(lr, logr, side="right")) - 1, lr.size - 2)
        t = (magnitude - m[j]) / (m[j + 1] - m[j])
        u = (logr - lr[k]) / (lr[k + 1] - lr[k])
        return j, k, t, u

    @staticmethod
    def _bilinear(y: FloatArray, j: int, k: int, t: float, u: float) -> float:
        return float(
            (1.0 - t) * (1.0 - u) * y[j, k]
            + t * (1.0 - u) * y[j + 1, k]
            + t * u * y[j + 1, k + 1]
            + (1.0 - t) * u * y[j, k + 1]
        )

    @staticmethod
    def _ratio(period_over_dex: float, damping: float, c: FloatArray) -> float:
        # Drms/Dex for one set of coefficients (Fortran bt12osc_eval_eq)
        eta = period_over_dex
        f = c[0] + c[1] * (1.0 - eta ** c[2]) / (1.0 + eta ** c[2])
        osc = 1.0 + c[3] / (2.0 * math.pi * damping) * (eta / (1.0 + c[4] * eta ** c[5])) ** c[6]
        return float(f * osc)

    def rms_to_excitation(
        self, period_over_dex: float, damping: float, magnitude: float, rps: float
    ) -> float:
        """Drms/Dex for an oscillator (Fortran ``drms_dex_f``).

        log10(Drms/Dex) is computed at the four grid points around (M,
        log10 Rps) and interpolated bilinearly.

        Parameters
        ----------
        period_over_dex : float
            Oscillator period divided by the excitation duration.
        damping : float
            Oscillator damping.
        magnitude, rps : float
            Moment magnitude and point-source distance (km).

        Returns
        -------
        float
            Drms/Dex, or NaN outside the table (or where its c1 and c2 are 0).
        """
        cell = self._cell(magnitude, rps)
        if cell is None:
            return math.nan
        j, k, t, u = cell
        c = self.coefficients
        if c[j, k, 0] == 0.0 and c[j, k, 1] == 0.0:
            return math.nan
        logs = np.empty((2, 2))
        for dj in (0, 1):
            for dk in (0, 1):
                logs[dj, dk] = math.log10(self._ratio(period_over_dex, damping, c[j + dj, k + dk]))
        return float(10.0 ** self._bilinear(logs, 0, 0, t, u))

    def td_rv_ratio(self, quantity: MotionQuantity, magnitude: float, rps: float) -> float:
        """TD/RV ratio for PGA or PGV (Fortran ``bt12_td_rv_for_pga_or_pgv_f``).

        Interpolated bilinearly in M and log10 Rps. There is none for
        displacement, which gets 1.

        Returns
        -------
        float
            The ratio, or NaN outside the table (or where it is 0).
        """
        if quantity is MotionQuantity.DISPLACEMENT:
            return 1.0
        cell = self._cell(magnitude, rps)
        if cell is None:
            return math.nan
        j, k, t, u = cell
        y = self.td_rv_pga if quantity is MotionQuantity.ACCELERATION else self.td_rv_pgv
        if y[j, k] == 0.0:
            return math.nan
        return self._bilinear(y, j, k, t, u)


class OscillatorCorrection(ABC):
    """Base class of the RMS-duration corrections (Fortran ``osc_crrctn``).

    A subclass sets :attr:`name` and implements :meth:`rms_duration`; the
    Boore and Thompson (2015) corrections also override
    :meth:`peak_motion_factor`.
    """

    __slots__ = ()

    name: ClassVar[str]
    """The name that selects the correction (``osc_correction`` in a run file)."""

    @abstractmethod
    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """RMS duration Drms (s) of an oscillator's response.

        Parameters
        ----------
        dur_ex : float
            Excitation duration Dex (s).
        period, damping : float
            The oscillator's natural period (s) and damping.
        magnitude, rps : float
            Moment magnitude and point-source distance (km).
        bandwidth : float
            Bandwidth of the response, sqrt(1 - m1**2/(m0*m2)).

        Returns
        -------
        float
            Drms, or NaN where the correction is not defined.
        """

    def peak_motion_factor(self, quantity: MotionQuantity, magnitude: float, rps: float) -> float:
        """Factor on the peak ground motion (1 except for the BT15 corrections)."""
        return 1.0


@dataclass(frozen=True, slots=True)
class NoCorrection(OscillatorCorrection):
    """No correction: Drms = Dex (Fortran ``osc_crrctn`` 0; Boore, 1983)."""

    name: ClassVar[str] = "none"

    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """Return Dex."""
        return dur_ex


@dataclass(frozen=True, slots=True)
class BooreJoyner1984(OscillatorCorrection):
    """Boore and Joyner (1984) (Fortran ``osc_crrctn`` 1).

    Drms = Dex + To * r/(r + 1/3), with r = (Dex/T)**3 and To the
    oscillator's decay time T/(2*pi*damping).
    """

    name: ClassVar[str] = "bj84"

    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """Return Dex + To*r/(r + 1/3)."""
        r = (dur_ex / period) ** 3
        return dur_ex + oscillator_decay_time(period, damping) * r / (r + 1.0 / 3.0)


@dataclass(frozen=True, slots=True)
class LiuPezeshk1999(OscillatorCorrection):
    """Liu and Pezeshk (1999) (Fortran ``osc_crrctn`` 2).

    Drms = Dex + To * r/(r + alpha), with r = (Dex/T)**2 and
    alpha = q*sqrt(2*pi), q being the bandwidth of the response.
    """

    name: ClassVar[str] = "lp99"

    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """Return Dex + To*r/(r + q*sqrt(2*pi))."""
        r = (dur_ex / period) ** 2
        alpha = bandwidth * math.sqrt(2.0 * math.pi)
        return dur_ex + oscillator_decay_time(period, damping) * r / (r + alpha)


@dataclass(frozen=True, slots=True)
class BooreThompson2015(OscillatorCorrection):
    """Boore and Thompson (2015), one table (Fortran ``osc_crrctn`` 3 or 4).

    Drms = Dex * (Drms/Dex from the table at T/Dex, M and Rps); PGA and PGV
    are multiplied by the table's TD/RV ratios.

    Parameters
    ----------
    table : BT15Table
        The coefficients.
    """

    name: ClassVar[str] = "bt15"

    table: BT15Table

    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """Return Dex * (Drms/Dex); NaN outside the table."""
        return dur_ex * self.table.rms_to_excitation(period / dur_ex, damping, magnitude, rps)

    def peak_motion_factor(self, quantity: MotionQuantity, magnitude: float, rps: float) -> float:
        """Return the table's TD/RV ratio; NaN outside the table."""
        return self.table.td_rv_ratio(quantity, magnitude, rps)


@dataclass(frozen=True, slots=True)
class BooreThompson2015Average(OscillatorCorrection):
    """Average of two Boore and Thompson (2015) tables (Fortran ``osc_crrctn`` 5).

    Drms/Dex and the TD/RV ratios are the averages of those of the two
    tables; NaN if either is outside its table.

    Parameters
    ----------
    first, second : BT15Table
        The two tables (in SMSIM, WNA and ENA).
    """

    name: ClassVar[str] = "bt15_average"

    first: BT15Table
    second: BT15Table

    def rms_duration(
        self,
        dur_ex: float,
        *,
        period: float,
        damping: float,
        magnitude: float,
        rps: float,
        bandwidth: float,
    ) -> float:
        """Return Dex times the average Drms/Dex."""
        ratios = [
            table.rms_to_excitation(period / dur_ex, damping, magnitude, rps)
            for table in (self.first, self.second)
        ]
        return dur_ex * 0.5 * (ratios[0] + ratios[1])

    def peak_motion_factor(self, quantity: MotionQuantity, magnitude: float, rps: float) -> float:
        """Return the average of the two tables' TD/RV ratios."""
        return 0.5 * (
            self.first.td_rv_ratio(quantity, magnitude, rps)
            + self.second.td_rv_ratio(quantity, magnitude, rps)
        )


def oscillator_correction(
    rv: RVConfig, load_table: Callable[[str], BT15Table] | None = None
) -> OscillatorCorrection:
    """Build the correction that random-vibration settings name.

    Parameters
    ----------
    rv : RVConfig
        The settings: ``osc_correction`` and, for the BT15 corrections, the
        names of the two tables (``bt15_tables``; the first is used by
        ``bt15_wna``, the second by ``bt15_ena``).
    load_table : callable, optional
        Returns a table by name; the bundled tables by default
        (:func:`pysmsim.io.bt15_tables.load_bt15_table`).
    """
    from pysmsim.params import OscillatorCorrection as Choice

    if load_table is None:
        from pysmsim.io.bt15_tables import load_bt15_table

        load_table = load_bt15_table
    first, second = rv.bt15_tables
    builders: dict[Choice, Callable[[], OscillatorCorrection]] = {
        Choice.NONE: NoCorrection,
        Choice.BJ84: BooreJoyner1984,
        Choice.LP99: LiuPezeshk1999,
        Choice.BT15_WNA: lambda: BooreThompson2015(load_table(first)),
        Choice.BT15_ENA: lambda: BooreThompson2015(load_table(second)),
        Choice.BT15_AVERAGE: lambda: BooreThompson2015Average(
            load_table(first), load_table(second)
        ),
    }
    return builders[rv.osc_correction]()


OSCILLATOR_CORRECTIONS: dict[str, type[OscillatorCorrection]] = {
    cls.name: cls
    for cls in (
        NoCorrection,
        BooreJoyner1984,
        LiuPezeshk1999,
        BooreThompson2015,
        BooreThompson2015Average,
    )
}
"""The correction classes by name."""
