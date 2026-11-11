"""Ground-motion and instrument responses applied to the spectrum.

SMSIM's ``spect_amp`` gives the Fourier spectrum of ground displacement
(cm-s) and multiplies it by a response chosen with ``iaorins`` and ``idva``:
ground displacement, velocity or acceleration, a damped single-degree-of-
freedom oscillator, or a Wood-Anderson or WWSSN short-period seismometer.
Each response is a class derived from :class:`Response`; its
:meth:`~Response.factor` is the factor that multiplies the displacement
spectrum::

    from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator
    GroundMotion(MotionQuantity.ACCELERATION).factor(frequencies)  # (2*pi*f)**2
    Oscillator(period=0.2, damping=0.05).factor(frequencies)

The oscillator is computed by a compiled kernel
(:class:`pysmsim._jit.Kernel`), with its plain NumPy twin kept for the
tests.

References
----------
Uhrhammer, R. A. and E. R. Collins (1990). Synthesis of Wood-Anderson
seismograms from broadband digital records. Bull. Seismol. Soc. Am. 80,
702-716 (the Wood-Anderson gain and natural period used in SMSIM).

The WWSSN short-period poles and zeros are those in SMSIM's ``spect_amp``,
from notes by C. R. Hutt (2006) that are not in the SMSIM distribution.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import Kernel, kernel

__all__ = [
    "GroundMotion",
    "MotionQuantity",
    "Oscillator",
    "PolesZerosInstrument",
    "Response",
    "WWSSNShortPeriod",
    "WoodAnderson",
]

FloatArray = npt.NDArray[np.float64]


class Response(ABC):
    """Base class of the responses.

    A subclass implements :meth:`factor`.
    """

    __slots__ = ()

    @abstractmethod
    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Factor that multiplies the Fourier displacement spectrum.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.

        Returns
        -------
        ndarray
            The factor, with the shape of ``f``.
        """


class MotionQuantity(StrEnum):
    """Ground-motion quantity (Fortran ``idva`` with ``iaorins`` 1)."""

    DISPLACEMENT = "displacement"
    """0: displacement (spectrum in cm-s)."""
    VELOCITY = "velocity"
    """1: velocity (cm)."""
    ACCELERATION = "acceleration"
    """2: acceleration (cm/s)."""


_DERIVATIVE_ORDER = {
    MotionQuantity.DISPLACEMENT: 0,
    MotionQuantity.VELOCITY: 1,
    MotionQuantity.ACCELERATION: 2,
}


@dataclass(frozen=True, slots=True)
class GroundMotion(Response):
    """Ground displacement, velocity or acceleration (Fortran ``iaorins`` 1).

    The factor is (2*pi*f)**n, with n = 0, 1 or 2 (``idva``).
    """

    quantity: MotionQuantity = MotionQuantity.DISPLACEMENT

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return (2*pi*f)**n on frequencies (Hz) of any shape."""
        freqs = np.asarray(f, dtype=np.float64)
        n = _DERIVATIVE_ORDER[self.quantity]
        if n == 0:
            return np.ones(freqs.shape)
        return np.asarray((2.0 * math.pi * freqs) ** n, dtype=np.float64)


def _oscillator(f: FloatArray, fosc: float, damping: float) -> FloatArray:
    # 2*pi*fosc * harmoscf(f, fosc, damping, idva = 0)
    out = np.empty_like(f)
    for i in range(f.size):
        fi = f[i]
        out[i] = (
            2.0
            * np.pi
            * fosc
            * fi
            * fi
            / np.sqrt((fosc * fosc - fi * fi) ** 2 + (2.0 * fi * fosc * damping) ** 2)
        )
    return out


_OSCILLATOR_KERNEL = kernel(_oscillator)


@dataclass(frozen=True, slots=True, kw_only=True)
class Oscillator(Response):
    """Pseudo-velocity response of a damped oscillator (Fortran ``iaorins`` 2).

    2*pi*fosc * f**2 / sqrt((fosc**2 - f**2)**2 + (2*f*fosc*damping)**2),
    the displacement response of the oscillator to ground displacement
    (Fortran ``harmoscf`` with ``idva`` 0) times 2*pi*fosc. Applied to the
    displacement spectrum it gives the spectrum of the oscillator's
    pseudo-velocity (cm); its peak is PSV, and PSA = 2*pi*fosc*PSV.

    Parameters
    ----------
    period : float
        Natural period (s), fosc = 1/period.
    damping : float
        Damping, as a fraction of critical.
    """

    kernel: ClassVar[Kernel] = _OSCILLATOR_KERNEL
    """The kernel that computes the response; it takes fosc and the damping."""

    period: float
    damping: float = 0.05

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if not 0 < self.damping < 1:
            raise ValueError(f"damping must be between 0 and 1, got {self.damping}")

    @property
    def frequency(self) -> float:
        """Natural frequency fosc (Hz)."""
        return 1.0 / self.period

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the oscillator response on frequencies (Hz) of any shape."""
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        return self.kernel.compiled(flat, self.frequency, self.damping).reshape(freqs.shape)


class PolesZerosInstrument(Response):
    """Base class of the seismometers given by poles and zeros.

    The response to ground displacement is
    gain * |s**nzeros / prod(s - p)| / norm, with s = 2*pi*i*f, where norm
    is the same expression without the gain at ``gain_frequency``, so that
    the response there equals the gain (Fortran ``poles_zeros_response``).
    A subclass sets :attr:`poles`, :attr:`n_zeros`, :attr:`gain` and
    :attr:`gain_frequency`.
    """

    __slots__ = ()

    poles: ClassVar[tuple[complex, ...]]
    """Poles (rad/s)."""
    n_zeros: ClassVar[int]
    """Number of zeros, all at s = 0."""
    gain: ClassVar[float]
    """Gain at :attr:`gain_frequency`."""
    gain_frequency: ClassVar[float]
    """Frequency (Hz) where the response equals the gain."""

    @classmethod
    def _unscaled(cls, f: FloatArray) -> FloatArray:
        s = 2j * math.pi * f
        work = s**cls.n_zeros
        for p in cls.poles:
            work = work / (s - p)
        return np.asarray(np.abs(work), dtype=np.float64)

    def factor(self, f: npt.ArrayLike) -> FloatArray:
        """Return the instrument response on frequencies (Hz) of any shape."""
        freqs = np.asarray(f, dtype=np.float64)
        norm = float(self._unscaled(np.array([self.gain_frequency]))[0])
        return np.asarray((self.gain / norm) * self._unscaled(freqs), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class WoodAnderson(PolesZerosInstrument):
    """Wood-Anderson seismometer (Fortran ``iaorins`` 3).

    Gain 2080 (Uhrhammer and Collins, 1990), applied at 100 Hz, where the
    displacement response is flat; natural frequency 1.25 Hz and damping
    0.69, as poles -5.41925 +/- 5.68479i.
    """

    poles: ClassVar[tuple[complex, ...]] = (complex(-5.41925, -5.68479), complex(-5.41925, 5.68479))
    n_zeros: ClassVar[int] = 2
    gain: ClassVar[float] = 2080.0
    gain_frequency: ClassVar[float] = 100.0


@dataclass(frozen=True, slots=True)
class WWSSNShortPeriod(PolesZerosInstrument):
    """WWSSN short-period seismometer (Fortran ``iaorins`` 4).

    Three zeros at 0 and five poles, with gain 1 at 1 Hz.
    """

    poles: ClassVar[tuple[complex, ...]] = (
        complex(-3.725, -6.22),
        complex(-3.725, 6.22),
        complex(-5.612, 0.0),
        complex(-13.24, 0.0),
        complex(-21.08, 0.0),
    )
    n_zeros: ClassVar[int] = 3
    gain: ClassVar[float] = 1.0
    gain_frequency: ClassVar[float] = 1.0
