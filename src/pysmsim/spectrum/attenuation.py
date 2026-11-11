"""Anelastic attenuation: the quality factor Q(f) and the path factor.

The anelastic path factor is exp(-pi*f*R/(c_q*Q(f))), the Q part of
SMSIM's ``dimin``. Each Q model is a class derived from :class:`QModel`
that holds only its own parameters and chooses its distance R::

    from pysmsim.spectrum.attenuation import PiecewiseQ
    q = PiecewiseQ(qr1=180.0, s1=0.45, qr2=180.0, s2=0.45)
    q.q(frequencies, magnitude=6.0)                     # Q(f)
    q.attenuation(frequencies, 6.0, rps=50.0, rrup=50.0)  # path factor

Q(f) is computed by a compiled kernel (:class:`pysmsim._jit.Kernel`), with
its plain NumPy twin kept for the tests. Q is not defined at f = 0; the
kernels return infinity there, so the path factor is 1.

The models are picked by name in a TOML run file (``model =
"piecewise"``); :data:`Q_MODELS` maps each name to its class.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysmsim._jit import Kernel, kernel

__all__ = ["Q_MODELS", "PiecewiseQ", "QModel", "SimpleQ", "Stafford2022Q"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class QModel(ABC):
    """Base class of the Q models.

    A subclass sets :attr:`name` and :attr:`kernel` and implements
    :meth:`kernel_args` and :meth:`distance`.

    Parameters
    ----------
    c_q : float
        Velocity (km/s) in the path factor (Fortran ``c_q``).
    """

    name: ClassVar[str]
    """The name that selects the model in a run file."""
    kernel: ClassVar[Kernel]
    """The kernel that computes Q(f); it takes the frequencies and then
    :meth:`kernel_args`."""

    c_q: float = 3.5

    def __post_init__(self) -> None:
        if self.c_q <= 0:
            raise ValueError(f"c_q must be positive, got {self.c_q}")

    @abstractmethod
    def kernel_args(self, magnitude: float) -> tuple[float, ...]:
        """Return the numbers that :attr:`kernel` takes after the frequencies."""

    @abstractmethod
    def distance(self, rps: float, rrup: float) -> float:
        """Return the distance (km) that the path factor uses."""

    def q(self, f: npt.ArrayLike, magnitude: float) -> FloatArray:
        """Quality factor Q(f) (Fortran ``q``).

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.
        magnitude : float
            Moment magnitude (used by :class:`Stafford2022Q`).

        Returns
        -------
        ndarray
            Q, with the shape of ``f``; infinity at f = 0.
        """
        freqs = np.asarray(f, dtype=np.float64)
        flat = np.ascontiguousarray(freqs.ravel())
        return self.kernel.compiled(flat, *self.kernel_args(magnitude)).reshape(freqs.shape)

    def attenuation(
        self, f: npt.ArrayLike, magnitude: float, rps: float, rrup: float
    ) -> FloatArray:
        """Anelastic path factor exp(-pi*f*R/(c_q*Q(f))).

        Parameters
        ----------
        f : array_like
            Frequencies (Hz), any shape.
        magnitude : float
            Moment magnitude.
        rps, rrup : float
            Point-source and rupture distances (km); :meth:`distance`
            chooses which one is used.

        Returns
        -------
        ndarray
            The factor, with the shape of ``f``.
        """
        freqs = np.asarray(f, dtype=np.float64)
        r = self.distance(rps, rrup)
        return np.exp(-math.pi * freqs * r / (self.c_q * self.q(freqs, magnitude)))


def _simple_q(
    f: FloatArray,
    q0: float,
    eta: float,
) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        fi = f[i]
        if fi == 0.0:
            out[i] = np.inf
        else:
            out[i] = q0 * fi**eta
    return out


def _piecewise_q(
    f: FloatArray,
    fr1: float,
    qr1: float,
    s1: float,
    ft1: float,
    fr2: float,
    qr2: float,
    s2: float,
    ft2: float,
    qt1: float,
    st: float,
) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        fi = f[i]
        if fi == 0.0:
            out[i] = np.inf
        elif fi <= ft1:
            out[i] = qr1 * (fi / fr1) ** s1
        elif fi >= ft2:
            out[i] = qr2 * (fi / fr2) ** s2
        else:
            out[i] = qt1 * (fi / ft1) ** st
    return out


def _stafford_2022_q(f: FloatArray, q0: float, eta: float) -> FloatArray:
    out = np.empty_like(f)
    for i in range(f.size):
        out[i] = np.inf if f[i] == 0.0 else q0 * f[i] ** eta
    return out


_PIECEWISE_Q_KERNEL = kernel(_piecewise_q)
_STAFFORD_2022_Q_KERNEL = kernel(_stafford_2022_q)
_SIMPLE_Q_KERNEL = kernel(_simple_q)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleQ(QModel):
    """Simple power law: Q = q0 * f**eta.

    The path factor uses Rps. This is :class:`PiecewiseQ` with
    qr1 = qr2 = q0, s1 = s2 = eta and fr1 = fr2 = 1, given by its two
    parameters (SMSIM has no separate form for it).

    Parameters
    ----------
    q0, eta : float
        Q at 1 Hz and the exponent of the power law.
    c_q : float
        Velocity (km/s) in the path factor.
    """

    name: ClassVar[str] = "simple"
    kernel: ClassVar[Kernel] = _SIMPLE_Q_KERNEL

    q0: float
    eta: float

    def __post_init__(self) -> None:
        QModel.__post_init__(self)
        if self.q0 <= 0:
            raise ValueError(f"q0 must be positive, got {self.q0}")

    def kernel_args(self, magnitude: float) -> tuple[float, ...]:
        """(q0, eta); Q does not depend on M."""
        return (self.q0, self.eta)

    def distance(self, rps: float, rrup: float) -> float:
        """Return the distance (km) that the path factor uses: Rps."""
        return rps


@dataclass(frozen=True, slots=True, kw_only=True)
class PiecewiseQ(QModel):
    """Power laws below ft1 and above ft2, joined by a power law between.

    Q = qr1*(f/fr1)**s1 for f <= ft1, qr2*(f/fr2)**s2 for f >= ft2, and a
    power law through the values at ft1 and ft2 between them (Fortran ``q``
    with fr1 > 0). With ft1 = ft2 there is no middle part. The path factor
    uses Rps.

    Parameters
    ----------
    qr1, s1, fr1 : float
        Low-frequency power law.
    qr2, s2, fr2 : float
        High-frequency power law.
    ft1, ft2 : float
        Transition frequencies (Hz), ft1 <= ft2.
    c_q : float
        Velocity (km/s) in the path factor.
    """

    name: ClassVar[str] = "piecewise"
    kernel: ClassVar[Kernel] = _PIECEWISE_Q_KERNEL

    qr1: float
    s1: float = 0.0
    fr1: float = 1.0
    qr2: float
    s2: float = 0.0
    fr2: float = 1.0
    ft1: float = 1.0
    ft2: float = 1.0

    def __post_init__(self) -> None:
        QModel.__post_init__(self)
        if min(self.qr1, self.qr2, self.fr1, self.fr2, self.ft1) <= 0:
            raise ValueError("piecewise Q needs positive qr1, qr2, fr1, fr2 and ft1")
        if self.ft2 < self.ft1:
            raise ValueError(f"piecewise Q needs ft1 <= ft2, got {self.ft1} and {self.ft2}")

    def kernel_args(self, magnitude: float) -> tuple[float, ...]:
        """(fr1, qr1, s1, ft1, fr2, qr2, s2, ft2, qt1, st); Q does not depend on M."""
        qt1 = self.qr1 * (self.ft1 / self.fr1) ** self.s1
        qt2 = self.qr2 * (self.ft2 / self.fr2) ** self.s2
        st = 0.0
        if self.ft1 != self.ft2:
            st = math.log10(qt2 / qt1) / math.log10(self.ft2 / self.ft1)
        return (
            self.fr1,
            self.qr1,
            self.s1,
            self.ft1,
            self.fr2,
            self.qr2,
            self.s2,
            self.ft2,
            qt1,
            st,
        )

    def distance(self, rps: float, rrup: float) -> float:
        """Return the distance (km) that the path factor uses: Rps."""
        return rps


@dataclass(frozen=True, slots=True, kw_only=True)
class Stafford2022Q(QModel):
    """Stafford et al. (2022): Q = q0 * f**eta(M), eta = eta_a + eta_b*tanh(M - eta_g).

    Fortran ``q`` with fr1 < 0, where qr1, s1, ft1 and ft2 hold q0, eta_a,
    eta_b and eta_g. The path factor uses Rrup.

    Parameters
    ----------
    q0 : float
        Q at 1 Hz.
    eta_a, eta_b, eta_g : float
        Coefficients of the exponent.
    c_q : float
        Velocity (km/s) in the path factor.

    Notes
    -----
    SMSIM uses Rrup only when fr1 is exactly -1, and Rps for other negative
    values, though Q itself follows this model for any fr1 < 0. pysmsim
    always uses Rrup.
    """

    name: ClassVar[str] = "stafford_2022"
    kernel: ClassVar[Kernel] = _STAFFORD_2022_Q_KERNEL

    q0: float
    eta_a: float
    eta_b: float
    eta_g: float

    def __post_init__(self) -> None:
        QModel.__post_init__(self)
        if self.q0 <= 0:
            raise ValueError(f"q0 must be positive, got {self.q0}")

    def kernel_args(self, magnitude: float) -> tuple[float, ...]:
        """(q0, eta(M))."""
        return (self.q0, self.eta_a + self.eta_b * math.tanh(magnitude - self.eta_g))

    def distance(self, rps: float, rrup: float) -> float:
        """Return the distance (km) that the path factor uses: Rrup."""
        return rrup


Q_MODELS: dict[str, type[QModel]] = {cls.name: cls for cls in (PiecewiseQ, Stafford2022Q, SimpleQ)}
"""The Q classes by name."""
