"""One simulated acceleration series (Fortran ``acc_ts``).

The windowed white noise is transformed to the frequency domain, its
spectrum normalised to unit mean-square amplitude and multiplied by the
Fourier acceleration spectrum of the scenario, and transformed back::

    from pysmsim.td.acc_ts import acc_ts, target_spectrum
    spectrum = target_spectrum(grid, params, magnitude=6.0, rrup=20.0)
    acc = acc_ts(noise, grid, w, spectrum)       # cm/s^2

``noise`` may hold many simulations, one per row; they are transformed
together.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pysmsim.params import Params
from pysmsim.spectrum.fas import fourier_spectrum
from pysmsim.spectrum.response import GroundMotion, MotionQuantity
from pysmsim.td.window import TimeGrid

__all__ = ["acc_ts", "fft_frequencies", "target_spectrum"]

FloatArray = npt.NDArray[np.float64]


def fft_frequencies(grid: TimeGrid) -> FloatArray:
    """Frequencies of the real FFT of the series: 0 to the Nyquist frequency (Hz)."""
    return np.fft.rfftfreq(grid.npts, grid.dt)


def target_spectrum(grid: TimeGrid, params: Params, magnitude: float, rrup: float) -> FloatArray:
    """Fourier acceleration spectrum (cm/s) on the FFT frequencies of the series.

    The value at 0 Hz is 0: SMSIM sets the mean of the acceleration to 0.
    """
    f = fft_frequencies(grid)
    spectrum = np.zeros_like(f)
    spectrum[1:] = fourier_spectrum(
        f[1:], params, magnitude, rrup, GroundMotion(MotionQuantity.ACCELERATION)
    )
    return spectrum


def acc_ts(
    noise: npt.ArrayLike, grid: TimeGrid, window: FloatArray, spectrum: FloatArray
) -> FloatArray:
    """Shape white noise into acceleration series (Fortran ``acc_ts``).

    Parameters
    ----------
    noise : array_like
        White noise for the points ``grid.nstart`` to ``grid.nstop``: shape
        ``(nmotion,)`` for one series or ``(nsims, nmotion)`` for several.
    grid : TimeGrid
        The time grid of the scenario.
    window : ndarray
        The window on the whole series (:func:`pysmsim.td.window.window`).
    spectrum : ndarray
        The Fourier acceleration spectrum on the FFT frequencies
        (:func:`target_spectrum`).

    Returns
    -------
    ndarray
        Acceleration (cm/s^2), shape ``(npts,)`` or ``(nsims, npts)``.

    Notes
    -----
    SMSIM normalises by the mean squared amplitude of the noise spectrum
    without the 0 Hz and Nyquist values, and does not remove the mean of
    the noise.
    """
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape[-1] != grid.nmotion:
        raise ValueError(f"noise must have {grid.nmotion} points, got {noise.shape[-1]}")
    n = grid.npts
    work = np.zeros((*noise.shape[:-1], n))
    work[..., grid.nstart : grid.nstop + 1] = window[grid.nstart : grid.nstop + 1] * noise
    x = np.fft.rfft(work, axis=-1)
    power = x.real[..., 1 : n // 2] ** 2 + x.imag[..., 1 : n // 2] ** 2
    rms = np.sqrt(power.sum(axis=-1) / (n // 2 - 1))
    x *= spectrum / rms[..., np.newaxis]
    # SMSIM's inverse realft and its factor 2*df amount to irfft / dt
    return np.fft.irfft(x, n, axis=-1) / grid.dt
