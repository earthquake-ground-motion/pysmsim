"""White noise for the time-domain simulations.

SMSIM draws its noise with the Numerical Recipes ``gasdev`` and ``ran1``,
which pysmsim does not reproduce; :func:`white_noise` draws it from a
:class:`numpy.random.Generator`. Only the shape of the distribution
matters, because :func:`pysmsim.td.acc_ts.acc_ts` normalises the spectrum
of the windowed noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pysmsim.params import NoiseDistribution

__all__ = ["white_noise"]


def white_noise(
    rng: np.random.Generator, shape: int | tuple[int, ...], distribution: NoiseDistribution
) -> npt.NDArray[np.float64]:
    """Draw white noise (Fortran ``gasdev`` or ``ran1 - 0.5``).

    Parameters
    ----------
    rng : numpy.random.Generator
        The random-number generator.
    shape : int or tuple of int
        Shape of the result; with several rows, row k holds the noise of
        simulation k, drawn in turn from ``rng``.
    distribution : NoiseDistribution
        Standard normal, or uniform on [-0.5, 0.5).

    Returns
    -------
    ndarray
    """
    if distribution is NoiseDistribution.GAUSSIAN:
        return rng.standard_normal(shape)
    return rng.random(shape) - 0.5
