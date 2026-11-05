"""Numba compilation for the numerical kernels.

Kernels are written once, as plain NumPy functions that Numba can also
compile. :func:`jit` compiles such a function with ``numba.njit`` (cached to
disk); the plain function stays available as the NumPy twin, which the tests
compare against the compiled one.

Setting the environment variable ``NUMBA_DISABLE_JIT=1`` makes Numba return
the plain functions, which is useful for debugging and coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt

__all__ = ["Kernel", "jit", "kernel"]


def jit[F: Callable[..., Any]](func: F) -> F:
    """Compile a NumPy kernel with ``numba.njit(cache=True)``."""
    return cast(F, numba.njit(cache=True)(func))


@dataclass(frozen=True, slots=True)
class Kernel:
    """A numerical kernel, compiled and as its NumPy twin.

    Both take the same arguments: a one-dimensional array of frequencies
    followed by plain numbers, which the model that owns the kernel
    supplies.
    """

    numpy: Callable[..., npt.NDArray[np.float64]]
    """The plain NumPy function."""
    compiled: Callable[..., npt.NDArray[np.float64]]
    """The same function compiled by Numba."""


def kernel(func: Callable[..., npt.NDArray[np.float64]]) -> Kernel:
    """Make a :class:`Kernel` from a plain NumPy function."""
    return Kernel(numpy=func, compiled=jit(func))
