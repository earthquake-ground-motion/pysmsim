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
from typing import Any, cast

import numba

__all__ = ["jit"]


def jit[F: Callable[..., Any]](func: F) -> F:
    """Compile a NumPy kernel with ``numba.njit(cache=True)``."""
    return cast(F, numba.njit(cache=True)(func))
