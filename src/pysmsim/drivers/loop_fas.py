"""Fourier spectra for a grid of scenarios (Fortran ``fmrsk_loop_fas_drvr``).

:func:`fas_loop` computes the Fourier amplitude spectrum for every
combination of kappa_0, stress, magnitude and distance in a
:class:`~pysmsim.scenario.ScenarioGrid` and returns a table with one row per
frequency, in the order of SMSIM's loops (kappa_0, stress, magnitude,
distance, frequency)::

    import numpy as np
    from pysmsim.io.config import load_config
    from pysmsim.drivers.loop_fas import fas_loop
    config = load_config("run.toml")
    table = fas_loop(config.params, config.scenarios, np.logspace(-2, 2, 21))

The driver only composes the functions of :mod:`pysmsim.spectrum`.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from pysmsim.params import Params
from pysmsim.scenario import DistanceMetric, ScenarioGrid
from pysmsim.spectrum.distance import rjb_to_rrup_depth
from pysmsim.spectrum.fas import spectrum_factors
from pysmsim.spectrum.stress import LogLinearStress

__all__ = ["FAS_COLUMNS", "G", "fas_loop", "params_for"]

G = 981.0
"""Acceleration of gravity (cm/s^2), as in SMSIM."""

FAS_COLUMNS = (
    "f",
    "period",
    "magnitude",
    "rjb",
    "h",
    "rrup",
    "f_ff",
    "rps",
    "kappa0",
    "stress",
    "fds",
    "fas",
    "fas_g",
    "gsprd",
    "q",
    "gsprd_q",
)
"""Columns of the table from :func:`fas_loop`.

f (Hz), period (s; inf at f = 0), magnitude, rjb (km; NaN when the grid
gives Rrup), h (km, the depth of the Rjb-to-Rrup conversion; NaN when the
grid gives Rrup), rrup, f_ff (finite-fault factor, km), rps (km), kappa0
(s), stress (bars; NaN for source models without one), fds (displacement
spectrum, cm-s), fas (acceleration spectrum, cm/s), fas_g (fas/g, g-s),
gsprd (geometrical spreading), q (Q(f)) and gsprd_q (spreading times the
anelastic path factor).
"""


def params_for(
    params: Params,
    *,
    kappa0: float | None = None,
    stress: float | None = None,
    fortran_compat: bool = False,
) -> Params:
    """Return the parameters with a grid's kappa_0 or stress put in.

    Parameters
    ----------
    params : Params
        Model parameters.
    kappa0 : float, optional
        Kappa_0 (s) to use instead of the one in ``params``. Kappa then does
        not depend on magnitude (``dkappa_dmag`` is set to 0), as in SMSIM.
    stress : float, optional
        Stress parameter (bars) to use instead of the source's stress model,
        which becomes a constant :class:`~pysmsim.spectrum.stress.LogLinearStress`.
    fortran_compat : bool
        Set ``dkappa_dmag`` to 0 even without ``kappa0``, as SMSIM's loop
        drivers do whatever the control file says.

    Raises
    ------
    ValueError
        If ``stress`` is given for a source model without a stress parameter.
    """
    site = params.site
    if kappa0 is not None:
        site = replace(site, kappa0=kappa0, dkappa_dmag=0.0, kappa_magnitude_ref=0.0)
    elif fortran_compat:
        site = replace(site, dkappa_dmag=0.0, kappa_magnitude_ref=0.0)
    source = params.source
    if stress is not None:
        if "stress" not in {f.name for f in fields(source)}:  # type: ignore[arg-type]
            raise ValueError(f"source model {source.name!r} has no stress parameter to replace")
        source = replace(source, stress=LogLinearStress(stress_ref=stress))  # type: ignore[type-var]
    return replace(params, site=site, source=source)


def fas_loop(
    params: Params,
    grid: ScenarioGrid,
    frequencies: npt.ArrayLike,
    *,
    fortran_compat: bool = False,
) -> pd.DataFrame:
    """Fourier spectra for every scenario of a grid.

    Parameters
    ----------
    params : Params
        Model parameters.
    grid : ScenarioGrid
        Magnitudes and distances, the Rjb-to-Rrup conversion and optional
        kappa_0 and stress values; the periods and peak motions are not
        used.
    frequencies : array_like
        Frequencies (Hz), one-dimensional.
    fortran_compat : bool
        Make kappa independent of magnitude even when the grid does not
        give kappa_0, as SMSIM does (see :func:`params_for`).

    Returns
    -------
    pandas.DataFrame
        One row per scenario and frequency, with the columns
        :data:`FAS_COLUMNS`.

    Notes
    -----
    SMSIM's FAS driver always uses the strike-slip CY14 Ztor; this one uses
    ``grid.mechanism``.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or np.any(f < 0):
        raise ValueError("frequencies must be a one-dimensional array of values >= 0")
    period = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    acc = (2.0 * math.pi * f) ** 2
    blocks: list[dict[str, npt.NDArray[np.float64]]] = []
    for kappa0 in grid.kappa0 or (None,):
        for stress in grid.stress or (None,):
            run = params_for(params, kappa0=kappa0, stress=stress, fortran_compat=fortran_compat)
            for magnitude in grid.magnitudes:
                for distance in grid.distances:
                    if grid.distance_metric is DistanceMetric.RJB:
                        rjb = distance
                        h = rjb_to_rrup_depth(
                            magnitude,
                            conversion=grid.rjb_to_rrup,
                            mechanism=grid.mechanism,
                            depth_function=grid.depth_function,
                        )
                        rrup = math.sqrt(rjb**2 + h**2)
                    else:
                        rjb = h = math.nan
                        rrup = distance
                    factors = spectrum_factors(f, run, magnitude, rrup)
                    terms = factors.terms
                    fds = factors.product()
                    n = f.size
                    blocks.append(
                        {
                            "f": f,
                            "period": period,
                            "magnitude": np.full(n, magnitude),
                            "rjb": np.full(n, rjb),
                            "h": np.full(n, h),
                            "rrup": np.full(n, rrup),
                            "f_ff": np.full(n, terms.finite_fault_h),
                            "rps": np.full(n, terms.rps),
                            "kappa0": np.full(n, run.site.kappa0),
                            "stress": np.full(
                                n, math.nan if terms.source.stress is None else terms.source.stress
                            ),
                            "fds": fds,
                            "fas": acc * fds,
                            "fas_g": acc * fds / G,
                            "gsprd": factors.spreading,
                            "q": run.q.q(f, magnitude),
                            "gsprd_q": factors.spreading * factors.q_path,
                        }
                    )
    return pd.DataFrame({name: np.concatenate([b[name] for b in blocks]) for name in FAS_COLUMNS})
