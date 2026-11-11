"""Fourier spectra for a grid of scenarios (Fortran ``fmrsk_loop_fas_drvr``).

:func:`fas_loop` computes the Fourier amplitude spectrum for every
combination of kappa_0, stress, magnitude and distance in a
:class:`~pysmsim.scenario.ScenarioGrid` and returns a table with one row per
scenario, in the order of SMSIM's loops (kappa_0, stress, magnitude,
distance), and the acceleration spectrum at each frequency as a column
``fas(f)``::

    import numpy as np
    from pysmsim.io.config import load_config
    from pysmsim.drivers.loop_fas import fas_loop
    config = load_config("run.toml")
    table = fas_loop(config.params, config.scenarios, np.logspace(-2, 2, 21))

:func:`fas_loop_details` gives instead one row per scenario and frequency,
with the displacement spectrum, spreading and Q as well.

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

__all__ = [
    "FAS_DETAIL_COLUMNS",
    "FAS_SCENARIO_COLUMNS",
    "fas_column",
    "fas_loop",
    "fas_loop_details",
    "params_for",
]

FAS_DETAIL_COLUMNS = (
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
    "gsprd",
    "q",
    "gsprd_q",
)
"""Columns of the table from :func:`fas_loop_details`.

f (Hz), period (s; inf at f = 0), magnitude, rjb (km; NaN when the grid
gives Rrup), h (km, the depth of the Rjb-to-Rrup conversion; NaN when the
grid gives Rrup), rrup, f_ff (finite-fault factor, km), rps (km), kappa0
(s), stress (bars; NaN for source models without one), fds (displacement
spectrum, cm-s), fas (acceleration spectrum, cm/s),
gsprd (geometrical spreading), q (Q(f)) and gsprd_q (spreading times the
anelastic path factor).
"""

FAS_SCENARIO_COLUMNS = ("magnitude", "rjb", "h", "rrup", "f_ff", "rps", "kappa0", "stress")
"""The columns of :func:`fas_loop` that describe each scenario (see
:data:`FAS_DETAIL_COLUMNS`); the spectrum columns follow them."""


def fas_column(frequency: float) -> str:
    """Name of the FAS column for a frequency (Hz): ``fas(f)``, f with up to 6 digits."""
    return f"fas({frequency:g})"


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
    """Fourier acceleration spectra, one row per scenario of a grid.

    Parameters and options as for :func:`fas_loop_details`.

    Returns
    -------
    pandas.DataFrame
        The columns :data:`FAS_SCENARIO_COLUMNS`, then ``fas(f)``, the
        acceleration spectrum (cm/s), for each frequency f in the order
        given. The rows are in the order kappa_0, stress, magnitude,
        distance.

    Raises
    ------
    ValueError
        If there are no frequencies, or two give the same column name.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    names = [fas_column(x) for x in f]
    if len(set(names)) != len(names):
        raise ValueError("frequencies must be distinct to 6 significant digits")
    details = fas_loop_details(params, grid, f, fortran_compat=fortran_compat)
    # Each scenario is a block of rows, one per frequency
    first = details.iloc[:: f.size]
    table = first[list(FAS_SCENARIO_COLUMNS)].reset_index(drop=True)
    spectra = pd.DataFrame(details.fas.to_numpy().reshape(len(table), f.size), columns=names)
    return pd.DataFrame(pd.concat([table, spectra], axis=1))


def fas_loop_details(
    params: Params,
    grid: ScenarioGrid,
    frequencies: npt.ArrayLike,
    *,
    fortran_compat: bool = False,
) -> pd.DataFrame:
    """Fourier spectra and their factors for every scenario of a grid.

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
        :data:`FAS_DETAIL_COLUMNS`.

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
                            "gsprd": factors.spreading,
                            "q": run.q.q(f, magnitude),
                            "gsprd_q": factors.spreading * factors.q_path,
                        }
                    )
    return pd.DataFrame(
        {name: np.concatenate([b[name] for b in blocks]) for name in FAS_DETAIL_COLUMNS}
    )
