"""Everything about one scenario (Fortran ``gm_rv_drvr``, ``gm_td_drvr``, ``FAS_DRVR``).

The loop drivers give a few measures for many scenarios; these give every
intermediate quantity for one magnitude and rupture distance::

    import numpy as np
    from pysmsim.io.config import load_config
    from pysmsim.drivers.single import fas_scenario, rv_scenario, td_scenario
    params = load_config("run.toml").params
    periods = (0.1, 1.0)
    rv = rv_scenario(params, 6.0, 20.0, periods)
    rv.pga.peak, rv.arias, rv.table()
    td = td_scenario(params, 6.0, 20.0, periods, keep_series=True)
    td.table(), td.series(0)
    fas = fas_scenario(params, 6.0, 20.0, np.logspace(-2, 2, 41), periods)

- :func:`rv_scenario` (``gm_rv_drvr``): PGD, PGV, PGA, the Arias intensity
  and the response spectrum by random vibration, with the moments,
  durations and peak factors of each.
- :func:`td_scenario` (``gm_td_drvr``): the same measures, and the
  significant durations, from time-domain simulations, as averages over the
  simulations; and the acceleration, velocity and displacement of any one
  simulation.
- :func:`fas_scenario` (``FAS_DRVR``): the Fourier spectra of ground
  displacement, velocity and acceleration, each factor of the spectrum, and
  the Fourier spectra of oscillator responses.

Like SMSIM's drivers, these take the rupture distance Rrup;
:func:`pysmsim.spectrum.distance.rjb_to_rrup_depth` gives the depth that
converts Rjb to Rrup. Unlike the loop drivers, they do not make kappa
independent of magnitude.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pysmsim.params import Params
from pysmsim.rv.duration import oscillator_correction
from pysmsim.rv.gm_rv import RVResult, rv_motion
from pysmsim.spectrum.fas import ScenarioTerms, spectrum_factors
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator
from pysmsim.td.gm_td import TDResult, statistics, td_motion
from pysmsim.td.intensity import G, velocity_displacement

__all__ = [
    "FAS_SCENARIO_COLUMNS",
    "RV_SCENARIO_COLUMNS",
    "SERIES_COLUMNS",
    "TD_SCENARIO_COLUMNS",
    "RVScenario",
    "TDScenario",
    "fas_scenario",
    "osc_columns",
    "rv_scenario",
    "td_scenario",
]

FloatArray = npt.NDArray[np.float64]

_TERM_COLUMNS = ("magnitude", "rrup", "f_ff", "rps", "kappa", "stress", "fa", "fb")

RV_SCENARIO_COLUMNS = (
    "measure",
    "period",
    "damping",
    *_TERM_COLUMNS,
    "fup",
    "dur_source",
    "dur_path",
    "dur_ex",
    "dur_rms",
    "m0",
    "m1",
    "m2",
    "m4",
    "dominant_frequency",
    "bandwidth",
    "epsilon",
    "n_zero_crossings",
    "n_extrema",
    "rms",
    "peak_factor",
    "peak_factor_cl",
    "peak_factor_dk80_asymptotic",
    "peak_motion_factor",
    "value",
    "sd",
    "psv",
    "psa_pga",
)
"""Columns of :meth:`RVScenario.table`, one row per measure.

- measure: ``"pgd"``, ``"pgv"``, ``"pga"``, ``"arias"`` or ``"psa"`` (an
  oscillator); period (s) and damping of the oscillators (NaN otherwise);
- magnitude, rrup (km), f_ff (finite-fault factor, km), rps (km), kappa (s,
  at the magnitude), stress (bars; NaN for source models without one), fa
  and fb (corner frequencies, Hz);
- fup (upper frequency of the moments, Hz), dur_source, dur_path, dur_ex
  and dur_rms (s); the moments m0, m1, m2 and m4 (of the pseudo-velocity
  for an oscillator), dominant_frequency (sqrt(m2/m0)/(2*pi), Hz),
  bandwidth (sqrt(1 - m1**2/(m0*m2))), epsilon (sqrt(1 - m2**2/(m0*m4))),
  n_zero_crossings and n_extrema (over dur_ex);
- rms (the RMS motion), peak_factor (DK80), peak_factor_cl (Cartwright and
  Longuet-Higgins eq. 6.8) and peak_factor_dk80_asymptotic, for
  comparison, and peak_motion_factor (the BT15 TD/RV ratio for PGV and
  PGA, else 1);
- value: the measure in cgs units, PGD (cm), PGV (cm/s), PGA (cm/s^2), the
  Arias intensity (cm/s) or PSA (cm/s^2); for an oscillator also sd (cm),
  psv (cm/s) and psa_pga, PSA over PGA.

The Arias intensity row has only the scenario and value columns: it is
pi/(2g) times the moment m0 of the acceleration spectrum, the integral of
acc**2 over time (g = 981 cm/s^2).
"""

TD_SCENARIO_COLUMNS = (
    "measure",
    "period",
    "damping",
    *_TERM_COLUMNS,
    "dur_source",
    "dur_path",
    "dt",
    "npts",
    "te",
    "nsims",
    "mean",
    "std",
    "sem",
    "gmean",
    "gfactor",
    "gsem",
)
"""Columns of :meth:`TDScenario.table`, one row per measure.

- measure: ``"pgd"``, ``"pgv"``, ``"pga"``, ``"arias"``, ``"dur_75_05"``,
  ``"dur_95_05"`` or ``"psa"`` (an oscillator); period (s) and damping of
  the oscillators (NaN otherwise);
- the scenario columns of :data:`RV_SCENARIO_COLUMNS`, dur_source and
  dur_path (s), dt (s), npts (points of each series), te (the end of the
  window, s) and nsims (the number of simulations);
- the averages over the simulations, in cgs units (cm, cm/s, cm/s^2, cm/s
  for the Arias intensity, s for the durations): mean, std (with n - 1),
  sem = std/sqrt(nsims), gmean (geometric mean), gfactor = 10**std(log10)
  and gsem = gfactor**(1/sqrt(nsims)), the standard error of the geometric
  mean as a factor.

SMSIM gives the standard errors relative to the mean (``sem/psv``); here
sem is in the units of the measure. For an oscillator, PSV and SD are PSA
times T/(2*pi) and (T/(2*pi))**2, and their averages scale the same way.
"""

SERIES_COLUMNS = ("time", "acc", "vel", "dis")
"""Columns of :meth:`TDScenario.series`: time (s), acceleration (cm/s^2),
velocity (cm/s) and displacement (cm), from rest."""

FAS_SCENARIO_COLUMNS = (
    "f",
    "period",
    *_TERM_COLUMNS,
    "source",
    "gsprd",
    "q_path",
    "site_amp",
    "site_dim",
    "low_cut",
    "high_cut",
    "fds",
    "fvs",
    "fas",
)
"""The leading columns of :func:`fas_scenario`, one row per frequency.

f (Hz), period (s; inf at f = 0), the scenario columns of
:data:`RV_SCENARIO_COLUMNS`, the factors of the spectrum
(:class:`~pysmsim.spectrum.fas.SpectrumFactors`): source (the shape S(f)),
gsprd (geometrical spreading), q_path (the anelastic path factor),
site_amp (crustal amplification), site_dim (kappa and fmax filters),
low_cut and high_cut; and the Fourier spectra of ground displacement fds
(cm-s), velocity fvs (cm) and acceleration fas (cm/s).

Then, for each oscillator period T, ``osc_sd(T)``, ``osc_sv(T)`` and
``osc_sa(T)``: the Fourier spectra of the oscillator's displacement (cm-s)
and of its velocity and acceleration relative to the ground, 2*pi*f and
(2*pi*f)**2 times the displacement (SMSIM's ``FAS_OSCD``, ``FAS_OSCV`` and
``FAS_OSCA``).
"""


def _term_values(terms: ScenarioTerms, params: Params) -> dict[str, float]:
    stress = terms.source.stress
    return {
        "magnitude": terms.magnitude,
        "rrup": terms.rrup,
        "f_ff": terms.finite_fault_h,
        "rps": terms.rps,
        "kappa": params.site.kappa(terms.magnitude),
        "stress": math.nan if stress is None else stress,
        "fa": terms.source.fa,
        "fb": terms.source.fb,
    }


def _periods(periods: Sequence[float] | npt.ArrayLike) -> FloatArray:
    out = np.atleast_1d(np.asarray(periods, dtype=np.float64))
    if out.ndim != 1:
        raise ValueError("periods must be one-dimensional")
    if np.any(~(out > 0)):
        raise ValueError("oscillator periods must be positive")
    return out


# --- Random vibration ---------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RVScenario:
    """Random-vibration results of one scenario (Fortran ``gm_rv_drvr``).

    Attributes
    ----------
    params : Params
        The model parameters.
    terms : ScenarioTerms
        M0, corner frequencies, Rps and the other frequency-independent terms.
    pgd, pgv, pga : RVResult
        Peak ground displacement (cm), velocity (cm/s) and acceleration
        (cm/s^2), each with its intermediate quantities (``.peak`` is the
        value).
    arias : float
        Arias intensity pi/(2g) * m0 of the acceleration (cm/s), with
        g = 981 cm/s^2.
    spectrum : tuple of RVResult
        One result per oscillator period, in the order given (``.peak`` is
        PSV, ``.psa`` and ``.sd`` follow from it).
    """

    params: Params
    terms: ScenarioTerms
    pgd: RVResult
    pgv: RVResult
    pga: RVResult
    arias: float
    spectrum: tuple[RVResult, ...]

    @property
    def periods(self) -> FloatArray:
        """Oscillator periods (s)."""
        return np.array([r.response.period for r in self.spectrum])  # type: ignore[union-attr]

    @property
    def psa(self) -> FloatArray:
        """PSA at each period (cm/s^2)."""
        return np.array([r.psa for r in self.spectrum])

    def table(self) -> pd.DataFrame:
        """Every quantity, one row per measure (see :data:`RV_SCENARIO_COLUMNS`)."""
        base = _term_values(self.terms, self.params)
        rows: list[dict[str, Any]] = []
        motions = (("pgd", self.pgd), ("pgv", self.pgv), ("pga", self.pga))
        for measure, result in (*motions, *(("psa", r) for r in self.spectrum)):
            m = result.moments
            row: dict[str, Any] = {
                "measure": measure,
                **base,
                "fup": result.fup,
                "dur_source": result.dur_source,
                "dur_path": result.dur_path,
                "dur_ex": result.dur_ex,
                "dur_rms": result.dur_rms,
                "m0": m.m0,
                "m1": m.m1,
                "m2": m.m2,
                "m4": m.m4,
                "dominant_frequency": m.dominant_frequency,
                "bandwidth": m.bandwidth,
                "epsilon": m.epsilon,
                "n_zero_crossings": result.n_zero_crossings,
                "n_extrema": result.n_extrema,
                "rms": result.rms,
                "peak_factor": result.peak_factor,
                "peak_factor_cl": result.peak_factor_cl,
                "peak_factor_dk80_asymptotic": result.peak_factor_dk80_asymptotic,
                "peak_motion_factor": result.peak_motion_factor,
                "value": result.peak,
            }
            if isinstance(result.response, Oscillator):
                row["period"] = result.response.period
                row["damping"] = result.response.damping
                row["value"] = result.psa
                row["sd"] = result.sd
                row["psv"] = result.peak
                row["psa_pga"] = result.psa / self.pga.peak
            rows.append(row)
            if measure == "pga":
                rows.append({"measure": "arias", **base, "value": self.arias})
        table = pd.DataFrame(rows).reindex(columns=list(RV_SCENARIO_COLUMNS))
        numeric = [c for c in RV_SCENARIO_COLUMNS if c != "measure"]
        table[numeric] = table[numeric].astype(np.float64)
        return table


def rv_scenario(
    params: Params,
    magnitude: float,
    rrup: float,
    periods: Sequence[float] | npt.ArrayLike = (),
    damping: float = 0.05,
    *,
    fortran_compat: bool = False,
) -> RVScenario:
    """Peak motions of one scenario by random vibration (Fortran ``gm_rv_drvr``).

    Parameters
    ----------
    params : Params
        Model parameters.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    periods : sequence of float
        Oscillator periods (s) of the response spectrum; none by default.
    damping : float
        Oscillator damping, as a fraction of critical.
    fortran_compat : bool
        Compute fup from kappa_0 and the DK80 peak factor with SMSIM's sum
        (see :func:`~pysmsim.rv.gm_rv.rv_motion`).

    Returns
    -------
    RVScenario
    """
    correction = oscillator_correction(params.rv)

    def motion(response: GroundMotion | Oscillator) -> RVResult:
        return rv_motion(
            params,
            magnitude,
            rrup,
            response,
            correction=correction,
            fortran_compat=fortran_compat,
        )

    pga = motion(GroundMotion(MotionQuantity.ACCELERATION))
    return RVScenario(
        params=params,
        terms=pga.terms,
        pgd=motion(GroundMotion(MotionQuantity.DISPLACEMENT)),
        pgv=motion(GroundMotion(MotionQuantity.VELOCITY)),
        pga=pga,
        arias=math.pi / (2.0 * G) * pga.moments.m0,
        spectrum=tuple(motion(Oscillator(period=p, damping=damping)) for p in _periods(periods)),
    )


# --- Time domain --------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TDScenario:
    """Time-domain simulations of one scenario (Fortran ``gm_td_drvr``).

    Attributes
    ----------
    params : Params
        The model parameters.
    result : TDResult
        The simulations: the peak motions, Arias intensity, durations and
        oscillator responses of each, the time grid, and the acceleration
        series when kept.
    """

    params: Params
    result: TDResult

    def table(self) -> pd.DataFrame:
        """Averages over the simulations, one row per measure.

        See :data:`TD_SCENARIO_COLUMNS`.
        """
        r = self.result
        n = r.nsims
        base = {
            **_term_values(r.terms, self.params),
            "dur_source": r.dur_source,
            "dur_path": r.dur_path,
            "dt": r.grid.dt,
            "npts": r.grid.npts,
            "te": r.grid.te,
            "nsims": n,
        }
        series: list[tuple[str, float, FloatArray]] = [
            (name, math.nan, getattr(r, name))
            for name in ("pgd", "pgv", "pga", "arias", "dur_75_05", "dur_95_05")
        ]
        psa = r.psa
        series += [("psa", float(p), psa[:, k]) for k, p in enumerate(r.periods)]
        rows = []
        for measure, period, values in series:
            s = statistics(values)
            rows.append(
                {
                    "measure": measure,
                    "period": period,
                    "damping": r.damping if measure == "psa" else math.nan,
                    **base,
                    "mean": s.mean,
                    "std": s.std,
                    "sem": s.std / math.sqrt(n),
                    "gmean": s.gmean,
                    "gfactor": s.gfactor,
                    "gsem": s.gfactor ** (1.0 / math.sqrt(n)),
                }
            )
        table = pd.DataFrame(rows, columns=list(TD_SCENARIO_COLUMNS))
        numeric = [c for c in TD_SCENARIO_COLUMNS if c != "measure"]
        table[numeric] = table[numeric].astype(np.float64)
        return table

    def series(self, simulation: int = 0) -> pd.DataFrame:
        """Return the acceleration, velocity and displacement of one simulation.

        Parameters
        ----------
        simulation : int
            Index of the simulation, from 0 (SMSIM counts from 1).

        Returns
        -------
        pandas.DataFrame
            The columns :data:`SERIES_COLUMNS`, one row per time step of the
            whole series (``result.grid.npts`` rows, from time 0), as in
            the file ``gm_td_drvr`` writes.

        Raises
        ------
        ValueError
            If the series were not kept (``keep_series=False``).
        IndexError
            If there is no such simulation.
        """
        acc_all = self.result.acc
        if acc_all is None:
            raise ValueError("the series were not kept; use td_scenario(..., keep_series=True)")
        if not -self.result.nsims <= simulation < self.result.nsims:
            raise IndexError(f"simulation {simulation} out of range (nsims {self.result.nsims})")
        acc = acc_all[simulation]
        vel, dis = velocity_displacement(acc, self.result.grid.dt)
        return pd.DataFrame(
            {"time": self.result.grid.times(), "acc": acc, "vel": vel, "dis": dis},
            columns=list(SERIES_COLUMNS),
        )


def td_scenario(
    params: Params,
    magnitude: float,
    rrup: float,
    periods: Sequence[float] | npt.ArrayLike = (),
    damping: float = 0.05,
    *,
    nsims: int | None = None,
    keep_series: bool = False,
) -> TDScenario:
    """Time-domain simulations of one scenario (Fortran ``gm_td_drvr``).

    Parameters
    ----------
    params : Params
        Model parameters; ``params.td`` gives the window, the timing, the
        noise, the seed and the number of simulations.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    periods : sequence of float
        Oscillator periods (s); none by default.
    damping : float
        Oscillator damping, as a fraction of critical.
    nsims : int, optional
        Number of simulations (default ``params.td.nsims``).
    keep_series : bool
        Keep the acceleration series, for :meth:`TDScenario.series`
        (memory: nsims times the points of a series).

    Returns
    -------
    TDScenario
    """
    result = td_motion(
        params,
        magnitude,
        rrup,
        _periods(periods),
        damping,
        nsims=nsims,
        keep_series=keep_series,
    )
    return TDScenario(params=params, result=result)


# --- Fourier spectra ----------------------------------------------------------------


def osc_columns(period: float) -> tuple[str, str, str]:
    """Names of the oscillator FAS columns for a period (s), T with up to 6 digits."""
    return f"osc_sd({period:g})", f"osc_sv({period:g})", f"osc_sa({period:g})"


def fas_scenario(
    params: Params,
    magnitude: float,
    rrup: float,
    frequencies: npt.ArrayLike,
    periods: Sequence[float] | npt.ArrayLike = (),
    damping: float = 0.05,
) -> pd.DataFrame:
    """Fourier spectra of one scenario and their factors (Fortran ``FAS_DRVR``).

    Parameters
    ----------
    params : Params
        Model parameters.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    frequencies : array_like
        Frequencies (Hz), one-dimensional, >= 0.
    periods : sequence of float
        Oscillator periods (s) for the oscillator spectra; none by default.
    damping : float
        Oscillator damping, as a fraction of critical.

    Returns
    -------
    pandas.DataFrame
        One row per frequency, with the columns :data:`FAS_SCENARIO_COLUMNS`
        and three columns per period.

    Raises
    ------
    ValueError
        If the frequencies are not a one-dimensional array of values >= 0,
        or two periods give the same column names.

    Notes
    -----
    SMSIM writes 0 for spectral values below 1e-30; they are kept here.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1 or np.any(~(f >= 0)):
        raise ValueError("frequencies must be a one-dimensional array of values >= 0")
    periods_arr = _periods(periods)
    names = [osc_columns(p) for p in periods_arr]
    if len(set(names)) != len(names):
        raise ValueError("periods must be distinct to 6 significant digits")
    factors = spectrum_factors(f, params, magnitude, rrup)
    fds = factors.product()
    w = 2.0 * math.pi * f
    columns: dict[str, Any] = {
        "f": f,
        "period": np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0),
        **{k: np.full(f.size, v) for k, v in _term_values(factors.terms, params).items()},
        "source": factors.source,
        "gsprd": factors.spreading,
        "q_path": factors.q_path,
        "site_amp": factors.site_amplification,
        "site_dim": factors.site_diminution,
        "low_cut": factors.low_cut,
        "high_cut": factors.high_cut,
        "fds": fds,
        "fvs": w * fds,
        "fas": w * w * fds,
    }
    for period, (sd_name, sv_name, sa_name) in zip(periods_arr, names, strict=True):
        osc = Oscillator(period=float(period), damping=damping)
        sd = fds * osc.factor(f) / (2.0 * math.pi / period)
        columns[sd_name] = sd
        columns[sv_name] = w * sd
        columns[sa_name] = w * w * sd
    return pd.DataFrame(columns)
