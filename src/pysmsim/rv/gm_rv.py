"""Peak motions of one scenario by random vibration (Fortran ``gm_rv``).

:func:`rv_motion` computes the peak ground displacement, velocity or
acceleration, or an oscillator's peak pseudo-velocity, for one magnitude
and rupture distance::

    from pysmsim.io.config import load_config
    from pysmsim.rv.gm_rv import rv_motion
    from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator
    params = load_config("run.toml").params
    pga = rv_motion(params, 6.0, 20.0, GroundMotion(MotionQuantity.ACCELERATION)).peak
    psv = rv_motion(params, 6.0, 20.0, Oscillator(period=1.0, damping=0.05)).peak

The steps are those of SMSIM's ``get_motion``, each a public function:

1. the spectral moments of the Fourier spectrum up to fup
   (:mod:`pysmsim.rv.moments`);
2. the excitation duration (source plus path) and the RMS duration, with
   the oscillator correction (:mod:`pysmsim.rv.duration`);
3. the RMS motion sqrt(m0/Drms), the numbers of zero crossings and extrema
   over Dex, and the peak factors (:mod:`pysmsim.rv.peak_factors`).

The peak is the RMS motion times the DK80 peak factor (times the BT15 TD/RV
ratio for PGA and PGV), as in SMSIM. The result keeps every intermediate
quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pysmsim.params import Params
from pysmsim.rv.duration import OscillatorCorrection, oscillator_correction
from pysmsim.rv.moments import SpectralMoments, spectral_moments, upper_frequency
from pysmsim.rv.peak_factors import (
    cartwright_longuet_higgins,
    der_kiureghian_1980,
    der_kiureghian_1980_asymptotic,
    der_kiureghian_1980_fortran,
    extrema,
    zero_crossings,
)
from pysmsim.spectrum.fas import ScenarioTerms, fourier_spectrum, scenario_terms
from pysmsim.spectrum.response import GroundMotion, Oscillator

__all__ = ["RVResult", "rv_fup", "rv_motion"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RVResult:
    """The peak motion of one scenario and the quantities behind it.

    Attributes
    ----------
    terms : ScenarioTerms
        M0, corner frequencies, Rps and the other frequency-independent terms.
    response : GroundMotion or Oscillator
        The response.
    fup : float
        Upper limit of the moment integrals (Hz).
    moments : SpectralMoments
        m0, m1, m2, m4 (of the pseudo-velocity for an oscillator).
    dur_source, dur_path, dur_ex : float
        Source, path and excitation durations (s).
    dur_rms : float
        RMS duration (s); Dex for ground motion.
    rms : float
        RMS motion sqrt(m0/Drms).
    n_zero_crossings, n_extrema : float
        Numbers of zero crossings and extrema over Dex (``anz``, ``ane``).
    peak_factor : float
        DK80 peak factor (eq. 2 of Der Kiureghian, 1980).
    peak_factor_steps : int or None
        Final number of steps of SMSIM's sum for the DK80 peak factor
        (``na_dk80_eq2``); None unless ``fortran_compat``.
    peak_factor_cl : float
        Cartwright and Longuet-Higgins (1956) eq. 6.8 peak factor, for
        comparison.
    peak_factor_dk80_asymptotic : float
        Der Kiureghian's asymptotic peak factor, for comparison.
    peak_motion_factor : float
        TD/RV ratio applied to PGA and PGV by the BT15 corrections (else 1).
    peak : float
        The peak motion: peak_motion_factor * rms * peak_factor. For an
        oscillator, the peak pseudo-velocity PSV (cm/s); PSA = 2*pi/T * PSV
        and SD = T/(2*pi) * PSV. NaN where a BT15 table does not reach.
    """

    terms: ScenarioTerms
    response: GroundMotion | Oscillator
    fup: float
    moments: SpectralMoments
    dur_source: float
    dur_path: float
    dur_ex: float
    dur_rms: float
    rms: float
    n_zero_crossings: float
    n_extrema: float
    peak_factor: float
    peak_factor_steps: int | None
    peak_factor_cl: float
    peak_factor_dk80_asymptotic: float
    peak_motion_factor: float
    peak: float

    @property
    def psa(self) -> float:
        """Pseudo-acceleration 2*pi/T * PSV (cm/s^2); oscillators only."""
        if not isinstance(self.response, Oscillator):
            raise TypeError("psa is defined for an oscillator response only")
        return 2.0 * math.pi / self.response.period * self.peak

    @property
    def sd(self) -> float:
        """Relative displacement T/(2*pi) * PSV (cm); oscillators only."""
        if not isinstance(self.response, Oscillator):
            raise TypeError("sd is defined for an oscillator response only")
        return self.response.period / (2.0 * math.pi) * self.peak


def rv_fup(params: Params, magnitude: float, *, fortran_compat: bool = False) -> float:
    """Upper integration frequency for a scenario (Fortran ``get_fup``).

    Uses kappa at the scenario magnitude; with ``fortran_compat``, kappa_0
    (``akappa``), as SMSIM's drivers do.
    """
    site = params.site
    kappa = site.kappa0 if fortran_compat else site.kappa(magnitude)
    return upper_frequency(site.fmax, kappa, params.rv.amp_cutoff)


def rv_motion(
    params: Params,
    magnitude: float,
    rrup: float,
    response: GroundMotion | Oscillator,
    *,
    correction: OscillatorCorrection | None = None,
    fortran_compat: bool = False,
) -> RVResult:
    """Peak motion of one scenario by random vibration (Fortran ``gm_rv``).

    Parameters
    ----------
    params : Params
        Model parameters; ``params.rv`` gives the integration settings and
        the oscillator correction.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    response : GroundMotion or Oscillator
        Ground displacement, velocity or acceleration, or an oscillator.
    correction : OscillatorCorrection, optional
        The RMS-duration correction; by default the one ``params.rv`` names
        (pass it to avoid looking it up for every scenario).
    fortran_compat : bool
        Compute fup from kappa_0 instead of kappa at the magnitude, and the
        DK80 peak factor with SMSIM's sum instead of an accurate integral
        (:func:`~pysmsim.rv.peak_factors.der_kiureghian_1980_fortran`).

    Returns
    -------
    RVResult
    """
    if not isinstance(response, (GroundMotion, Oscillator)):
        raise TypeError("rv_motion supports ground motion and oscillator responses")
    if correction is None:
        correction = oscillator_correction(params.rv)
    terms = scenario_terms(params, magnitude, rrup)
    fup = rv_fup(params, magnitude, fortran_compat=fortran_compat)
    breakpoints = (response.frequency,) if isinstance(response, Oscillator) else ()
    moments = spectral_moments(
        lambda f: fourier_spectrum(f, params, magnitude, rrup, response),
        fup,
        rtol=params.rv.eps_int,
        breakpoints=breakpoints,
    )

    dur_source = params.source.duration(terms.source)
    dur_path = params.path_duration.duration(terms.rps)
    dur_ex = dur_source + dur_path
    bandwidth = moments.bandwidth
    if isinstance(response, Oscillator):
        dur_rms = correction.rms_duration(
            dur_ex,
            period=response.period,
            damping=response.damping,
            magnitude=magnitude,
            rps=terms.rps,
            bandwidth=bandwidth,
        )
        peak_motion_factor = 1.0
        damping: float | None = response.damping
    else:
        dur_rms = dur_ex
        peak_motion_factor = correction.peak_motion_factor(response.quantity, magnitude, terms.rps)
        damping = None

    rms = math.sqrt(moments.m0 / dur_rms) if dur_rms > 0 else math.nan
    n_zero = zero_crossings(moments.dominant_frequency, dur_ex)
    n_ext = extrema(moments.extrema_frequency, dur_ex)
    steps: int | None = None
    if fortran_compat:
        peak_factor, steps = der_kiureghian_1980_fortran(n_zero, bandwidth, params.rv.eps_int)
    else:
        peak_factor = der_kiureghian_1980(n_zero, bandwidth)
    return RVResult(
        terms=terms,
        response=response,
        fup=fup,
        moments=moments,
        dur_source=dur_source,
        dur_path=dur_path,
        dur_ex=dur_ex,
        dur_rms=dur_rms,
        rms=rms,
        n_zero_crossings=n_zero,
        n_extrema=n_ext,
        peak_factor=peak_factor,
        peak_factor_steps=steps,
        peak_factor_cl=cartwright_longuet_higgins(n_ext, moments.xi, params.rv.zup),
        peak_factor_dk80_asymptotic=der_kiureghian_1980_asymptotic(n_zero, bandwidth, damping),
        peak_motion_factor=peak_motion_factor,
        peak=peak_motion_factor * rms * peak_factor,
    )
