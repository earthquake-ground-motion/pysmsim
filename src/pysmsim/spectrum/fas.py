"""The Fourier amplitude spectrum of one scenario (Fortran ``spect_amp``).

The spectrum is the product of the factors in the other modules of
:mod:`pysmsim.spectrum`::

    A(f) = C * M0 * S(f) * G(f) * P(f) * A_site(f) * D_site(f) * L(f) * H(f) * R(f)

with C the radiation constant (:meth:`pysmsim.params.Medium.radiation_constant`),
M0 and S(f) the source, G(f) the geometrical spreading, P(f) the anelastic
(Q) path factor, A_site(f) the crustal amplification, D_site(f) the kappa
and fmax filters, L(f) and H(f) the low-cut and high-cut filters and R(f)
the response (:mod:`pysmsim.spectrum.response`; ground displacement by
default, which gives the displacement spectrum in cm-s).

From a notebook::

    from pysmsim.io.config import load_config
    from pysmsim.spectrum.fas import fourier_spectrum, spectrum_factors
    from pysmsim.spectrum.response import GroundMotion, MotionQuantity
    params = load_config("run.toml").params
    acc = GroundMotion(MotionQuantity.ACCELERATION)
    fas = fourier_spectrum(frequencies, params, magnitude=6.0, rrup=20.0, response=acc)
    factors = spectrum_factors(frequencies, params, magnitude=6.0, rrup=20.0)
    factors.spreading, factors.site_amplification, ...

:func:`scenario_terms` gives the quantities that do not depend on
frequency (M0, the corner frequencies, Rps and the radiation constant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pysmsim.params import Params
from pysmsim.spectrum.response import GroundMotion, Response
from pysmsim.spectrum.source import SourceTerms

__all__ = [
    "ScenarioTerms",
    "SpectrumFactors",
    "fourier_spectrum",
    "scenario_terms",
    "spectrum_factors",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioTerms:
    """Quantities of one scenario that do not depend on frequency.

    Attributes
    ----------
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    finite_fault_h : float
        Finite-fault factor h (km) (Fortran ``f_ff``).
    rps : float
        Point-source distance (km) (Fortran ``rmod``).
    source : SourceTerms
        M0, stress and corner frequencies (Fortran ``spect_scale``).
    radiation_constant : float
        C, with the r_ref of the spreading model (Fortran ``const``).
    """

    magnitude: float
    rrup: float
    finite_fault_h: float
    rps: float
    source: SourceTerms
    radiation_constant: float


def scenario_terms(params: Params, magnitude: float, rrup: float) -> ScenarioTerms:
    """Return the frequency-independent quantities of a scenario.

    Parameters
    ----------
    params : Params
        Model parameters.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).

    Returns
    -------
    ScenarioTerms
    """
    return ScenarioTerms(
        magnitude=magnitude,
        rrup=rrup,
        finite_fault_h=params.finite_fault.h(magnitude),
        rps=params.finite_fault.rps(rrup, magnitude),
        source=params.source.terms(magnitude, params.medium.shear_velocity),
        radiation_constant=params.medium.radiation_constant(params.spreading.reference_distance()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectrumFactors:
    """The factors of the Fourier spectrum on an array of frequencies.

    Each array has the shape of the frequencies; :meth:`product` multiplies
    them together into the spectrum.

    Attributes
    ----------
    terms : ScenarioTerms
        The frequency-independent quantities.
    source : ndarray
        Source shape S(f) (Fortran ``spect_shape``).
    spreading : ndarray
        Geometrical spreading G(f).
    q_path : ndarray
        Anelastic path factor exp(-pi*f*R/(c_q*Q(f))).
    site_amplification : ndarray
        Crustal amplification (Fortran ``crustal_amp_factor``).
    site_diminution : ndarray
        Kappa and fmax filters.
    low_cut, high_cut : ndarray
        Filters (Fortran ``buttrlcf``, ``high_cut_filter``).
    response : ndarray
        Response factor R(f).
    """

    terms: ScenarioTerms
    source: FloatArray
    spreading: FloatArray
    q_path: FloatArray
    site_amplification: FloatArray
    site_diminution: FloatArray
    low_cut: FloatArray
    high_cut: FloatArray
    response: FloatArray

    def product(self) -> FloatArray:
        """Return the Fourier spectrum: C * M0 times all the factors."""
        t = self.terms
        return (
            t.radiation_constant
            * t.source.m0
            * self.spreading
            * self.low_cut
            * self.high_cut
            * self.source
            * self.site_amplification
            * self.q_path
            * self.site_diminution
            * self.response
        )


def spectrum_factors(
    f: npt.ArrayLike,
    params: Params,
    magnitude: float,
    rrup: float,
    response: Response | None = None,
) -> SpectrumFactors:
    """Return each factor of the Fourier spectrum of a scenario.

    Parameters
    ----------
    f : array_like
        Frequencies (Hz), any shape.
    params : Params
        Model parameters.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km).
    response : Response, optional
        Response; ground displacement by default.

    Returns
    -------
    SpectrumFactors
    """
    freqs = np.asarray(f, dtype=np.float64)
    terms = scenario_terms(params, magnitude, rrup)
    rps = terms.rps
    return SpectrumFactors(
        terms=terms,
        source=params.source.shape(freqs, terms.source),
        spreading=params.spreading.factor(freqs, rps, rrup, magnitude),
        q_path=params.q.attenuation(freqs, magnitude, rps, rrup),
        site_amplification=params.site.amplification(freqs),
        site_diminution=params.site.diminution(freqs, magnitude),
        low_cut=params.filters.low_cut.factor(freqs),
        high_cut=params.filters.high_cut.factor(freqs),
        response=(response or GroundMotion()).factor(freqs),
    )


def fourier_spectrum(
    f: npt.ArrayLike,
    params: Params,
    magnitude: float,
    rrup: float,
    response: Response | None = None,
) -> FloatArray:
    """Fourier amplitude spectrum of a scenario (Fortran ``spect_amp``).

    Parameters
    ----------
    f : array_like
        Frequencies (Hz), any shape.
    params : Params
        Model parameters.
    magnitude : float
        Moment magnitude.
    rrup : float
        Rupture distance (km); the finite-fault model turns it into Rps.
    response : Response, optional
        Response; ground displacement by default, which gives the spectrum
        in cm-s (multiply by (2*pi*f)**2, or use
        ``GroundMotion(MotionQuantity.ACCELERATION)``, for acceleration in
        cm/s).

    Returns
    -------
    ndarray
        The spectrum, with the shape of ``f``.
    """
    return spectrum_factors(f, params, magnitude, rrup, response).product()
