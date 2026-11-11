"""Conversion of the Joyner-Boore distance Rjb to the rupture distance Rrup.

SMSIM's loop drivers take Rjb and compute Rrup = sqrt(Rjb**2 + h**2), where
the depth h depends on magnitude (Fortran ``h4rjb2rrup_f``)::

    from pysmsim.scenario import Mechanism, RjbToRrup
    from pysmsim.spectrum.distance import rjb_to_rrup
    rjb_to_rrup(10.0, 6.0, conversion=RjbToRrup.CY14_ZTOR, mechanism=Mechanism.STRIKE_SLIP)

References
----------
Chiou, B. S.-J. and R. R. Youngs (2014). Update of the Chiou and Youngs NGA
model for the average horizontal component of peak ground motion and
response spectra. Earthquake Spectra 30, 1117-1153 (the Ztor(M) relations,
as coded in SMSIM).
"""

from __future__ import annotations

import math

from pysmsim.scenario import DepthFunction, Mechanism, RjbToRrup

__all__ = ["cy14_ztor", "rjb_to_rrup", "rjb_to_rrup_depth"]


def cy14_ztor(magnitude: float, mechanism: Mechanism) -> float:
    """Depth to the top of rupture Ztor (km) of Chiou and Youngs (2014).

    Reverse faults: max(2.704 - 1.226*max(M - 5.849, 0), 0)**2; strike-slip
    and normal faults: max(2.673 - 1.136*max(M - 4.970, 0), 0)**2.
    """
    if mechanism is Mechanism.REVERSE:
        return max(2.704 - 1.226 * max(magnitude - 5.849, 0.0), 0.0) ** 2
    return max(2.673 - 1.136 * max(magnitude - 4.970, 0.0), 0.0) ** 2


def rjb_to_rrup_depth(
    magnitude: float,
    *,
    conversion: RjbToRrup,
    mechanism: Mechanism = Mechanism.STRIKE_SLIP,
    depth_function: DepthFunction | None = None,
) -> float:
    """Depth h (km) in Rrup = sqrt(Rjb**2 + h**2) (Fortran ``h4rjb2rrup_f``).

    Parameters
    ----------
    magnitude : float
        Moment magnitude.
    conversion : RjbToRrup
        How h is obtained.
    mechanism : Mechanism
        Fault mechanism, for ``cy14_ztor``.
    depth_function : DepthFunction, optional
        h(M), for ``magnitude_function`` (the default has h = 0).
    """
    if conversion is RjbToRrup.CY14_ZTOR:
        return cy14_ztor(magnitude, mechanism)
    if conversion is RjbToRrup.MAGNITUDE_FUNCTION:
        return (depth_function or DepthFunction()).depth(magnitude)
    return 0.0


def rjb_to_rrup(
    rjb: float,
    magnitude: float,
    *,
    conversion: RjbToRrup,
    mechanism: Mechanism = Mechanism.STRIKE_SLIP,
    depth_function: DepthFunction | None = None,
) -> float:
    """Rupture distance Rrup = sqrt(Rjb**2 + h**2) (km); see :func:`rjb_to_rrup_depth`."""
    h = rjb_to_rrup_depth(
        magnitude, conversion=conversion, mechanism=mechanism, depth_function=depth_function
    )
    return math.sqrt(rjb**2 + h**2)
