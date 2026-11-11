"""Build every input in Python and compute Fourier amplitude spectra.

No run file is needed: each part of the model (medium, source, finite-fault
factor, spreading, Q, site and filters) is built from its class, collected
in a ``Params`` object and passed to ``fourier_spectrum``. Run from the
repository root:

    python examples/fourier_spectrum_by_hand.py

Needs matplotlib (``pip install -e ".[notebook]"``). The same lines work
in a Jupyter notebook.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysmsim.params import Medium, Params, PathDuration
from pysmsim.spectrum.attenuation import SimpleQ
from pysmsim.spectrum.fas import fourier_spectrum, scenario_terms, spectrum_factors
from pysmsim.spectrum.filters import Filters, LowCutFilter
from pysmsim.spectrum.finite_fault import TwoLinesSmoothed
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator
from pysmsim.spectrum.site import Site
from pysmsim.spectrum.source import SingleCorner
from pysmsim.spectrum.spreading import Hinged, SpreadingSegment
from pysmsim.spectrum.stress import LogLinearStress

# --- The model ------------------------------------------------------------

params = Params(
    title="WNA by hand",
    medium=Medium(density=2.72, shear_velocity=3.5),
    # Brune source with 100 bars at all magnitudes
    source=SingleCorner(stress=LogLinearStress(stress_ref=100.0)),
    # Boore and Thompson (2015) finite-fault factor, Rps = sqrt(Rrup**2 + h**2)
    finite_fault=TwoLinesSmoothed(c1=-1.72, c2=0.43, c3=-0.405, c4=0.235, delta_m=2.0),
    # 1/R to 40 km, then 1/sqrt(R)
    spreading=Hinged(
        segments=(
            SpreadingSegment(r_start=1.0, a=-1.0),
            SpreadingSegment(r_start=40.0, a=-0.5),
        )
    ),
    q=SimpleQ(q0=180.0, eta=0.45),
    # The spectrum does not use the path duration (the RV and TD steps do)
    path_duration=PathDuration(distances=(0.0, 10.0, 70.0), durations=(0.0, 0.0, 9.6)),
    # Arrays work as well as tuples for the amplification table
    site=Site(
        amp_frequencies=np.array([0.01, 0.1, 1.0, 4.0, 10.0, 80.0]),
        amplifications=np.array([1.00, 1.13, 1.58, 2.18, 2.75, 3.96]),
        kappa0=0.035,
    ),
    filters=Filters(low_cut=LowCutFilter(fcut=0.05, order=4)),
)

frequencies = np.logspace(-2, 2, 200)  # Hz
magnitude = 6.5
rrup = 20.0  # km

# --- The frequency-independent terms ----------------------------------------

terms = scenario_terms(params, magnitude, rrup)
print(f"M0 = {terms.source.m0:.3e} dyne-cm, fa = {terms.source.fa:.3f} Hz")
print(f"h = {terms.finite_fault_h:.2f} km, Rps = {terms.rps:.2f} km")
print(f"radiation constant = {terms.radiation_constant:.3e}")

# --- Spectra --------------------------------------------------------------

displacement = fourier_spectrum(frequencies, params, magnitude, rrup)  # cm-s
acceleration = fourier_spectrum(
    frequencies, params, magnitude, rrup, GroundMotion(MotionQuantity.ACCELERATION)
)  # cm/s
psv_1s = fourier_spectrum(
    frequencies, params, magnitude, rrup, Oscillator(period=1.0, damping=0.05)
)  # cm, oscillator pseudo-velocity

# Each factor separately, e.g. to see what shapes the spectrum
factors = spectrum_factors(frequencies, params, magnitude, rrup)
assert np.allclose(factors.product(), displacement)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for m in (4.5, 5.5, 6.5, 7.5):
    fas = fourier_spectrum(frequencies, params, m, rrup, GroundMotion(MotionQuantity.ACCELERATION))
    ax1.loglog(frequencies, fas, label=f"M {m:g}")
ax1.set(
    xlabel="Frequency (Hz)",
    ylabel="Fourier acceleration (cm/s)",
    title=f"Acceleration spectra, Rrup {rrup:g} km",
)
ax1.legend()

for name in ("source", "spreading", "q_path", "site_amplification", "site_diminution", "low_cut"):
    ax2.loglog(frequencies, getattr(factors, name), label=name)
ax2.set(
    xlabel="Frequency (Hz)",
    ylabel="Factor",
    title=f"Factors of the spectrum, M {magnitude:g}",
    ylim=(1e-3, 10.0),
)
ax2.legend()
fig.tight_layout()
plt.show()
