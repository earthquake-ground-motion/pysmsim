"""Build source models in Python and compare their spectra.

No run file is needed: each source model is a class that takes only its own
parameters. Run from the repository root:

    python examples/compare_source_models.py

Needs matplotlib (``pip install -e ".[notebook]"``). The same lines work
in a Jupyter notebook.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysmsim.spectrum.source import (
    AtkinsonSilva2000,
    JiArchuleta2020,
    SingleCorner,
    TwoCornerAdditive,
)
from pysmsim.spectrum.stress import LogLinearStress, Stafford2022Stress

shear_velocity = 3.5  # km/s, near the source
magnitude = 6.0
frequencies = np.logspace(-2, 2, 200)  # Hz

sources = {
    # Brune omega-squared source, 100 bars at all magnitudes
    "single corner, 100 bars": SingleCorner(stress=LogLinearStress(stress_ref=100.0)),
    # The same shape with the Stafford et al. (2022) stress model
    "single corner, Stafford stress": SingleCorner(
        stress=Stafford2022Stress(c1=4.599, c2=0.4624)
    ),
    # Generalised additive two-corner model (Boore et al., 2014), fa and eps
    # as in Atkinson and Silva (2000)
    "two corner additive": TwoCornerAdditive(
        stress=LogLinearStress(stress_ref=100.0),
        log10_fa=(2.181, -0.496, 0.0),
        log10_eps=(0.605, -0.255, 0.0),
    ),
    # Models with corner frequencies fixed by magnitude take no parameters
    "Atkinson and Silva (2000)": AtkinsonSilva2000(),
    "Ji and Archuleta (2020)": JiArchuleta2020(),
}

fig, ax = plt.subplots(figsize=(6, 4.5))
for label, source in sources.items():
    terms = source.terms(magnitude, shear_velocity)
    acceleration = (2 * np.pi * frequencies) ** 2 * terms.m0 * source.shape(frequencies, terms)
    stress = "-" if terms.stress is None else f"{terms.stress:.3g} bars"
    print(
        f"{label:32s} stress {stress:>10s}  fa {terms.fa:.3g} Hz  fb {terms.fb:.3g} Hz  "
        f"duration {source.duration(terms):.3g} s"
    )
    ax.loglog(frequencies, acceleration, label=label)

ax.set(
    xlabel="Frequency (Hz)",
    ylabel="(2 pi f)^2 M0 S(f)",
    title=f"Acceleration source spectra, M {magnitude:g}",
)
ax.legend()
fig.tight_layout()
plt.show()
