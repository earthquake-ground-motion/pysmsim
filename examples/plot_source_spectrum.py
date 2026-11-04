"""Load a TOML run file and plot its source spectrum.

Run from the repository root:

    python examples/plot_source_spectrum.py

Needs matplotlib (``pip install -e ".[notebook]"``). The same lines work
in a Jupyter notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysmsim.io.config import load_config

HERE = Path(__file__).parent

config = load_config(HERE / "wna_bt15_example.toml")
source = config.params.source
beta = config.params.medium.shear_velocity
print(source)  # the source model and its parameters

frequencies = np.logspace(-2, 2, 200)  # Hz
magnitudes = (4.0, 5.0, 6.0, 7.0, 8.0)

fig, (ax_disp, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for magnitude in magnitudes:
    terms = source.terms(magnitude, beta)  # M0, stress, fa, fb, eps
    displacement = terms.m0 * source.shape(frequencies, terms)  # M0 * S(f)
    acceleration = (2 * np.pi * frequencies) ** 2 * displacement
    print(
        f"M {magnitude}: M0 = {terms.m0:.3g} dyne-cm, fa = {terms.fa:.3g} Hz, "
        f"fb = {terms.fb:.3g} Hz, duration = {source.duration(terms):.3g} s"
    )
    ax_disp.loglog(frequencies, displacement, label=f"M {magnitude:g}")
    ax_acc.loglog(frequencies, acceleration)

ax_disp.set(xlabel="Frequency (Hz)", ylabel="M0 S(f) (dyne-cm)", title="Displacement source")
ax_acc.set(xlabel="Frequency (Hz)", ylabel="(2 pi f)^2 M0 S(f)", title="Acceleration source")
ax_disp.legend()
fig.suptitle(f"{source.name}, {config.params.title}")
fig.tight_layout()
plt.show()
