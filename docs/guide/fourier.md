# Fourier spectra

The Fourier amplitude spectrum of a scenario is the product of the source,
path, site, filter and response factors (SMSIM's `spect_amp`):

```text
A(f) = C · M0 · S(f) · G(f) · P(f) · A_site(f) · D_site(f) · L(f) · H(f) · R(f)
```

with C the radiation constant, M0 and S(f) the source, G(f) the geometrical
spreading, P(f) the anelastic (Q) path factor, A_site(f) the crustal
amplification, D_site(f) the kappa and fmax filters, L(f) and H(f) the
low-cut and high-cut filters and R(f) the response.

## One spectrum: `fourier_spectrum`

```python
import dataclasses

import numpy as np

from pysmsim.io.config import load_config
from pysmsim.spectrum.fas import fourier_spectrum
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator

config = load_config("examples/wna_bt15_example.toml")
params = config.params

f = np.logspace(-2, 2, 200)                       # Hz
fds = fourier_spectrum(f, params, 6.0, 20.0)      # M 6, Rrup 20 km: displacement (cm-s)
fas = fourier_spectrum(f, params, 6.0, 20.0, GroundMotion(MotionQuantity.ACCELERATION))  # cm/s
fvs = fourier_spectrum(f, params, 6.0, 20.0, GroundMotion(MotionQuantity.VELOCITY))      # cm
```

[`fourier_spectrum`][pysmsim.spectrum.fas.fourier_spectrum] takes the
frequencies (any shape), the parameters, the moment magnitude and the
**rupture distance** Rrup (km); the finite-fault factor turns Rrup into the
point-source distance Rps. Without a response it gives the displacement
spectrum. The other responses are:

| Response | Factor R(f) | Spectrum |
| --- | --- | --- |
| `GroundMotion()` | 1 | displacement, cm-s |
| `GroundMotion(MotionQuantity.VELOCITY)` | 2πf | velocity, cm |
| `GroundMotion(MotionQuantity.ACCELERATION)` | (2πf)² | acceleration, cm/s |
| `Oscillator(period=T, damping=0.05)` | the oscillator's pseudo-velocity response | oscillator pseudo-velocity |
| `WoodAnderson()`, `WWSSNShortPeriod()` | the seismometer response | |

```python
psv_1s = fourier_spectrum(f, params, 6.0, 20.0, Oscillator(period=1.0, damping=0.05))
```

To plot the spectra, for example with matplotlib:

```text
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for m in (5.0, 6.0, 7.0):
    ax.loglog(f, fourier_spectrum(f, params, m, 20.0, GroundMotion(MotionQuantity.ACCELERATION)),
              label=f"M {m}")
ax.set(xlabel="Frequency (Hz)", ylabel="Fourier acceleration (cm/s)")
ax.legend()
```

## The factors of the spectrum

[`spectrum_factors`][pysmsim.spectrum.fas.spectrum_factors] gives each
factor on the same frequencies, and
[`scenario_terms`][pysmsim.spectrum.fas.scenario_terms] the quantities
that do not depend on frequency:

```python
from pysmsim.spectrum.fas import scenario_terms, spectrum_factors

terms = scenario_terms(params, 6.0, 20.0)
terms.rps                    # point-source distance (km)
terms.finite_fault_h         # finite-fault factor h (km)
terms.source.m0              # seismic moment (dyne-cm)
terms.source.fa              # corner frequency (Hz)
terms.radiation_constant

factors = spectrum_factors(f, params, 6.0, 20.0)
factors.source               # S(f)
factors.spreading            # G(f)
factors.q_path               # exp(-pi f R / (c_q Q(f)))
factors.site_amplification
factors.site_diminution      # kappa and fmax
factors.low_cut, factors.high_cut, factors.response
np.allclose(factors.product(), fds)   # their product is the spectrum
```

This makes it easy to see how one part of the model shapes the spectrum,
for example kappa:

```python
for kappa0 in (0.01, 0.035, 0.06):
    p = dataclasses.replace(params, site=dataclasses.replace(params.site, kappa0=kappa0))
    print(kappa0, fourier_spectrum([10.0], p, 6.0, 20.0)[0])
```

## One scenario with every factor: `fas_scenario`

[`fas_scenario`][pysmsim.drivers.single.fas_scenario] (SMSIM's
`FAS_DRVR`) gives one table with the factors, the displacement, velocity
and acceleration spectra and, for each period asked for, the oscillator
spectra:

```python
from pysmsim.drivers.single import fas_scenario

table = fas_scenario(params, 6.0, 20.0, np.logspace(-2, 2, 41), periods=[0.1, 1.0])
table[["f", "source", "gsprd", "q_path", "site_amp", "site_dim", "fas"]].head()
```

The columns are listed in
[`FAS_SCENARIO_COLUMNS`][pysmsim.drivers.single.FAS_SCENARIO_COLUMNS]; the
oscillator columns are `osc_sd(T)`, `osc_sv(T)` and `osc_sa(T)`.

## Your own loop

`fourier_spectrum` is cheap (a few tens of microseconds for 200
frequencies), so a loop over scenarios in Python is fast enough for most
uses. Here, a table of spectra for every magnitude and distance:

```python
import pandas as pd

rows = []
for magnitude in (5.0, 6.0, 7.0):
    for rrup in (10.0, 30.0, 100.0):
        fas = fourier_spectrum(f, params, magnitude, rrup, GroundMotion(MotionQuantity.ACCELERATION))
        rows.append({"magnitude": magnitude, "rrup": rrup, "fas": fas})
spectra = pd.DataFrame(rows)
```

## The loop over a grid: `fas_loop`

[`fas_loop`][pysmsim.drivers.loop_fas.fas_loop] computes the Fourier
acceleration spectrum of every scenario of a
[`ScenarioGrid`][pysmsim.scenario.ScenarioGrid], one row per scenario with
a column `fas(f)` per frequency:

```python
from pysmsim.drivers.loop_fas import fas_loop, fas_loop_details

grid = config.scenarios        # M 4, 6, 8; Rjb 10 km; kappa_0 0.02 and 0.05 s
wide = fas_loop(params, grid, grid.frequencies)
wide.iloc[:, :10]
```

The grid's Joyner-Boore distances are converted to Rrup (the columns `h`
and `rrup`), and its kappa_0 and stress values replace those of the
parameters (see [`params_for`][pysmsim.drivers.loop_fas.params_for]).
[`fas_loop_details`][pysmsim.drivers.loop_fas.fas_loop_details] gives one
row per scenario and frequency, with the displacement spectrum and the
spreading and Q factors (SMSIM's `fmrsk_loop_fas_drvr`):

```python
long = fas_loop_details(params, grid, [0.1, 1.0, 10.0])
long[["f", "magnitude", "rrup", "kappa0", "fas", "gsprd", "q"]]
```

A grid can also be built in Python:

```python
from pysmsim.scenario import DistanceMetric, ScenarioGrid

my_grid = ScenarioGrid(
    magnitudes=(5.0, 6.0, 7.0),
    distances=(10.0, 30.0, 100.0),
    distance_metric=DistanceMetric.RRUP,     # the distances are Rrup
)
wide = fas_loop(params, my_grid, f)
```

The rows of `fas_loop` match the loop above: each is
`fourier_spectrum(f, p, magnitude, rrup, GroundMotion(MotionQuantity.ACCELERATION))`,
with `p` the parameters after `params_for` puts in the grid's kappa_0 and
stress.
