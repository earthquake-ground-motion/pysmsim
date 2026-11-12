# Random vibration

The random-vibration (RV) method turns the Fourier spectrum into a peak
motion without simulating a time series (Boore, 2003):

1. the spectral moments m0, m1, m2 and m4 of the spectrum, integrated up to
   an upper frequency fup ([`pysmsim.rv.moments`][pysmsim.rv.moments]);
2. the excitation duration (source plus path) and the RMS duration, with
   the oscillator correction ([`pysmsim.rv.duration`][pysmsim.rv.duration]);
3. the RMS motion sqrt(m0/Drms), and the peak factor from the numbers of
   zero crossings and extrema ([`pysmsim.rv.peak_factors`][pysmsim.rv.peak_factors]).

The peak is the RMS motion times the Der Kiureghian (1980) peak factor
(times the BT15 TD/RV ratio for PGA and PGV), as in SMSIM. The result is
deterministic.

## One motion: `rv_motion`

[`rv_motion`][pysmsim.rv.gm_rv.rv_motion] computes one peak motion of one
scenario: the parameters, the magnitude, the **rupture distance** Rrup
(km) and the response.

```python
import dataclasses
import math

import numpy as np

from pysmsim.io.config import load_config
from pysmsim.rv.gm_rv import rv_motion
from pysmsim.spectrum.response import GroundMotion, MotionQuantity, Oscillator

config = load_config("examples/wna_bt15_example.toml")
params = config.params

pga = rv_motion(params, 6.0, 20.0, GroundMotion(MotionQuantity.ACCELERATION))
pgv = rv_motion(params, 6.0, 20.0, GroundMotion(MotionQuantity.VELOCITY))
pgd = rv_motion(params, 6.0, 20.0, GroundMotion())             # displacement
osc = rv_motion(params, 6.0, 20.0, Oscillator(period=1.0, damping=0.05))

pga.peak          # PGA (cm/s^2)
pgv.peak          # PGV (cm/s)
pgd.peak          # PGD (cm)
osc.peak          # peak pseudo-velocity PSV (cm/s)
osc.psa           # PSA = 2*pi/T * PSV (cm/s^2)
osc.sd            # SD = T/(2*pi) * PSV (cm)
```

The result, an [`RVResult`][pysmsim.rv.gm_rv.RVResult], keeps every
intermediate quantity:

```python
pga.fup                          # upper integration frequency (Hz)
pga.moments.m0, pga.moments.m2   # spectral moments
pga.moments.bandwidth            # sqrt(1 - m1^2/(m0 m2))
pga.dur_source, pga.dur_path     # durations (s)
pga.dur_ex, pga.dur_rms          # excitation and RMS durations (s)
pga.rms                          # RMS acceleration (cm/s^2)
pga.n_zero_crossings, pga.n_extrema
pga.peak_factor                  # DK80, the one used
pga.peak_factor_cl               # Cartwright and Longuet-Higgins, for comparison
pga.peak_motion_factor           # BT15 TD/RV ratio (1 for other corrections)
pga.terms.rps, pga.terms.source.fa
```

The peak is NaN where a BT15 table does not cover the magnitude and
distance.

### A response spectrum

A response spectrum is one `rv_motion` per period. The oscillator
correction (for example the BT15 tables) is looked up once and passed in,
so that it is not read again for every period:

```python
from pysmsim.rv.duration import oscillator_correction

correction = oscillator_correction(params.rv)
periods = np.logspace(-2, 1, 50)
psa = np.array([
    rv_motion(params, 6.0, 20.0, Oscillator(period=t, damping=0.05), correction=correction).psa
    for t in periods
])
```

### Inside `rv_motion`

The steps are public functions too, for example to look at the moments or
the peak factors on their own:

```python
from pysmsim.rv.gm_rv import rv_fup
from pysmsim.rv.moments import spectral_moments
from pysmsim.rv.peak_factors import der_kiureghian_1980, zero_crossings
from pysmsim.spectrum.fas import fourier_spectrum

acc = GroundMotion(MotionQuantity.ACCELERATION)
fup = rv_fup(params, 6.0)
moments = spectral_moments(lambda f: fourier_spectrum(f, params, 6.0, 20.0, acc), fup)
nz = zero_crossings(moments.dominant_frequency, pga.dur_ex)
pf = der_kiureghian_1980(nz, moments.bandwidth)
rms = math.sqrt(moments.m0 / pga.dur_rms)
rms * pf * pga.peak_motion_factor          # PGA again
```

## One scenario in full: `rv_scenario`

[`rv_scenario`][pysmsim.drivers.single.rv_scenario] (SMSIM's
`gm_rv_drvr`) computes PGD, PGV, PGA, the Arias intensity and a response
spectrum for one magnitude and Rrup, and gives them as one table with every
intermediate quantity:

```python
from pysmsim.drivers.single import rv_scenario

rv = rv_scenario(params, 6.0, 20.0, periods=[0.1, 0.2, 0.5, 1.0, 2.0])
rv.pga.peak, rv.pgv.peak, rv.arias
rv.psa                         # PSA at each period (cm/s^2)
rv.table()                     # one row per measure, all quantities
```

## Your own loop

A loop over scenarios is a few lines. This one computes PGA and PSA(1 s)
for every magnitude and distance of a list, with the distances as Rrup:

```python
import pandas as pd

rows = []
for magnitude in (5.0, 6.0, 7.0):
    for rrup in (10.0, 30.0, 100.0):
        rows.append({
            "magnitude": magnitude,
            "rrup": rrup,
            "pga": rv_motion(params, magnitude, rrup, acc, correction=correction).peak,
            "psa(1)": rv_motion(
                params, magnitude, rrup, Oscillator(period=1.0), correction=correction
            ).psa,
        })
table = pd.DataFrame(rows)
```

With Joyner-Boore distances, convert them to Rrup first with
[`rjb_to_rrup`][pysmsim.spectrum.distance.rjb_to_rrup], as the loop
drivers do:

```python
from pysmsim.scenario import Mechanism, RjbToRrup
from pysmsim.spectrum.distance import rjb_to_rrup

rrup = rjb_to_rrup(10.0, 6.0, conversion=RjbToRrup.CY14_ZTOR, mechanism=Mechanism.STRIKE_SLIP)
```

## The loop over a grid: `rv_loop`

[`rv_loop`][pysmsim.drivers.loop_rv.rv_loop] (SMSIM's
`tmrsk_loop_rv_drvr`) computes PGV, PGA and the response spectrum for
every scenario of a [`ScenarioGrid`][pysmsim.scenario.ScenarioGrid]: every
combination of stress, kappa_0, magnitude and distance. It returns one row
per scenario with a column per measure: `pgv`, `pga` and `psa(T)`.

```python
from pysmsim.drivers.loop_rv import rv_loop, rv_loop_details

grid = config.scenarios        # M 4, 6, 8; Rjb 10 km; kappa_0 0.02 and 0.05 s; 200 periods
wide = rv_loop(params, grid)
wide[["magnitude", "rjb", "rrup", "kappa0", "pgv", "pga"]]
wide.filter(like="psa(")       # the 200 periods, psa(0.01) to psa(100)
```

[`rv_loop_details`][pysmsim.drivers.loop_rv.rv_loop_details] gives one row
per scenario and measure, with every intermediate quantity (the columns of
[`RV_DETAIL_COLUMNS`][pysmsim.drivers.loop_rv.RV_DETAIL_COLUMNS]):

```python
long = rv_loop_details(params, grid)
long[long.measure == "pga"][["magnitude", "kappa0", "fup", "dur_ex", "peak_factor", "pga"]]
```

A smaller grid for a quick look, built in Python:

```python
from pysmsim.scenario import DistanceMetric, ScenarioGrid

my_grid = ScenarioGrid(
    magnitudes=(5.0, 6.0, 7.0),
    distances=(10.0, 30.0, 100.0),
    distance_metric=DistanceMetric.RRUP,
    periods=(0.1, 1.0),
)
mine = rv_loop(params, my_grid)
```

Each row of the loop is the `rv_motion` of that scenario, with the
parameters after [`params_for`][pysmsim.drivers.loop_fas.params_for] puts
in the grid's kappa_0 and stress, so `mine` holds the same PGA as the
`table` of the loop above, to the last digit:

```python
assert (mine.pga == table.pga).all()
```

For large grids, `rv_loop(params, grid, workers=4)` shares the scenarios
among four processes (see [Speed](speed.md)).

## Matching SMSIM exactly

`fortran_compat=True` (on `rv_motion`, `rv_scenario`, `rv_loop` and
`rv_loop_details`) reproduces two of SMSIM's choices that pysmsim changes
by default: fup from kappa_0 rather than kappa at the magnitude, and the
DK80 peak factor from SMSIM's sum rather than an accurate integral; the
loops also make kappa independent of magnitude. Even so, RV results can
differ from SMSIM's by a few parts in 10,000, because SMSIM's spectral
moments are less accurate (see [Comparison with SMSIM](../smsim.md)).
