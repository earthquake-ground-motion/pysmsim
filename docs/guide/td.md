# Time domain

The time-domain (TD) method simulates acceleration series (Boore, 2003):
white noise is windowed, transformed to the frequency domain, its spectrum
is normalised and multiplied by the Fourier acceleration spectrum of the
scenario, and the result is transformed back. Each simulation gives PGA,
PGV, PGD, the Arias intensity, the 5-75% and 5-95% significant durations
and the oscillator responses; SMSIM reports their averages over the
simulations.

The noise comes from NumPy's random-number generator, seeded with
`[td] seed`, so a run gives the same results every time. As in SMSIM, the
generator starts again from the seed for every scenario. SMSIM's own random
numbers are not reproduced, so the simulations differ from SMSIM's; their
averages agree within the scatter of the simulations.

## One scenario: `td_motion`

[`td_motion`][pysmsim.td.gm_td.td_motion] simulates one scenario: the
parameters, the magnitude, the **rupture distance** Rrup (km) and the
oscillator periods.

```python
import dataclasses

import numpy as np

from pysmsim.io.config import load_config
from pysmsim.td.gm_td import td_motion

config = load_config("examples/wna_bt15_example.toml")
params = config.params

sims = td_motion(params, 6.0, 20.0, periods=[0.1, 0.2, 0.5, 1.0, 2.0], nsims=100)

sims.nsims                  # 100
sims.pga                    # PGA of each simulation (cm/s^2), shape (100,)
sims.pgv, sims.pgd          # cm/s, cm
sims.arias                  # Arias intensity (cm/s)
sims.dur_75_05, sims.dur_95_05   # significant durations (s)
sims.psa                    # PSA (cm/s^2), shape (100, 5): one row per simulation
sims.sd, sims.psv           # oscillator displacement (cm) and pseudo-velocity (cm/s)
```

`nsims` defaults to `params.td.nsims` (800 in the example). Without
periods, no oscillator responses are computed.

The averages SMSIM reports are given by
[`statistics`][pysmsim.td.gm_td.TDResult.statistics]: the arithmetic mean
and standard deviation, the geometric mean, and the factor
10^std(log10) of one geometric standard deviation:

```python
stats = sims.statistics("pga")
stats.mean, stats.std, stats.gmean, stats.gfactor
psa_stats = sims.statistics("psa")          # a list, one per period
[s.gmean for s in psa_stats]
```

The result also holds the scenario's terms, durations and time grid:

```python
sims.terms.rps, sims.terms.source.fa
sims.dur_source, sims.dur_path      # s
sims.grid.dt, sims.grid.npts        # time step (s) and number of points
sims.grid.nstart, sims.grid.nstop   # where the motion starts and stops (indices)
```

## The time series

With `keep_series=True` the result keeps the acceleration series, one row
per simulation (memory: nsims × npts values):

```python
from pysmsim.td.intensity import velocity_displacement

sims = td_motion(params, 6.0, 20.0, nsims=10, keep_series=True)
t = sims.grid.times()               # s, from 0
acc = sims.acc[0]                   # cm/s^2, the first simulation
vel, dis = velocity_displacement(acc, sims.grid.dt)   # cm/s, cm
assert np.abs(acc).max() == sims.pga[0]
```

To plot them, for example with matplotlib:

```text
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, sharex=True)
for ax, y, label in zip(axes, (acc, vel, dis), ("acc (cm/s²)", "vel (cm/s)", "dis (cm)")):
    ax.plot(t, y)
    ax.set_ylabel(label)
axes[-1].set_xlabel("Time (s)")
```

## Inside `td_motion`

Every step is a public function, so a simulation can be taken apart, for
example to change the window or look at the noise. These lines repeat what
`td_motion` does for ten simulations, and give the same series:

```python
from pysmsim.spectrum.fas import scenario_terms
from pysmsim.td.acc_ts import acc_ts, target_spectrum
from pysmsim.td.intensity import series_measures
from pysmsim.td.noise import white_noise
from pysmsim.td.response import oscillator_displacement
from pysmsim.td.window import time_grid, window

td = params.td
terms = scenario_terms(params, 6.0, 20.0)
dur_source = params.source.duration(terms.source)            # s
dur_path = params.path_duration.duration(terms.rps)          # s
grid = time_grid(td, params.filters.low_cut, dur_source, dur_path)
w = window(grid, td)                                         # npts values
spectrum = target_spectrum(grid, params, 6.0, 20.0)          # FAS (cm/s) on the FFT frequencies

rng = np.random.default_rng(td.seed)
noise = white_noise(rng, (10, grid.nmotion), td.noise)       # one row per simulation
acc = acc_ts(noise, grid, w, spectrum)                       # cm/s^2, shape (10, npts)
assert np.array_equal(acc, sims.acc)

measures = series_measures(acc, grid.dt)    # pga, pgv, pgd, arias, dur_75_05, dur_95_05
sd = oscillator_displacement(acc, [0.1, 1.0], 0.05, grid.dt)   # cm, shape (10, 2)
```

| Step | Function | SMSIM |
| --- | --- | --- |
| Length of the series, start and end of the motion | [`time_grid`][pysmsim.td.window.time_grid] | `get_npts_for_td_calcs` |
| Box or exponential window | [`window`][pysmsim.td.window.window] | `wind_box`, `wind_exp` |
| Gaussian or uniform noise | [`white_noise`][pysmsim.td.noise.white_noise] | `gasdev`, `ran1` |
| Target spectrum on the FFT frequencies | [`target_spectrum`][pysmsim.td.acc_ts.target_spectrum] | `spect_amp` |
| Shaping the noise | [`acc_ts`][pysmsim.td.acc_ts.acc_ts] | `acc_ts` |
| Velocity and displacement | [`velocity_displacement`][pysmsim.td.intensity.velocity_displacement] | `acc2vd` |
| Peaks, Arias intensity, durations | [`series_measures`][pysmsim.td.intensity.series_measures] | `accsqint`, `gm_td` |
| Oscillator response | [`oscillator_displacement`][pysmsim.td.response.oscillator_displacement] | `rdrvaa` |

## One scenario in full: `td_scenario`

[`td_scenario`][pysmsim.drivers.single.td_scenario] (SMSIM's
`gm_td_drvr`) gives the averages of every measure as one table, and the
acceleration, velocity and displacement of any one simulation:

```python
from pysmsim.drivers.single import td_scenario

td6 = td_scenario(params, 6.0, 20.0, periods=[0.1, 1.0], nsims=50, keep_series=True)
td6.table()                # mean, std, sem, gmean, gfactor... of each measure
td6.series(0)              # time, acc, vel, dis of the first simulation
```

## Your own loop

```python
import pandas as pd

quick = dataclasses.replace(params, td=dataclasses.replace(params.td, nsims=100))
rows = []
for magnitude in (5.0, 6.0, 7.0):
    for rrup in (10.0, 30.0, 100.0):
        r = td_motion(quick, magnitude, rrup, periods=[1.0])
        rows.append({
            "magnitude": magnitude,
            "rrup": rrup,
            "pga": r.statistics("pga").mean,
            "psa(1)": r.statistics("psa")[0].mean,
        })
table = pd.DataFrame(rows)
```

Because every scenario starts from the same seed, each row is the same
whichever order the scenarios are computed in, and the same as in the
loop drivers.

## The loop over a grid: `td_loop`

[`td_loop`][pysmsim.drivers.loop_td.td_loop] (SMSIM's
`tmrsk_loop_td_drvr`) simulates every scenario of a
[`ScenarioGrid`][pysmsim.scenario.ScenarioGrid] and returns one row per
scenario with the arithmetic means of the simulations, as SMSIM reports
them:

```python
from pysmsim.drivers.loop_td import td_loop, td_loop_details

grid = config.scenarios        # 6 scenarios, 200 periods
wide = td_loop(quick, grid)    # 100 simulations of each scenario
wide[["magnitude", "rjb", "kappa0", "npts", "pga", "pgv", "pgd", "arias", "dur_75_05"]]
```

[`td_loop_details`][pysmsim.drivers.loop_td.td_loop_details] gives one row
per scenario and measure with all four statistics (`mean`, `std`, `gmean`,
`gfactor`):

```python
long = td_loop_details(quick, grid)
long[long.measure == "pga"][["magnitude", "kappa0", "mean", "std", "gmean", "gfactor"]]
```

A grid built in Python gives the rows of the loop above:

```python
from pysmsim.scenario import DistanceMetric, ScenarioGrid

my_grid = ScenarioGrid(
    magnitudes=(5.0, 6.0, 7.0),
    distances=(10.0, 30.0, 100.0),
    distance_metric=DistanceMetric.RRUP,
    periods=(1.0,),
)
mine = td_loop(quick, my_grid)
assert (mine.pga == table.pga).all()
```

Two more functions help with large runs:

- [`td_time_grids`][pysmsim.drivers.loop_td.td_time_grids] gives the time
  grid of every scenario without simulating, to check the series lengths
  (and so the time and memory a run needs) first;
- [`td_loop_series`][pysmsim.drivers.loop_td.td_loop_series] yields the
  simulations of one scenario at a time with their series, so that a large
  grid never holds every series in memory.

```python
from pysmsim.drivers.loop_td import td_loop_series, td_time_grids

[g.npts for g in td_time_grids(quick, grid)]
for result in td_loop_series(quick, my_grid, nsims=5):
    result.acc.shape          # (5, npts) for each scenario in turn
```

## Speed

A TD scenario takes about 0.3 s for 800 simulations and a few periods, and
more with many periods. The compiled kernels already use every core.
`dt`, `nsims` and the number of periods set the time; see
[Speed](speed.md).
