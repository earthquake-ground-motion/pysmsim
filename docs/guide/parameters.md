# Model parameters

All the model parameters of a run are one [`Params`][pysmsim.params.Params]
object: the content of one SMSIM `.params` file. It holds settings only,
never values derived from a magnitude or distance, so one object serves
every scenario.

## From a run file

```python
import dataclasses

import numpy as np

from pysmsim.io.config import load_config

config = load_config("examples/wna_bt15_example.toml")
params = config.params
grid = config.scenarios

params.source               # SingleCorner(stress=LogLinearStress(...), ...)
params.finite_fault         # TwoLinesSmoothed(c1=-1.72, ...)
params.site.kappa0          # 0.035
params.td.nsims             # 800
```

`params` has one attribute per section of the run file: `medium`, `source`
(with `source.stress`), `finite_fault`, `spreading`, `q`, `path_duration`,
`site`, `filters`, `rv` and `td`. Each is a small class whose fields are
the keys of its section (see [Run files](../run-files.md)).

## Changing parameters

The parameters are frozen dataclasses, so a change makes a new object with
[`dataclasses.replace`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace)
and leaves the old one as it was. This makes it safe to compare variants
side by side in a notebook:

```python
from pysmsim.spectrum.stress import LogLinearStress

high_kappa = dataclasses.replace(
    params, site=dataclasses.replace(params.site, kappa0=0.06)
)
low_stress = dataclasses.replace(
    params,
    source=dataclasses.replace(params.source, stress=LogLinearStress(stress_ref=50.0)),
)
quick_td = dataclasses.replace(
    params, td=dataclasses.replace(params.td, nsims=100, dt=0.01)
)
```

The loop drivers put a grid's kappa_0 or stress values into the parameters
with [`params_for`][pysmsim.drivers.loop_fas.params_for], which you can
also call yourself:

```python
from pysmsim.drivers.loop_fas import params_for

p100 = params_for(params, stress=100.0, kappa0=0.02)
p100.source.stress(6.0)     # 100 bars at every magnitude
p100.site.kappa(6.0)        # 0.02 s, now independent of magnitude
```

## Building parameters in Python

Every part can also be built directly, without a run file. This is the
example model of `examples/wna_bt15_example.toml`, written out in full:

```python
from pysmsim.params import Medium, Params, PathDuration, RVConfig, TDConfig
from pysmsim.spectrum.attenuation import PiecewiseQ
from pysmsim.spectrum.filters import Filters
from pysmsim.spectrum.finite_fault import TwoLinesSmoothed
from pysmsim.spectrum.site import Site
from pysmsim.spectrum.source import SingleCorner
from pysmsim.spectrum.spreading import Hinged, SpreadingSegment
from pysmsim.spectrum.stress import LogLinearStress

my_params = Params(
    title="WNA, built in Python",
    medium=Medium(density=2.72, shear_velocity=3.5),
    source=SingleCorner(stress=LogLinearStress(stress_ref=250.0)),
    finite_fault=TwoLinesSmoothed(c1=-1.72, c2=0.43, c3=-0.405, c4=0.235, delta_m=2.0),
    spreading=Hinged(
        segments=(SpreadingSegment(r_start=1.0, a=-1.0), SpreadingSegment(r_start=40.0, a=-0.5))
    ),
    q=PiecewiseQ(qr1=180.0, s1=0.45, qr2=180.0, s2=0.45),
    path_duration=PathDuration(
        distances=(0.0, 7.0, 45.0, 125.0, 175.0, 270.0),
        durations=(0.0, 2.4, 8.4, 10.9, 17.4, 34.2),
        slope_beyond=0.156,
    ),
    site=Site(
        amp_frequencies=params.site.amp_frequencies,   # the BJ97gr760 table
        amplifications=params.site.amplifications,
        kappa0=0.035,
    ),
    filters=Filters(),                                  # no filters
    rv=RVConfig(),                                      # BT15 WNA correction
    td=TDConfig(dt=0.005, tshift=60.0, f_tb2te=2.12),
)
```

Any field left out takes its default, which is the same as in a run file.
Invalid values raise a `ValueError` when the object is built.

## Looking at one factor

Each part of the model can be evaluated on its own, for example to plot it:

```python
f = np.logspace(-2, 2, 200)                  # Hz

terms = params.source.terms(6.0, params.medium.shear_velocity)
terms.m0, terms.fa, terms.stress             # dyne-cm, Hz, bars
shape = params.source.shape(f, terms)        # source shape S(f), 1 at f = 0

params.source.stress(7.0)                    # stress (bars) at M 7
params.finite_fault.h(6.0)                   # finite-fault factor h (km)
rps = params.finite_fault.rps(20.0, 6.0)     # point-source distance for Rrup 20 km
g = params.spreading.factor(f, rps=rps, rrup=20.0, magnitude=6.0)
p = params.q.attenuation(f, 6.0, rps=rps, rrup=20.0)   # exp(-pi f R / (c_q Q))
amp = params.site.amplification(f)
dim = params.site.diminution(f, 6.0)         # kappa and fmax filters
params.path_duration.duration(rps)           # path duration (s)
params.source.duration(terms)                # source duration (s)
```

[`spectrum_factors`][pysmsim.spectrum.fas.spectrum_factors] gives all of
them at once for a scenario (see [Fourier spectra](fourier.md#the-factors-of-the-spectrum)).

## From SMSIM files

[`read_params`][pysmsim.io.legacy.read_params] reads an SMSIM `.params`
file (dated 02/25/21) into a `Params`, with notes on where SMSIM would
behave differently:

```text
from pysmsim.io.legacy import read_params

params, notes = read_params("model.params")
```

`pysmsim convert` turns the same file into a run file (see
[Command line](../cli.md#converting-smsim-input-files)).

## Writing a run file from Python

```python
from pysmsim.io.config import RunConfig, dumps_config
from pysmsim.scenario import DistanceMetric, ScenarioGrid

my_grid = ScenarioGrid(
    magnitudes=(5.0, 6.0, 7.0),
    distances=(10.0, 30.0, 100.0),
    distance_metric=DistanceMetric.RRUP,
    periods=(0.1, 0.2, 0.5, 1.0, 2.0),
)
text = dumps_config(RunConfig(params=my_params, scenarios=my_grid))
print(text[:300])
```

Write `text` to a `.toml` file to run it with `pysmsim run`.
