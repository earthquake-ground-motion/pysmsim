# Python guide

pysmsim is built in layers. Each layer is plain functions and small frozen
dataclasses, so you can stop at whichever level you need, for example in a
Jupyter notebook:

| Layer | Modules | What you call | SMSIM |
| --- | --- | --- | --- |
| Model | `pysmsim.params`, `pysmsim.io.config` | `load_config`, `Params(...)` | `.params` file |
| Spectrum factors | `pysmsim.spectrum.*` | `source.shape`, `spreading.factor`, `site.amplification`... | `spect_shape`, `gsprd`, `dimin`... |
| One spectrum | `pysmsim.spectrum.fas` | [`fourier_spectrum`][pysmsim.spectrum.fas.fourier_spectrum] | `spect_amp` |
| One RV motion | `pysmsim.rv` | [`rv_motion`][pysmsim.rv.gm_rv.rv_motion] | `gm_rv` |
| One TD scenario | `pysmsim.td` | [`td_motion`][pysmsim.td.gm_td.td_motion] | `gm_td` |
| One scenario in full | `pysmsim.drivers.single` | `rv_scenario`, `td_scenario`, `fas_scenario` | `gm_rv_drvr`, `gm_td_drvr`, `FAS_DRVR` |
| A grid of scenarios | `pysmsim.drivers` | `fas_loop`, `rv_loop`, `td_loop`, `run` | `*_loop_*_drvr` |

The pages of this guide go from the bottom up:

- [Model parameters](parameters.md): reading a run file, building the
  parameters in Python and changing them;
- [Fourier spectra](fourier.md): `fourier_spectrum` for one scenario, its
  factors, and the FAS loop;
- [Random vibration](rv.md): `rv_motion` for one scenario and response,
  your own loops, and `rv_loop`;
- [Time domain](td.md): `td_motion` for one scenario, the series and the
  steps inside it, and `td_loop`;
- [Runs and result files](runs.md): `run`, and writing and reading results;
- [Speed](speed.md): Numba and the `workers` option.

## Conventions

The functions take the model parameters first, then the magnitude and the
**rupture distance** Rrup in km:

```text
fourier_spectrum(f, params, magnitude, rrup)
rv_motion(params, magnitude, rrup, response)
td_motion(params, magnitude, rrup, periods)
```

The loops take a [`ScenarioGrid`][pysmsim.scenario.ScenarioGrid], whose
distances may be Joyner-Boore distances that the loop converts to Rrup.

Every result is a frozen dataclass that keeps the intermediate quantities
(durations, moments, peak factors, corner frequencies...), and every table
is a pandas DataFrame.

The snippets in this guide run as they are from the root folder of the
repository, one page at a time, in order (each page starts by loading the
example run file). The tests run them, so they stay up to date.

## Setting up a notebook

```bash
uv sync --extra notebook
uv run jupyter lab
```

A first cell that most notebooks need:

```python
import dataclasses

import numpy as np
import pandas as pd

from pysmsim.io.config import load_config

config = load_config("examples/wna_bt15_example.toml")
params = config.params        # the model
grid = config.scenarios       # the scenarios of the run file
```
