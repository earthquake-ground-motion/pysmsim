# Runs and result files

[`run`][pysmsim.drivers.run.run] computes everything a run file asks for
(what `pysmsim run` does): the Fourier spectra, the random-vibration peak
motions and, when asked, the time-domain simulations of every scenario of
a grid. It returns tables that share their rows, one per scenario.

## `run`

```python
import dataclasses

import numpy as np

from pysmsim.drivers.run import run
from pysmsim.io.config import load_config

config = load_config("examples/wna_bt15_example.toml")
params, grid = config.params, config.scenarios

result = run(params, grid)
result.scenarios     # magnitude, distances, kappa_0, stress, corner frequencies...
result.fas           # fas(f) for each of grid.frequencies (cm/s)
result.rv            # fup, durations, pgv, pga and psa(T) for each period
```

The FAS frequencies are `grid.frequencies` unless others are given, and
`rv=False` leaves the RV method out. With `td=True`, `run` also simulates
`params.td.nsims` series of every scenario:

```python
quick = dataclasses.replace(params, td=dataclasses.replace(params.td, nsims=100))
result = run(quick, grid, np.logspace(-1, 1, 5), td=True)
result.td            # means of the simulations: pga, pgv, pgd, arias, psa(T)...
result.td_details    # one row per scenario and measure, with the spreads
```

The tables share their index (the scenario number, from 0), so they join
into one table:

```python
table = result.scenarios.join(result.rv).join(result.td.add_prefix("td_"))
```

[`results_table`][pysmsim.io.results.results_table] gives this table, with
the FAS columns too, as the CSV and Parquet files hold it.

## Writing and reading results

```python
from pysmsim.io.hdf5 import read_hdf5, write_hdf5
from pysmsim.io.results import write_results

write_results(result, "results.csv", "csv")       # or "parquet", "col", "hdf5"
write_hdf5(result, "results.h5")                  # all the tables and the run file
again = read_hdf5("results.h5")                   # a RunResult again
again.rv.equals(result.rv)
```

| Format | Contents |
| --- | --- |
| `csv`, `parquet` | One row per scenario: the scenario columns, the RV motions, the TD means with the prefix `td_` and `fas(f)` |
| `hdf5` | The scenario, FAS, RV and TD tables (with all the TD statistics) and the run file itself, so that a result file says how it was made |
| `col` | SMSIM's `.col` layout of the RV results, for comparison with the Fortran |

pandas reads the CSV and Parquet files directly, for example
`pd.read_csv("results.csv")`.

## Time-series files

[`write_td_series`][pysmsim.io.hdf5.write_td_series] simulates the series
of a grid again, one scenario at a time (the same simulations as
`run(..., td=True)`), and adds them to an HDF5 file, so that the file can
be larger than memory.
[`read_td_series`][pysmsim.io.hdf5.read_td_series] reads them back, all
scenarios or some:

```python
from pysmsim.io.hdf5 import read_td_series, write_td_series

write_td_series("results.h5", quick, grid, nsims=5, tolerance=1e-4)
series = read_td_series("results.h5", scenarios=[0, 1])
series.times                 # s, from 0
series.acceleration.shape    # (5 simulations, time steps, 2 scenarios), cm/s^2
series.nstart, series.nstop  # where each scenario's motion starts and stops
```

With a `tolerance`, the start and end of each series below that fraction
of its peak are not stored and read back as NaN (about a third of the size
for 1e-4). The series are stored as 32-bit floats unless
`dtype=np.float64` is given.

## Matching SMSIM

`fortran_compat=True` reproduces SMSIM's conventions where pysmsim
deliberately differs from them (see [Comparison with SMSIM](../smsim.md)).
`write_results(result, "results.col", "col")` with a run made this way
gives a file to compare line by line with SMSIM's `tmrsk_loop_rv_drvr`
output.
