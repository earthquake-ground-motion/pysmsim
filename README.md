# pysmsim

Pure Python Library Implementing the SMSIM Methodology of (Boore, 2003)

pysmsim is a Python port of David Boore's Fortran SMSIM programs (version
2023-03-03). Release 0.1 covers the random-vibration (RV) method: peak
ground velocity and acceleration and pseudo-spectral acceleration for grids
of scenarios, as SMSIM's `tmrsk_loop_rv_drvr` computes them, together with
Fourier amplitude spectra. The time-domain method follows in a later
release.

## Installation

```bash
pip install -e .            # or: uv sync
pip install -e ".[dev]"     # with the test and lint tools
```

pysmsim needs Python 3.12 or later.

## Running from the command line

A run is described by one TOML file: the model parameters, the scenarios
(magnitudes, distances, periods, frequencies, kappa_0 and stress values)
and the output file. `examples/wna_bt15_example.toml` is SMSIM's own
example run.

```bash
pysmsim check examples/wna_bt15_example.toml          # validate and summarize
pysmsim run examples/wna_bt15_example.toml            # write the [output] file
pysmsim run examples/wna_bt15_example.toml -o results.h5
pysmsim run examples/wna_bt15_example.toml -o results.col --fortran-compat
```

The output format follows the suffix given with `-o`, or `--format`:

| Format | Contents |
| --- | --- |
| `csv`, `parquet` | One row per scenario: the scenario columns, then `pgv` (cm/s), `pga` and `psa(T)` (cm/s²) and `fas(f)` (cm/s) |
| `hdf5` | The scenario, FAS and RV tables in one file, with the run file stored in it |
| `col` | SMSIM's `.col` layout, one row per scenario and measure, for comparison with the Fortran |

`--fortran-compat` reproduces SMSIM's behaviour where pysmsim deliberately
differs from it (see `CLAUDE.md`). `pysmsim --help` and
`pysmsim run --help` list every option.

## Using it from Python

```python
from pysmsim.io.config import load_config
from pysmsim.drivers.run import run
from pysmsim.io.hdf5 import write_hdf5

config = load_config("examples/wna_bt15_example.toml")
result = run(config.params, config.scenarios)
result.rv[["pgv", "pga"]]      # psa(T) columns follow
write_hdf5(result, "results.h5")
```

Every step (source spectrum, path terms, spectral moments, peak factors)
is a public function that can be called on its own; see the docstrings.

## Licence

GNU General Public License v3.0; see `LICENSE`.
