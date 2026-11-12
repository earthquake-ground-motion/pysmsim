# Command line

The `pysmsim` command has four subcommands. `python -m pysmsim` does the
same. `pysmsim --help` and `pysmsim COMMAND --help` list every option.

| Command | What it does | SMSIM equivalent |
| --- | --- | --- |
| `pysmsim run` | Computes every scenario of a run file and writes the results | `tmrsk_loop_rv_drvr`, `tmrsk_loop_td_drvr`, `fmrsk_loop_fas_drvr` |
| `pysmsim scenario` | Gives every detail of one magnitude and distance | `gm_rv_drvr`, `gm_td_drvr`, `FAS_DRVR` |
| `pysmsim check` | Validates run files without computing anything | |
| `pysmsim convert` | Turns SMSIM `.params` and `.ctl` files into a run file | |

The exit status is 0 on success, 1 when a run file is invalid or the run
fails, and 2 for wrong command-line arguments.

## Checking a run file

```bash
pysmsim check examples/wna_bt15_example.toml
```

prints a summary of the model and the scenarios, or names each key that is
wrong. Several files can be checked at once.

## Running a grid of scenarios

A run is described by one TOML file: the model parameters, the scenarios
(magnitudes, distances, periods, frequencies, kappa_0 and stress values)
and the output file (see [Run files](run-files.md)).
`examples/wna_bt15_example.toml` is SMSIM's own example run.

```bash
pysmsim run examples/wna_bt15_example.toml            # write the [output] file
pysmsim run examples/wna_bt15_example.toml -o results.h5
pysmsim run examples/wna_bt15_example.toml -o results.col --fortran-compat
pysmsim run examples/wna_bt15_example.toml --td -o results.h5   # also TD
pysmsim run examples/wna_bt15_example.toml --td --workers 0 -o results.h5
```

By default a run computes the random-vibration peak motions of every
scenario and, when the run file gives `[scenarios] frequencies`, the
Fourier acceleration spectra.

| Option | Effect |
| --- | --- |
| `-o FILE` | Write to FILE instead of the run file's `[output] path`. A relative path is relative to the current folder; a relative `[output] path` is relative to the run file's folder. |
| `-f FORMAT` | `csv`, `parquet`, `hdf5` or `col`. Without it, the format follows the suffix of `-o` (`.csv`, `.parquet`, `.h5` or `.hdf5`, `.col`), else `[output] format`. |
| `--td` | Also simulate `[td] nsims` time series of every scenario (800 in the example), which takes much longer than the RV method. |
| `--no-rv` | Leave the RV method out. |
| `--td-series [NSIMS]` | Also write the acceleration series to the HDF5 output (all simulations, or the first NSIMS of each scenario). |
| `--series-tolerance F` | With `--td-series`, store each series only from where it first rises above F times its peak to where it last falls below it. |
| `--series-precision 64` | With `--td-series`, store 64-bit instead of 32-bit floats. |
| `-w N`, `--workers N` | Share the scenarios among N processes (0: one per CPU). The results are the same as with one; see [Speed](guide/speed.md). |
| `--fortran-compat` | Reproduce SMSIM's behaviour where pysmsim deliberately differs from it (see [Comparison with SMSIM](smsim.md)). |
| `-q` | Print nothing but errors. |

### Output formats

| Format | Contents |
| --- | --- |
| `csv`, `parquet` | One row per scenario: the scenario columns, then `pgv` (cm/s), `pga` and `psa(T)` (cm/s²), the TD means with the prefix `td_` (`td_pga`, `td_pgd` in cm, `td_arias` in cm/s, `td_dur_75_05` in s, `td_psa(T)`...) and `fas(f)` (cm/s) |
| `hdf5` | The scenario, FAS, RV and TD tables in one file, with the arithmetic and geometric means and spreads of the TD simulations and the run file stored in it |
| `col` | SMSIM's `.col` layout of the RV results, one row per scenario and measure, for comparison with the Fortran |

The column `psa(T)` names the period T in seconds with up to 6 significant
digits, for example `psa(0.1)` and `psa(1)`, and `fas(f)` the frequency in
Hz in the same way.

### Time series

`--td-series` writes the acceleration series to the HDF5 file, in
`/td/series`: `times` (s, from 0) and `acceleration`, an array of shape
(simulations, time steps, scenarios). The files are large: 800 simulations
of 10,000 points take 32 MB per scenario in 32-bit floats.
`--series-tolerance 1e-4` stores each series only between the points where
it first rises above and last falls below 1e-4 of its peak, about a third
of the size; the rest is NaN and takes no space.
[`read_td_series`][pysmsim.io.hdf5.read_td_series] reads them back (see
[Runs and result files](guide/runs.md#time-series-files)).

## One scenario in detail

`pysmsim scenario` computes one magnitude and distance with the model,
oscillator periods, damping and frequencies of a run file, and prints every
intermediate quantity:

```bash
pysmsim scenario examples/wna_bt15_example.toml -m 6 -r 20
pysmsim scenario examples/wna_bt15_example.toml -m 6 --rjb 15 --td -o m6_rjb15
pysmsim scenario examples/wna_bt15_example.toml -m 6 -r 20 --td --series 1 -o m6
```

The distance is the rupture distance (`-r`), or the Joyner-Boore distance
(`--rjb`), converted to Rrup as the run file's `[scenarios]` says. The
magnitudes, distances, kappa_0 and stress lists of `[scenarios]` are not
used.

- The RV results are PGD, PGV, PGA, the Arias intensity and PSA at each
  period, with their spectral moments, durations and peak factors.
- `--td` adds the means, standard errors, geometric means and factors of
  `[td] nsims` simulations (or `--nsims N`).
- The Fourier spectra and their factors (source, spreading, Q, site,
  filters, and the oscillator responses) come when the run file gives
  `[scenarios] frequencies`.

With `-o DIR`, the tables are also written as CSV files in DIR: `rv.csv`,
`td.csv`, `fas.csv` and, with `--series K`, `td_series.csv`: the time (s),
acceleration (cm/s²), velocity (cm/s) and displacement (cm) of simulation
K, counted from 1.

## Converting SMSIM input files

`pysmsim convert` turns an SMSIM parameter file (the format of SMSIM
2023-03-03, dated 02/25/21) and, optionally, a `tmrsk_loop_rv_drvr` or
`tmrsk_loop_td_drvr` control file into a run file:

```bash
pysmsim convert model.params tmrsk_loop_rv_drvr.ctl -o run.toml
pysmsim convert model.params > run.toml   # example scenarios to edit
```

Where SMSIM would behave differently from pysmsim with these inputs (for
example the stress model of sources 2, 11 and 12, or a low-cut `nslope`
that SMSIM rounds), the converter prints a note and writes it at the top
of the run file. An existing `-o` file is replaced only with `--force`.
