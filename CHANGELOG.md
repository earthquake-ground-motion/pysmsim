# Changelog

## Unreleased

- The time-domain method of SMSIM (`pysmsim.td`): windowed white noise
  shaped by the Fourier spectrum, with PGA, PGV, PGD, Arias intensity, the
  5-75% and 5-95% significant durations and oscillator PSA (an exact
  recurrence, compiled with Numba) for each simulation, and their
  arithmetic and geometric means and spreads (`td_motion`).
- Time-domain loops over scenarios (`pysmsim.drivers.loop_td`), and
  `run(..., td=True)` / `pysmsim run --td`. The means go in the CSV and
  Parquet tables with the prefix `td_`, and the HDF5 file gets a `/td`
  group with all four statistics of every measure. `--td-series [NSIMS]`
  (`pysmsim.io.hdf5.write_td_series`) also stores the acceleration series
  in the HDF5 file (32-bit by default, optionally trimmed where they are
  below a fraction of their peak), and `TimeGrid.times()` gives their time
  axis.
- `pysmsim convert PARAMS [CTL] [-o run.toml]` (`pysmsim.io.legacy`)
  converts an SMSIM parameter file and, optionally, a control file of
  `tmrsk_loop_rv_drvr` or `tmrsk_loop_td_drvr` into a TOML run file. Notes
  on where SMSIM would behave differently from pysmsim are printed and
  written at the top of the run file.

Given SMSIM's own white noise, a pysmsim time series agrees with the
double-precision Fortran build to about 1e-10 of its peak, for the
exponential and box windows, both noise distributions and a low-cut pad.
SMSIM's random numbers are not reproduced, so the means of 800 simulations
are compared statistically: they agree within 4 standard errors (all but
3% within 3) and within 5% for every scenario and measure.

## 0.1.0

First release: the random-vibration method of SMSIM 2023-03-03.

- TOML run files (`pysmsim.io.config`) for the model parameters, the
  scenario grid and the output.
- The Fourier amplitude spectrum with all of SMSIM's source, stress,
  finite-fault, spreading, Q, site and filter options (`pysmsim.spectrum`).
- Random-vibration PGV, PGA and PSA (`pysmsim.rv`): spectral moments,
  Der Kiureghian (1980) and Cartwright and Longuet-Higgins (1956) peak
  factors, and the BJ84, LP99 and BT15 oscillator corrections.
- Loops over scenarios (`pysmsim.drivers`) and combined FAS and RV runs,
  written as CSV, Parquet, HDF5 or SMSIM's `.col` layout.
- The `pysmsim run` and `pysmsim check` commands. A run may have at most
  1,000 periods and 1,000 frequencies.
- `fortran_compat` options that reproduce SMSIM's behaviour where pysmsim
  deliberately differs from it.

Results agree with the double-precision Fortran build to within 1e-3 for
the shipped example, and for all but a few narrow-band oscillator rows of
the shipped parameter files, where SMSIM's own integration of the spectral
moments is the less accurate (up to 4.3e-3).
