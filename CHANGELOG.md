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

- Single-scenario drivers (`pysmsim.drivers.single`), the equivalents of
  SMSIM's `gm_rv_drvr`, `gm_td_drvr` and `FAS_DRVR`: `rv_scenario` (PGD,
  PGV, PGA, Arias intensity and PSA by random vibration, with their
  moments, durations and peak factors), `td_scenario` (the averages of the
  time-domain simulations, and the acceleration, velocity and displacement
  of any one of them) and `fas_scenario` (the Fourier spectra of ground and
  oscillator motion with each factor of the spectrum), and the command
  `pysmsim scenario RUN.toml -m M (-r RRUP | --rjb RJB) [--td] [-o DIR]`.
  They agree with the three SMSIM programs to the digits those print, once
  an SMSIM bug is patched in the oracle: the programs leave the rupture
  distance of the Stafford (2022) spreading and Q unset.
- Speed. The time-domain measures of each series (PGA, PGV, PGD, Arias
  intensity and the durations) come from one compiled pass
  (`pysmsim.td.intensity.series_measures`), and the oscillator kernel
  shares its work over simulations and periods together, which halves the
  time of a TD run with few periods. `workers=N` on `rv_loop`,
  `rv_loop_details`, `td_loop`, `td_loop_details` and `run`, and
  `pysmsim run --workers N` (0: one per CPU), share the scenarios among N
  processes; the results are identical to one process. It pays off for
  the RV method on large grids (3.4 times faster with 4 processes on 4
  cores for 340 scenarios and 200 periods); the TD method already uses
  every core through Numba's threads. `examples/benchmark.py` times the
  FAS, RV and TD loops of a run file.

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
