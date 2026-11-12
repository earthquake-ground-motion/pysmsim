# Golden files

Reference outputs from the Fortran oracle (see `oracle/README.md`), for
comparing the Python port with SMSIM. Python tests read these files and never
need a Fortran compiler.

All files here come from the double-precision build (`double/`), which is
the main reference for the Python port. `double/BUILD_INFO.txt` records the
compiler, flags and patches. To regenerate them:

```bash
oracle/build.sh /path/to/smsim.2023-03-03_version
oracle/make_golden.sh /path/to/smsim.2023-03-03_version double
```

The same command with `single` writes a single-precision set to
`tests/golden/single/`; it is not committed.

## Contents

Each folder holds one file per shipped `.params` file, named after it. The
six params files cover source model 1, geometrical-spreading options 0 and 7,
the standard and the Stafford et al. (`fr1 < 0`) Q models, and the BT15
oscillator corrections for WNA and ENA. The control files in
`oracle/inputs/` take kappa_0 and the stress from the params file.

| Folder | Program | Grid | Rows per file |
| --- | --- | --- | --- |
| `factors/` | `factor_dump` | M 4, 6, 8; Rrup 5, 20, 100 km; 41 frequencies, 0.01-100 Hz | 41 (one file per M and Rrup) |
| `fas/` | `fmrsk_loop_fas_drvr` | M 4-8 (step 1); Rjb 1, 5, 20, 50, 100, 300 km; 21 frequencies, 0.01-100 Hz | 630 |
| `path/` | `factor_dump` | 17 synthetic path cases (below); M 3, 5, 7; Rrup 0.5, 5, 30, 55, 200 km; 21 frequencies, 0.01-100 Hz | 21 (one file per case, M and Rrup) |
| `response/` | `response_dump` | M 5, 7; Rrup 10 km; 41 frequencies, 0.01-100 Hz; ground displacement, velocity and acceleration, oscillators at 0.01, 0.1, 1 and 10 s (5% damping), Wood-Anderson and WWSSN-SP | 41 (one file per params file and M) |
| `rv/` | `tmrsk_loop_rv_drvr` | M 4-8 (step 1); Rjb 1, 5, 20, 50, 100, 300 km; PGV, PGA and PSA at 0.01, 0.1, 0.3, 1, 3, 10 s (5% damping) | 240 |
| `rv_cases/` | `tmrsk_loop_rv_drvr` | 12 synthetic RV cases (below), on the grid of `rv/` with the damping of the case | 240 (one file per case) |
| `example/` | `tmrsk_loop_rv_drvr` | The shipped example: 200 periods from 0.01 to 100 s, PGV and PGA; M 4, 6, 8; Rjb 10 km; kappa_0 0.02 and 0.05 s; 100 bars. `shipped.col` is the distribution's file, `double.col` the same run with the double build | 1212 |
| `source/` | `factor_dump` | 18 synthetic source cases (below); M 2, 3, 4.5, 5, 6, 8; Rrup 20 km; 21 frequencies, 0.01-100 Hz | 21 (one file per case and M) |
| `site/` | `factor_dump` | 10 synthetic site and filter cases (below); M 4, 6, 8; Rrup 20 km; 41 frequencies, 0.01-100 Hz | 41 (one file per case and M) |
| `site_lines/` | none | The crustal-amplification, diminution, low-cut and high-cut lines of each shipped `.params` file, copied without comments | (one file per params file) |
| `td/` | `tmrsk_loop_td_drvr` | M 4, 6, 8; Rjb 5, 20, 100 km; PGV, PGA and PSA at 0.01, 0.1, 0.2, 1, 2 s (5% damping); 800 simulations | 63 |
| `scenario/rv/` | `gm_rv_drvr` | M 4, Rrup 10 km; M 6, Rrup 20 km; M 7.5, Rrup 100 km; PSA at 0.01, 0.03, 0.1, 0.3, 1, 3, 10 s (5% damping). `.col` is the response spectrum, `.sum` the summary (PGD, PGV, PGA with their moments, the Arias intensity, the moments of each period) without its dates and times | 7 (one `.col` and one `.sum` per params file, M and Rrup) |
| `scenario/fas/` | `FAS_DRVR` | The scenarios of `scenario/rv/`; 41 frequencies, 0.01-100 Hz; oscillator spectra at 0.1, 0.3, 1, 3 s (5% damping) | 41 (one file per params file, M and Rrup) |
| `scenario/td/` | `gm_td_drvr` | M 6, Rrup 20 km; the periods of `scenario/rv/`; 800 simulations. `.col` and `.sum` as for `scenario/rv/`; for `ofr.params` also `.avd.col.gz`, the time, acceleration, velocity and displacement of simulation 1 | 7 (one `.col` and one `.sum` per params file) |
| `td_series/` | `td_dump` | One simulation with its white noise: 6 synthetic window and timing cases (below) at M 5, Rrup 10 km and M 7, Rrup 50 km, and `ofr.params` at M 6, Rrup 20 km; gzipped | one per point of the series (one file per case, M and Rrup) |

The `source/` cases cover the source models and stress models that the
shipped params files do not use. Each is `ofr.params` with the source-shape,
stress and spectral-scaling lines replaced, as listed in
`oracle/inputs/source_cases.txt`. Cases with `quirk` in the name show SMSIM
behaviour that pysmsim changes by default; the tests compare them with
`fortran_compat=True`.

The `path/` cases do the same for the finite-fault factor (forms 1, 2 and
999, Stafford's form with Rps in quadrature), geometrical spreading
(options 1-6, magnitude-dependent hinged slopes, r_ref = 2) and Q (constant,
three-part, and Stafford's Q with a finite-fault factor so that Rps and Rrup
differ). Each is `ofr.params` with the finite-fault, spreading and Q lines
replaced, as listed in `oracle/inputs/path_cases.txt`. The tests compare the
AB14 spreading cases (options 4 and 5) with `fortran_compat=True`.

The `site/` cases cover the crustal amplification (a one-point table and
five-point tables), kappa that depends on magnitude, fmax without kappa, the
low-cut filter with nslope 4, 6 and 12, and both high-cut shapes with
several powers. Each is `ofr.params` with the site and filter lines
replaced, as listed in `oracle/inputs/site_cases.txt`.
`s_lowcut_quirk_nslope6` shows SMSIM rounding nslope 6 down to 4.

The `rv_cases/` cover the oscillator corrections the shipped files do not
use (BJ84, LP99 and the average of the two BT15 tables), the BT15 tables
with the `ofr.params` spectrum, 2% and 10% damping, fup set by kappa or
fmax alone, and other `zup` and `amp_cutoff` values. Each is `ofr.params`
with the diminution and RV lines replaced, as listed in
`oracle/inputs/rv_cases.txt`. `rv_quirk_kappa_mdep` shows the loop driver
ignoring `dkappadmag`.

SMSIM's spectral moments carry its integration error (a few 1e-4), which
narrow-band oscillators amplify, so a few RV rows differ from pysmsim by
more than 1e-3; `test_rv_golden.py` explains the tolerances.

The `td_series/` files let the tests feed SMSIM's own noise to pysmsim, so
that one time-domain series can be compared point by point. Each starts with
`! key = value` lines: the scenario, the source and path durations, the
time grid (`nstart` and `nstop` are 1-based, `npw2` is the number of
points), the window's `te`, PGA, PGV, PGD, the integral of the squared
acceleration, the indices at 5, 75 and 95% of it, and the oscillator
displacement `rd` at 0.01, 0.1, 0.2, 1 and 2 s (5% damping). Then each row
gives the noise and the acceleration (cm/s^2) at one point; the noise is 0
outside the window. The cases (`oracle/inputs/td_cases.txt`) are
`ofr.params` with the low-cut, window and timing lines replaced: the
exponential window with Gaussian and uniform noise and with a longer
window, the box window with and without tapers, and a low-cut filter whose
pad is longer than `tshift`.

The `scenario/` programs are interactive; `make_golden.sh` feeds them their
answers on stdin. They take Rrup directly. SMSIM leaves the rupture distance
that the Stafford (2022) spreading and Q use unset in these three programs
(the loop programs set it), so they are built with a patch that sets it
(`oracle/patches/*.rrup.patch`); without it, the `stafford.optimal_model`
results are wrong. Their output has 3 to 6 significant digits, which sets
the tolerances of `test_scenario_golden.py`.

The loop programs convert Rjb to Rrup with the CY14 Ztor relation for a
strike-slip fault (case 2 in the control files).

## Things to know when using them

- **Precision of the inputs.** The `.col` files print the frequency and
  distances with few digits (for example `6.3096E+00` for 6.309573 Hz).
  Recomputing a row from its printed inputs therefore differs from the
  printed result by up to about 1e-4 at high frequencies. Use `factors/`,
  which prints everything with 16 digits, for checks tighter than that.
- **Sentinels.** `-9.999E+02` and `-999.00` mark values that are not
  computed. In `rv/` these are the PSA columns of the PGV and PGA rows, and
  in `wna.scf.xfff...` also the rows at M 7 and 8 and Rjb 1 km, where Rps
  falls outside the range of the BT15 oscillator-correction coefficients.
- **Time-domain results are random.** The `td/` values are averages over 800
  simulations with SMSIM's own random-number generator, which Python does not
  reproduce. Compare means within a few standard errors. For the PSA rows
  the `sd_am_sdev` column gives the standard deviation of the 800 simulated
  oscillator displacements (divide by the square root of 800 for the
  standard error of the mean); it is a sentinel in the PGV and PGA rows.
  The `td_series/` files remove the randomness for one simulation: given
  their noise, pysmsim gives the same series to about 1e-10.
- **Reproducible.** Running `make_golden.sh` twice with the same build gives
  identical files.
