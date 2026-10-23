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
| `rv/` | `tmrsk_loop_rv_drvr` | M 4-8 (step 1); Rjb 1, 5, 20, 50, 100, 300 km; PGV, PGA and PSA at 0.01, 0.1, 0.3, 1, 3, 10 s (5% damping) | 240 |
| `td/` | `tmrsk_loop_td_drvr` | M 4, 6, 8; Rjb 5, 20, 100 km; PGV, PGA and PSA at 0.01, 0.1, 0.2, 1, 2 s (5% damping); 800 simulations | 63 |

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
- **Reproducible.** Running `make_golden.sh` twice with the same build gives
  identical files.
