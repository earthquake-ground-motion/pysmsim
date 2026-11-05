# Fortran oracle

Scripts that build Boore's SMSIM Fortran programs and run them, so the Python
results can be compared with the original code. The SMSIM distribution is
never modified: it is copied into `oracle/build/` (ignored by Git) and
patched there.

## Requirements

- gfortran (Homebrew `gcc` provides it on macOS)
- `bash`, `awk`, `sed`, `patch`, `sort`

## Usage

```bash
oracle/build.sh /path/to/smsim.2023-03-03_version
oracle/run_example.sh /path/to/smsim.2023-03-03_version single
oracle/run_example.sh /path/to/smsim.2023-03-03_version double
oracle/dump_factors.sh /path/to/smsim.2023-03-03_version double
oracle/make_golden.sh /path/to/smsim.2023-03-03_version double
```

`build.sh` compiles `tmrsk_loop_rv_drvr`, `fmrsk_loop_fas_drvr`,
`tmrsk_loop_td_drvr` and the test-only `factor_dump` (see below) twice:

- `oracle/build/bin/single/`: the code as written, which is mostly single
  precision;
- `oracle/build/bin/double/`: the same code with every REAL promoted to 8
  bytes (`-fdefault-real-8 -fdefault-double-8`).

Each folder gets a `BUILD_INFO.txt` that records the compiler, the flags and
the patches used. Override the compiler with `FC` and the flags with `FFLAGS`.

`run_example.sh` runs the shipped `example.tmrsk_loop_rv_drvr` with the chosen
build (`single` by default) and compares the output with the reference `.col`
file. The example inputs need three adjustments for the 2023 program; the
script makes them in its run folder, and its header explains each one.

## Spectral factors (`factor_dump`)

`programs/factor_dump.for` is a test-only program written for pysmsim; it is
not part of SMSIM. It reads a `.params` file, sets up one magnitude and
rupture distance the same way as `fmrsk_loop_fas_drvr`, and prints every
factor of the Fourier spectrum computed by `spect_amp` on a log-spaced
frequency grid. It calls the unmodified SMSIM subroutines, which it includes
from `smsim_subs_for_rv_programs.for` and `smsim_util_subs.for`.

```bash
oracle/build/bin/double/factor_dump PARAMS M RRUP OUTFILE [NFREQ FMIN FMAX]
```

The default grid is 201 points from 0.01 to 100 Hz. The output starts with
comment lines giving the scenario and the frequency-independent terms (M0,
stress, fa, fb, Rps, the finite-fault factor, `const`, the geometrical
spreading, kappa and the source and path durations). Then one row per
frequency gives `f`, `source`, `gsprd`, `q`, `q_path`, `kappa`, `fmax`,
`dimin`, `site`, `lowcut`, `highcut`, `product`, `fds` and `fas`; the
program header defines each column. `product` multiplies the factors
together and `fds` is `spect_amp` itself, so their agreement shows that no
factor is missing.

`dump_factors.sh` runs `factor_dump` for every shipped `.params` file at
M 4, 6 and 8 and Rrup 5, 20 and 100 km (override with `MAGS` and `RRUPS`),
writing 54 files to `oracle/build/factors/<build>/`. It fails if `product`
and `fds` differ by more than `TOL` (1e-12 for the double build, 1e-5 for
the single build). With gfortran 13.3 the largest differences are 1.1e-15
(double) and 2.2e-7 (single).

The shipped `.params` files cover source model 1, spreading options 0 and 7,
the frequency-dependent Q of Stafford et al. (`fr1 < 0`) and the standard
two-segment Q. The other source models, finite-fault factors and spreading
options are covered by the synthetic cases that `make_golden.sh` writes.

## Golden files (`make_golden.sh`)

`make_golden.sh` writes the golden files in `tests/golden/<build>/` (see
`tests/golden/README.md` for what they contain). For every shipped `.params`
file it runs the RV, FAS and TD loop programs, using the control files in
`inputs/`, and `factor_dump` on a smaller grid than `dump_factors.sh`. It
also runs `factor_dump` for the synthetic source cases in
`inputs/source_cases.txt`, variants of `ofr.params` that cover all 13 source
models and the stress models, and for the synthetic path cases in
`inputs/path_cases.txt`, which cover the finite-fault factors, spreading
options and Q models that the shipped files do not use, and for the
synthetic site cases in `inputs/site_cases.txt`, which cover the crustal
amplification, kappa, fmax and the low-cut and high-cut filters. It also
saves the site and filter lines of each shipped `.params` file, so that the
tests can rebuild those settings without the distribution. The
double build takes about 7 minutes with gfortran 13.3, almost all of it in
the time-domain runs; `SKIP_TD=1` leaves those out.

The control files in `inputs/` are copies of the shipped ones in
`control_files/` with only the run settings changed: the magnitudes,
distances, periods and frequencies, kappa_0 and stress taken from the params
file, and the full set of output columns. `make_golden.sh` fills in the
params and output file names (`@PARAMS@`, `@COL@`). The shipped RV and TD
control files name a params file (`100b_1c_raoof_k_0.04.params`) that is not
in the distribution, so they cannot be run as they are.

The outputs are reproducible: running the script twice with the same build
gives identical files, including the time-domain results, because SMSIM
seeds its random numbers from the params file.

## Patches

Each patch is a unified diff against the source with LF line endings, and
its header says why it is needed. `patches/` applies to both builds;
`patches-double/` applies only to the double-precision build.

| Patch | Build | Why |
| --- | --- | --- |
| `patches/tmrsk_loop_rv_drvr.for.patch` | both | Removes a trailing comma in a format string, which gfortran rejects (Lahey and Intel accept it). Output is unchanged. |
| `patches-double/smsim_subs_for_rv_programs.for.patch` | double | Declares `q` and its saved values as plain `REAL` instead of `REAL(4)`, so they are promoted |
| `patches-double/td_subs.for.patch` | double | The same for `q`, `rd_calc` and `rs_calc` |
| `patches-double/rdrvaa.for.patch` | double | The same for the arguments of `rdrvaa` |

The double build also passes `-Wno-align-commons`. Promoting REALs to 8 bytes
means some COMMON blocks need padding. That is safe here because every
routine declares the COMMON blocks through the same include files.

## Agreement with the shipped example

Tested with gfortran 13.3 on Linux x86_64. Both builds reproduce all 1,212
rows, and the scenario columns (M, R, h, Rps, kappa and so on) match exactly.
Ground motions differ by these relative amounts:

| Comparison | Median | 99th percentile | Maximum |
| --- | --- | --- | --- |
| single build vs shipped reference | 2.4e-05 | 3.3e-04 | 1.2e-03 |
| double build vs shipped reference | 2.3e-05 | 2.9e-04 | 1.2e-03 |
| single build vs double build | 2.2e-05 | 2.4e-04 | 8.7e-04 |

The single and double builds differ from each other by about as much as each
differs from the reference. So the differences come mainly from
single-precision arithmetic (rounding in the integrations and peak factors),
not from the compiler. They do not change with optimisation level (`-O0`,
`-O2`) or `-ffloat-store`.

The Python port works in float64, so the double build is its main reference.
`run_example.sh` passes at a maximum relative difference of 2e-3 (`TOL`), not
exact equality.

## Not yet done

- The oscillator response (`harmoscf`), which `factor_dump` leaves out
  because it prints the Fourier spectrum without an instrument.
