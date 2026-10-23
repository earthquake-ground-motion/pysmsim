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
```

`build.sh` compiles `tmrsk_loop_rv_drvr`, `fmrsk_loop_fas_drvr` and
`tmrsk_loop_td_drvr` twice:

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

- The FAS and TD drivers are compiled but have not been run against
  reference outputs.
- The program that prints the individual spectral factors for the layer-1
  golden tests.
