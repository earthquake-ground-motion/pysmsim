# CLAUDE.md

Context for Claude (and humans) working in this repository.

## What this is

`pysmsim` is a Python 3.12+ re-implementation of SMSIM, David Boore's Fortran
programs for simulating earthquake ground motions with the stochastic method
(point source, magnitude M, distance R). The reference source is the
2023-03-03 Fortran distribution. It is kept read-only, outside this repository:
never edit it, and never copy Numerical Recipes code from it.

The repository is licensed under GPL-3.0 (see `LICENSE`). Numerical Recipes
code is not GPL-compatible, which is another reason never to copy it.

Repository specifications are still to come from the project owner. Where they
conflict with this file, the specifications win: update this file to match.

Planning documents (claude.ai):
- Migration workplan: https://claude.ai/code/artifact/238d68da-9091-4bc1-b801-9787ade8c0be
- SMSIM core workflow map: https://claude.ai/code/artifact/9aefd067-e15d-4062-89da-340ba9c2b09e

## Reference papers (`papers/`)

The `papers/` folder holds Markdown conversions of the papers behind the
methods. It is listed in `.gitignore` and must never be committed (do not
`git add -f` it, and do not quote large passages from it into the code or
docs). Read the relevant paper before implementing or changing an equation,
and cite it in the docstring.

The PDFs shipped with the Fortran distribution, and what to read them for:

- `smsim_manual.pdf`: the program manual (inputs, options, drivers)
- `boore_stochastic_method_pageoph.pdf`: Boore (2003), overview of the stochastic method
- `1983bssa_stochastic_sims.pdf`: Boore (1983), the original method
- `boore_exsim_smsim_bssa_dec09.pdf`: Boore (2009), SMSIM compared with EXSIM; finite-fault distance
- `boore_thompson_bssa_2015_..._published.pdf`: Boore and Thompson (2015), the BT15 RMS duration and path duration
- `generalized_2_corner_source_model_..._2014.pdf`: Boore et al. (2014), the two-corner source models 11 and 12
- `generalize_boore-thompson_2015_finite-fault_factor.v3.pdf`: the finite-fault factor in `rmod_calc`
- `geometrical_spreading_with_smooth_transition_..._segments.v3.pdf`: the smoothed hinged spreading
- `f0_stress_relations_boore_allmann_shearer...v04.pdf`: corner frequency and stress relations
- `generic_velocity_density_models_bssa_2016.pdf`: the crustal velocity and density models behind the crustal amplifications

If a paper needed for a routine is missing from `papers/`, say so rather than
working from memory.

## The science in one paragraph

Everything rests on one Fourier amplitude spectrum, `spect_amp(f)`. It is the
product of:
- the source term (M0, corner frequencies fa and fb, 13 source-shape models);
- geometrical spreading (options 0-7) and Q(f) attenuation;
- crustal amplification, kappa and fmax;
- low-cut and high-cut filters;
- the oscillator or instrument response.

There are two ways to turn the spectrum into ground motion:
- **RV (random vibration):** integrates spectral moments m0, m1, m2 and m4 and
  converts an RMS motion into a peak. The peak factors are Cartwright and
  Longuet-Higgins 1968 and Der Kiureghian 1980 (DK80 is the reported value).
  The RMS-duration correction is set by `osc_crrctn` 0-5 (none, BJ84, LP99,
  BT15). This path is deterministic.
- **TD (time domain):** shapes windowed Gaussian or uniform noise with the
  spectrum via FFT, runs `nsims` realisations and averages the peaks, Arias
  intensity and durations. This path is stochastic.

Units are cgs (cm, s, g/cm³), with distances in km and g = 981 cm/s².

## Decisions already made

1. **RV loop first.** Release 0.1 is the RV loop (`tmrsk_loop_rv_drvr`
   equivalent) only. The FAS loop is built first, but only as an internal check
   of `spect_amp`. TD comes after.
2. **Fix Fortran quirks by default.** Keep the original behaviour behind a
   `fortran_compat` flag, which the golden tests use. Each quirk is documented
   and has a named test. Known quirks:
   - `q()` caches values on its first call (`firstcall_q`);
   - `wind_exp` recomputes only when `new_mr` is set;
   - `get_npts_for_td_calcs` overwrites `tshift` inside loops;
   - failures return sentinels (-999.9, -9.99) instead of raising errors.
3. **One TOML input file per run.** It replaces the `.params`, `.ctl` and
   `.pars` files and is read with `tomllib`.
   - Settings are named keys, not positions.
   - Options are words (`"bt15"`), not integer codes.
   - The file carries `schema_version`.
   - A run can start from a preset (`base = "wna_bt15"`).
   - The BT15 and crustal-amp tables are bundled package data, chosen by name.
   - `pysmsim convert old.params old.ctl -o run.toml` migrates old inputs.
4. **Numba from the start**, on the hot paths:
   - `spect_amp` kernels;
   - RV integrands as `@cfunc` passed to `scipy.integrate.quad` through a
     `LowLevelCallable`;
   - the Nigam-Jennings oscillator;
   - `prange` over periods, scenarios and `nsims`.

   FFTs and random noise stay in NumPy (`numpy.random.Generator`). Every
   kernel has a plain NumPy twin, and the tests also run with
   `NUMBA_DISABLE_JIT=1`. Use `cache=True`.
5. **pandas for tables and I/O** (scenario grids, results, golden-test merges).
   Outputs are Parquet (via pyarrow), CSV, and a `.col` writer in the Fortran
   column layout for comparison. Consider Polars only if profiling later shows
   table handling is a bottleneck. Physics functions take and return NumPy
   arrays, never DataFrames.
6. **No precompiled Fortran binaries exist.** Phase 0 builds them from the
   unmodified source with gfortran, using two shims:
   - `DATETIME.FOR` uses the Lahey `DATE`/`TIME` extensions, so a
     `date_and_time` replacement is linked instead;
   - the `\smsim\` paths inside `.params` files are rewritten in copies.

   Make a default build and a `-fdefault-real-8 -fdefault-double-8` build. If
   gfortran fails, fall back to Intel ifx.

## Design rules

- Re-implement from the modular Fortran files (`rv_td_subs.for`, `rv_subs.for`,
  `gm_td.for`, `acc_ts.for`, parts of `td_subs.for`). Do not translate line by
  line. `smsim_subs_for_rv_programs.for` and `td_subs.for` are bundled copies
  of those files; ignore them.
- The `smsim.fi` common blocks become frozen dataclasses (`Scenario`, `Source`,
  `Spreading`, `Q`, `Site`, `Filters`, `RVConfig`, `TDConfig`, `Window`).
  Physics functions are pure: (frequency array, scenario, params) in, arrays
  out. Derived values (M0, fa, fb, the constant factor) come back in a
  `SourceTerms` result and are never mutated state.
- Replace the Numerical Recipes routines with SciPy and NumPy (the licence
  forbids copying them):

  | Numerical Recipes | Replacement |
  | --- | --- |
  | `odeint` | `quad` |
  | `qromb`, `qmidpnt` | `quad` |
  | `realft`, `four1` | `numpy.fft.rfft`/`irfft`; redo the normalisation |
  | `gasdev`, `ran1` | `Generator` |
  | `locate` | `searchsorted` |
  | `zbrent` | `brentq` |
  | `sort2` | `argsort` |

- Raise exceptions or return NaN instead of sentinel values. Write sentinels
  only in `.col` compatibility output.
- Only `io/` and `drivers/` touch files.

## Planned layout (provisional, pending the repository specifications)

```text
src/pysmsim/
  params.py, scenario.py
  io/        config.py (TOML), legacy.py (convert), bt15_tables.py, results.py
  spectrum/  source.py, path.py, site.py, filters.py, distance.py, fas.py
  rv/        moments.py, peak_factors.py, duration.py, gm_rv.py
  td/        noise.py, window.py, acc_ts.py, response.py, intensity.py, gm_td.py
  drivers/   loop_rv.py, loop_td.py, loop_fas.py
  cli.py     pysmsim run | convert | check
tests/unit, tests/golden
oracle/      scripts that build and run the Fortran (no source edits)
```

Dependencies: python>=3.12, numpy>=2.0, scipy>=1.13, numba>=0.60, pandas>=2.2,
pyarrow. Dev dependencies: pytest, ruff, mypy, hypothesis.

## Phases (each ends at a gate that compares against Fortran)

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | Fortran oracle (single and double builds), golden files, repository set-up | Built RV binary reproduces the example `.col` within 2e-3 max relative difference (decided by the owner; the shipped `.col` itself carries single-precision rounding of that size) |
| 1 | Dataclasses, TOML schema and converter, spectrum modules, FAS loop | Factors and FAS within 1e-4 |
| 2 | RV: moments, peak factors, `osc_crrctn` 0-5, RV loop, CLI → release 0.1 | PSA within 1e-3 on the example and all params files |
| 3 | TD: noise, windows, `acc_ts`, oscillator, Arias, TD loop | Injected-noise trace within 1e-5; means within 3 standard errors |
| 4 | Single-scenario drivers, docs, Numba tuning → release 1.0 | All layers green in CI, plus an acceptance run on the owner's inputs |

Later releases: 1.1 adds M and R from a file, Sea22 Q sampling, SMC export and
`fas_td`. Out of scope: EXSIM, NRATTLE and SITE_AMP, and any change to the
physics.

## Validation

- Compare against golden outputs from the Fortran oracle, committed under
  `tests/golden/` with the compiler version and flags. Python tests never need
  a Fortran compiler.
- Use relative tolerances, never exact equality, because the Fortran works
  mostly in single precision. Python should match the double-precision build
  more closely than the default build.
- The TD random sequences cannot match. Compare statistics, and use a
  noise-injection test: feed the Fortran noise into Python `acc_ts` and expect
  agreement to about 1e-5.
- All shipped `.params` files use source model 1 and spreading options 0 or 7.
  The other branches need synthetic `.params` variants as golden cases.
- The shipped example `example.tmrsk_loop_rv_drvr/` has a 1,212-row `.col`
  file. Its `.ctl` is dated 01/22/21 and has no mechanism line, so update a
  copy before running it.

## How we work

- Develop in small steps. Each step gets its own new Git branch from `main`,
  named by kind: `feature/...` for a new feature, `bug/...` for a bug fix,
  `doc/...` for documentation (repository set-up used `setup/...`).
- Use pydantic to validate the TOML input and hatchling as the build backend.
- Never commit, push or merge until the project owner has reviewed the changes
  and said to go ahead.
- Ask the owner when anything is unclear rather than guessing.
- Keep everything usable from Jupyter. Every physics step (a source spectrum,
  a spreading factor, the spectral moments, a peak factor, a single time
  series) is a public function that can be imported and called on its own
  with plain arguments and returns NumPy arrays or small dataclasses. Drivers
  only compose these functions; no logic is hidden inside them.
  `notebooks/` holds exploration notebooks, and the `notebook` extra installs
  JupyterLab and matplotlib.

## Repository layout

- `src/pysmsim/`: the package (`io/`, `spectrum/`, `rv/`, `td/`, `drivers/`).
- `src/pysmsim/data/`: bundled tables (`bt15/`, `crustal_amps/`, `presets/`).
- `tests/unit/` and `tests/golden/`: unit tests and Fortran comparisons.
- `oracle/`:
  - `shims/`: the gfortran date/time replacement;
  - `inputs/`: patched copies of the `.params` and `.ctl` files;
  - `build/`: build output, which Git ignores.
- `notebooks/`: exploration notebooks.
- `examples/`: example run TOML files.
- `docs/`: documentation.
- `papers/`: the reference papers, which Git ignores.

Empty folders hold a `.gitkeep` until real files arrive.

## Conventions

- Run `ruff` and `mypy` cleanly, use type hints throughout, and write
  NumPy-style docstrings that cite the paper or the Fortran routine behind
  each equation.
- Never edit the Fortran reference source.
