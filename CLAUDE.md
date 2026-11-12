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
   - `q()` caches values on its first call (`firstcall_q`), so later changes
     to the Q parameters in the same run are ignored (pysmsim has no cache);
   - `wind_exp` recomputes only when `new_mr` is set;
   - `get_npts_for_td_calcs` overwrites `tshift` inside loops;
   - failures return sentinels (-999.9, -9.99) instead of raising errors;
   - `get_params` silently rounds the low-cut `nslope` down to a multiple of
     4 (pysmsim rejects other values);
   - source models 2, 11 and 12 always use the `log_linear` stress formula,
     whatever stress model the params file names (this gives zero stress
     and NaN corner frequencies with `hoerl`; pysmsim uses the named model);
   - for source model 12, the magnitude where eps = 1 is computed as
     -c1/c2, ignoring the reference magnitude of log10(eps)
     (the `fortran_compat` field of `TwoCornerAdditive`);
   - `gsprd_freq` (spreading options 4 and 5) uses 1.570796 for pi/2
     (the `fortran_compat` field of `AtkinsonBoore2014` and
     `AtkinsonBoore2014R10`; the effect is below 1e-6);
   - spreading option 2 sets the shared `r_ref` to 1, which changes the
     radiation constant of later scenarios if the params file gave another
     value, and options 1 and 2 ignore `r_ref` while the radiation constant
     still divides by it (pysmsim uses r_ref = 1 km for both models);
   - `dimin` uses Rrup for the Stafford Q only when `fr1` is exactly -1,
     though `q()` uses the Stafford form for any `fr1 < 0` (pysmsim's
     `Stafford2022Q` always uses Rrup);
   - the RV drivers compute the upper integration frequency `fup` (via
     `get_fup`) from `akappa`, the kappa at the reference magnitude, not
     from kappa at the scenario magnitude, so `dkappadmag` does not change
     `fup` (the `fortran_compat` argument of `pysmsim.rv.gm_rv.rv_fup`);
     with neither kappa nor fmax, `get_fup` gives `fup = 0` and no motion
     (pysmsim raises an error);
   - `PF_DK80eq2PDF` computes the DK80 peak factor as a Riemann sum up to
     a = 6, doubling the number of steps until the relative change is below
     `eps_int`, without taking its absolute value, so a sum that decreases
     stops at once; it agrees with the exact integral to about 1e-4, but
     loses more than 1e-3 with very many zero crossings (pysmsim integrates
     exactly; `pysmsim.rv.peak_factors.der_kiureghian_1980_fortran` and
     `fortran_compat` reproduce the sum and its `na_dk80_eq2`);
   - `drms_dex_f` and `bt12_td_rv_for_pga_or_pgv_f` treat a BT15 table cell
     as missing only when its lower-left corner is zero (no shipped table
     has zeros, so this never matters);
   - `gm_rv` applies the oscillator correction to the Wood-Anderson and
     WWSSN-SP responses too, with a made-up oscillator frequency and the
     negative "period" the loop driver uses to select them; pysmsim's RV
     functions accept only ground motion and oscillators for now;
   - the loop drivers set `dkappadmag` and `amagkref` to 0 for every
     kappa_0, including the one taken from the params file, so kappa never
     depends on magnitude in a loop run (the `fortran_compat` argument of
     `pysmsim.drivers.loop_fas.fas_loop`; pysmsim keeps the params-file
     slope unless the grid gives kappa_0 values);
   - the `gsprd*Qop` column of `fmrsk_loop_fas_drvr` uses Rps for every Q
     model, though `spect_amp` uses Rrup for the Stafford Q (output only);
     that driver also always uses the strike-slip CY14 Ztor;
   - `skipcmnt` always skips the line before the values it guards, comment
     or not, so a params file without that comment line is misread (matters
     for the converter).
3. **One TOML input file per run.** It replaces the `.params`, `.ctl` and
   `.pars` files and is read with `tomllib`.
   - Settings are named keys, not positions.
   - Options are words (`"bt15"`), not integer codes. Fortran numbers
     appear only in docstrings and in the converter.
   - Each model with its own parameters is its own class, chosen by its
     `model` name, holding only the parameters it uses and defining its own
     behaviour; no if/elif chains over model types. The TOML section is a
     pydantic discriminated union on `model`, so keys another model uses are
     rejected. Source, stress, finite-fault, spreading, Q and high-cut
     models (`pysmsim.spectrum.source`, `.stress`, `.finite_fault`,
     `.spreading`, `.attenuation`, `.filters`) follow this. Settings with
     only one form in SMSIM (the site terms in `pysmsim.spectrum.site`, the
     low-cut filter) are single classes. Compiled kernels are `pysmsim._jit.Kernel`
     objects (compiled function plus NumPy twin) named by the class.
   - The file carries `schema_version`.
   - A run can start from a preset (`base = "wna_bt15"`).
   - The BT15 and crustal-amp tables are bundled package data, chosen by name.
   - `pysmsim convert old.params old.ctl -o run.toml` migrates old inputs.
4. **Numba from the start**, on the hot paths:
   - `spect_amp` kernels;
   - the RV moments: one adaptive Gauss-Kronrod integration of all four
     moments that evaluates the compiled spectrum kernels on arrays
     (`pysmsim.rv.moments.integrate`; decided in phase 2, as the spectrum
     is not one compiled function and per-point `quad` calls were too
     slow);
   - the Nigam-Jennings oscillator;
   - `prange` over periods, scenarios and `nsims`.

   FFTs and random noise stay in NumPy (`numpy.random.Generator`). Every
   kernel has a plain NumPy twin, and the tests also run with
   `NUMBA_DISABLE_JIT=1`. Use `cache=True`.
5. **pandas for tables and I/O** (scenario grids, results, golden-test merges).
   Outputs are Parquet (via pyarrow), CSV, a `.col` writer in the Fortran
   column layout for comparison, and one HDF5 file (via plain h5py, the
   owner's choice) holding a run's scenario, FAS and RV tables. Consider Polars only if profiling later shows
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
  | `odeint` | adaptive Gauss-Kronrod (`pysmsim.rv.moments.integrate`) |
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
  spectrum/  source.py, stress.py, finite_fault.py, spreading.py, attenuation.py,
             site.py, filters.py, response.py, distance.py, fas.py
  rv/        moments.py, peak_factors.py, duration.py, gm_rv.py
  td/        noise.py, window.py, acc_ts.py, response.py, intensity.py, gm_td.py
  drivers/   loop_rv.py, loop_td.py, loop_fas.py, run.py, single.py, _parallel.py
  cli.py     pysmsim run | scenario | convert | check
tests/unit, tests/golden
oracle/      scripts that build and run the Fortran (no source edits)
```

Dependencies: python>=3.12, numpy>=2.0, scipy>=1.13, numba>=0.60, pandas>=2.2,
pyarrow, h5py. Dev dependencies: pytest, ruff, mypy, hypothesis.

## Phases (each ends at a gate that compares against Fortran)

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | Fortran oracle (single and double builds), golden files, repository set-up | Built RV binary reproduces the example `.col` within 2e-3 max relative difference (decided by the owner; the shipped `.col` itself carries single-precision rounding of that size) |
| 1 | Dataclasses, TOML schema and converter, spectrum modules, FAS loop | Factors and FAS within 1e-4 |
| 2 | RV: moments, peak factors, `osc_crrctn` 0-5, RV loop, CLI → release 0.1 | PSA within 1e-3 on the example and all params files |
| 3 | TD: noise, windows, `acc_ts`, oscillator, Arias, TD loop | Injected-noise trace within 1e-5; means within 3 standard errors |
| 4 | Single-scenario drivers, docs (including why RV results can differ from SMSIM's by more than 1e-3; see Validation), Numba tuning → release 1.0 | All layers green in CI, plus an acceptance run on the owner's inputs |

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
  The statistical test allows |z| < 4 with at most 3% of rows beyond 3,
  because every scenario reuses the same noise (the owner kept this,
  2026-11-12; observed max 2.9 arithmetic, 3.01 geometric).
- All shipped `.params` files use source model 1 and spreading options 0 or 7.
  The other branches need synthetic `.params` variants as golden cases.
- The shipped example `example.tmrsk_loop_rv_drvr/` has a 1,212-row `.col`
  file. Its `.ctl` is dated 01/22/21 and has no mechanism line, so update a
  copy before running it.
- SMSIM's spectral moments (NR `odeint` with `eps_int` = 1e-5) are only
  accurate to a few 1e-4 (2.6e-3 in one golden case), because the step
  error accumulates; pysmsim's agree with `quad` to about 1e-6. Narrow-band
  oscillator responses amplify this through the bandwidth
  sqrt(1 - m1^2/(m0*m2)), so a few RV golden rows (5 of about 4,300)
  differ from SMSIM by more than 1e-3 (at most 4.3e-3). The example run
  agrees within 6e-4 of the double build. The owner accepted this
  (2026-11-11) rather than emulating `odeint`; explain it in the user
  documentation in phase 4.

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
  - `rv/`: `moments.py` (moments, fup), `peak_factors.py`, `duration.py`
    (durations, oscillator corrections, BT15 tables) and `gm_rv.py`
    (`rv_motion`, one scenario);
  - `td/`: `window.py` (`time_grid`, the box and exponential windows),
    `noise.py`, `acc_ts.py` (one series, batched over simulations),
    `intensity.py` (velocity, displacement, Arias, durations, and
    `series_measures`, all of them in one Numba pass, bitwise equal to the
    NumPy functions), `response.py` (Nigam-Jennings oscillator, a Numba
    kernel with a NumPy twin, parallel over series and periods together) and `gm_td.py` (`td_motion`: nsims simulations of one scenario,
    and their statistics). The noise comes from `numpy.random.default_rng`
    seeded with `[td] seed` at every scenario, as SMSIM reseeds; SMSIM's
    `ran1`/`gasdev` are not reproduced (only the noise-injection test needs
    the Fortran noise, from `oracle/programs/td_dump.for`);
  - `drivers/`: `loop_fas.py`, `loop_rv.py`, `loop_td.py` and `run.py`. Output tables
    have one row per scenario with the ground motions as columns, the
    layout the owner prefers: `fas_loop` gives `fas(f)` (cm/s), `rv_loop`
    `pgv` (cm/s), `pga` and `psa(T)` (cm/s^2), with no conversion to g.
    `fas_loop_details` and `rv_loop_details` give one row per scenario and
    frequency or measure, with the intermediate quantities. `run` computes
    both and returns `RunResult`: tables sharing their rows (scenarios,
    fas, rv, and with `td=True` td and td_details), as the owner wants for
    a typical run. `td_loop_details` gives every TD measure's mean, std,
    gmean and gfactor; `td_loop` the means. `single.py` is one scenario
    in full detail (SMSIM's `gm_rv_drvr`, `gm_td_drvr`, `FAS_DRVR`):
    `rv_scenario`, `td_scenario` (`.table()`, `.series(k)` with time, acc,
    vel, dis) and `fas_scenario`, all taking Rrup and leaving kappa
    magnitude-dependent, as those programs do (only the loop drivers force
    `dkappadmag` to 0). SMSIM's three programs never set the common
    `rrup`, which the Stafford spreading and `fr1 = -1` Q use; the oracle
    patches that bug (`*.rrup.patch`) and pysmsim does not reproduce it,
    even with `fortran_compat`;
    `_parallel.py` (`map_tasks`) runs the RV and TD loops' tasks (blocks
    of scenarios) in `workers` spawned processes; each task is computed as
    in one process, and TD reseeds at every scenario, so the tables are
    identical whatever `workers` is;
  - `io/`: `config.py` (TOML; `params_table` gives the parameter
    sections as a dict), `bt15_tables.py`, `hdf5.py` (`write_hdf5`
    and `read_hdf5`; a group per table, one dataset per column, FAS and PSA
    as 2-D arrays, the run file's TOML in an attribute; the owner kept one
    dataset per column rather than structured arrays) and `results.py`
    (`write_results` for csv, parquet, hdf5 and SMSIM's `.col` layout;
    TD columns get the prefix `td_` in csv and parquet, and `/td/<stat>/<measure>`
    in HDF5; `.col` holds RV only). `write_td_series` adds `/td/series`:
    `times` (0 to the longest npts; attributes dt, tolerance, and per
    scenario nstart, nstop, first, last, npts) and `acceleration` (nsims,
    ns, nscenarios). The owner chose (2026-11-12) the whole series in
    float32 by default; a tolerance > 0 stores each scenario only from
    first to last, where it exceeds tolerance times its peak, NaN
    (unwritten chunks) elsewhere (the FFT leaves series nonzero before
    nstart and after nstop). Integers ×1000 were rejected: a fixed step of
    0.001 cm/s² ruins weak motions. It
    re-simulates scenario by scenario (same seed) and writes as it goes;
  - `cli.py`: `pysmsim run` (`--td` adds the TD simulations, `--no-rv`
    drops RV), `pysmsim scenario RUN -m M (-r RRUP | --rjb RJB) [--td]
    [--series K] [-o DIR]` (prints the single-scenario tables; `-o` writes
    them as rv.csv, td.csv, fas.csv, td_series.csv; periods, damping,
    frequencies and the Rjb conversion come from the run file) and
    `pysmsim check` (also `python -m pysmsim`).
    `[scenarios] frequencies` in the run file adds the FAS to a
    run. A grid may have at most 1,000 periods and 1,000 frequencies (the
    owner's limit, so that a run file cannot exhaust memory; checked before
    `{start, stop, count}` is expanded). Output paths in a run file are
    not restricted (the owner accepted that);
  - `io/legacy.py` and `pysmsim convert PARAMS [CTL] [-o run.toml]`:
    convert a `.params` file (dated 02/25/21) and a `tmrsk_loop_rv_drvr`
    control file (03/03/23, or 01/22/21 without the mechanism line) or a
    `tmrsk_loop_td_drvr` one (03/03/23; no loop-order or PGV/PGA lines,
    recognised by the program named in its heading) into a run file. The reader copies `skipcmnt` and Fortran list-directed input
    (continuation lines, `n*v`, `d` exponents, a count and its list in one
    statement). Where SMSIM would behave differently it adds notes, printed
    and written at the top of the run file. Without a control file the
    scenarios are example values. Tests compare every shipped and synthetic
    oracle `.params` file with `oracle_files.py`; copies of the shipped
    files are in `tests/data/smsim/`.
- `src/pysmsim/data/`: bundled tables (`bt15/` holds the two BT15 `.pars`
  files as `bt15_wna_acr.pars` and `bt15_ena_scr.pars`; `crustal_amps/`,
  `presets/`).
- `tests/unit/` and `tests/golden/`: unit tests and Fortran comparisons
  (`tests/golden/README.md` describes the golden files).
- `oracle/`:
  - `shims/`: the gfortran date/time replacement;
  - `inputs/`: the control files for the golden runs;
  - `programs/`: test-only Fortran programs (`factor_dump`,
    `response_dump`, `td_dump`);
  - `build/`: build output, which Git ignores.
- `notebooks/`: exploration notebooks.
- `examples/`: example run TOML files and scripts; `benchmark.py` times
  the FAS, RV and TD loops (and checks that `--workers` changes nothing).
- `docs/`: documentation.
- `papers/`: the reference papers, which Git ignores.

Empty folders hold a `.gitkeep` until real files arrive.

## Conventions

- Run `ruff` and `mypy` cleanly, use type hints throughout, and write
  NumPy-style docstrings that cite the paper or the Fortran routine behind
  each equation.
- Never edit the Fortran reference source.
