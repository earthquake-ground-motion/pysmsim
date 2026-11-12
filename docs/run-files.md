# Run files

A run file is one TOML file that holds everything a run needs: the model
parameters (what an SMSIM `.params` file holds), the scenarios (what a
loop-driver `.ctl` file holds) and the output. `examples/wna_bt15_example.toml`
is SMSIM's own example run, with comments; `pysmsim convert` writes a run
file from SMSIM's files (see [Command line](cli.md#converting-smsim-input-files)).

```toml
schema_version = 1
title = "Western North America params"

[medium]
[source]
[source.stress]
[finite_fault]
[spreading]
[q]
[path_duration]
[site]
[filters.low_cut]
[filters.high_cut]
[rv]
[td]
[scenarios]
[output]
```

The rules:

- The keys in each section are the fields of the matching class in
  `pysmsim.params`, `pysmsim.scenario` or `pysmsim.spectrum`, so the
  [API reference](reference/index.md) documents each of them.
- Options are words, not SMSIM's integer codes: `osc_correction = "bt15_wna"`,
  `window = "exponential"`.
- In `[source]`, `[source.stress]`, `[finite_fault]`, `[spreading]`, `[q]`
  and `[filters.high_cut]`, `model` names the model, and the other keys
  are that model's parameters. A key the named model does not use is an
  error.
- Keys with a default (below, `key = default`) may be left out; the others
  are required.
- The file is checked when it is read: unknown keys, wrong types, values
  out of range and inconsistent settings are reported with the key that
  caused them. `pysmsim check` checks a file without running it.

Units follow SMSIM: density in g/cm³, velocity in km/s, distance in km,
stress in bars, frequency in Hz, time in s.

## Medium

```toml
[medium]
density = 2.72         # g/cm^3, near the source
shear_velocity = 3.5   # km/s, near the source
partition = 0.707      # partition onto two horizontal components
radiation = 0.55       # average radiation pattern
free_surface = 2.0     # free-surface amplification
```

See [`Medium`][pysmsim.params.Medium].

## Source

`[source] model` is one of:

| `model` | Class | Keys |
| --- | --- | --- |
| `"single_corner"` | [`SingleCorner`][pysmsim.spectrum.source.SingleCorner] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5, `stress`, `pf_a` = 2.0, `pd_a` = 1.0 |
| `"joyner_1984"` | [`Joyner1984`][pysmsim.spectrum.source.Joyner1984] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5, `stress`, `magnitude_hinge`, `fb_over_fa` = 4.0 |
| `"two_corner_multiplicative"` | [`TwoCornerMultiplicative`][pysmsim.spectrum.source.TwoCornerMultiplicative] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5, `stress`, `log10_fa`, `pf_a` = 2.0, `pd_a` = 0.5, `pf_b` = 2.0, `pd_b` = 0.5 |
| `"two_corner_additive"` | [`TwoCornerAdditive`][pysmsim.spectrum.source.TwoCornerAdditive] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5, `stress`, `log10_fa`, `log10_eps`, `pf_a` = 2.0, `pd_a` = 1.0, `pf_b` = 2.0, `pd_b` = 1.0, `fortran_compat` = false |
| `"atkinson_1993"` | [`Atkinson1993`][pysmsim.spectrum.source.Atkinson1993] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"atkinson_silva_1997"` | [`AtkinsonSilva1997`][pysmsim.spectrum.source.AtkinsonSilva1997] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"atkinson_boore_1998_california"` | [`AtkinsonBoore1998California`][pysmsim.spectrum.source.AtkinsonBoore1998California] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"atkinson_silva_2000"` | [`AtkinsonSilva2000`][pysmsim.spectrum.source.AtkinsonSilva2000] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"atkinson_2005"` | [`Atkinson2005`][pysmsim.spectrum.source.Atkinson2005] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"haddon_1996"` | [`Haddon1996`][pysmsim.spectrum.source.Haddon1996] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"boatwright_choy"` | [`BoatwrightChoy`][pysmsim.spectrum.source.BoatwrightChoy] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"joyner_ena"` | [`JoynerENA`][pysmsim.spectrum.source.JoynerENA] | `duration_weight_fa` = 0.5, `duration_weight_fb` = 0.5 |
| `"ji_archuleta_2020"` | [`JiArchuleta2020`][pysmsim.spectrum.source.JiArchuleta2020] | none |

`stress` is the `[source.stress]` table, with its own `model`:

| `model` | Class | Keys |
| --- | --- | --- |
| `"log_linear"` | [`LogLinearStress`][pysmsim.spectrum.stress.LogLinearStress] | `stress_ref`, `dlog10_stress_dm` = 0.0, `magnitude_ref` = 6.5 |
| `"hoerl"` | [`HoerlStress`][pysmsim.spectrum.stress.HoerlStress] | `c1`, `c2`, `c3` |
| `"stafford_2022"` | [`Stafford2022Stress`][pysmsim.spectrum.stress.Stafford2022Stress] | `c1`, `c2` |

`duration_weight_fa` and `duration_weight_fb` weight 1/fa and 1/fb in the
source duration. `log10_fa` and `log10_eps` are three coefficients each, as
in SMSIM. For example:

```toml
[source]
model = "single_corner"
pf_a = 2.0
pd_a = 1.0

[source.stress]
model = "log_linear"
stress_ref = 250.0     # bars at magnitude_ref
dlog10_stress_dm = 0.0
magnitude_ref = 6.5
```

## Finite fault

`[finite_fault] model` sets the finite-fault factor h that turns Rrup into
the effective distance Rps:

| `model` | Class | Keys |
| --- | --- | --- |
| `"none"` | [`NoFiniteFault`][pysmsim.spectrum.finite_fault.NoFiniteFault] | none |
| `"one_line"` | [`OneLine`][pysmsim.spectrum.finite_fault.OneLine] | `combination` = "quadrature", `c1`, `c2` |
| `"two_lines"` | [`TwoLines`][pysmsim.spectrum.finite_fault.TwoLines] | `combination` = "quadrature", `c1`, `c2`, `c3`, `c4` |
| `"two_lines_smoothed"` | [`TwoLinesSmoothed`][pysmsim.spectrum.finite_fault.TwoLinesSmoothed] | `combination` = "quadrature", `c1`, `c2`, `c3`, `c4`, `delta_m` |
| `"generalized"` | [`Generalized`][pysmsim.spectrum.finite_fault.Generalized] | `combination` = "quadrature", `c1`, `c2`, `c3`, `c4`, `h_ratio` |
| `"stafford_2022"` | [`Stafford2022FiniteFault`][pysmsim.spectrum.finite_fault.Stafford2022FiniteFault] | `combination` = "quadrature", `h_alpha`, `h_beta`, `h_gamma`, `h_delta`, `h_epsilon` |

`combination` is `"quadrature"` (Rps = sqrt(R² + h²)) or `"sum"`. Without
the section there is no finite-fault factor.

## Geometrical spreading

| `model` | Class | Keys |
| --- | --- | --- |
| `"hinged"` | [`Hinged`][pysmsim.spectrum.spreading.Hinged] | `segments`, `r_ref` = 1.0 |
| `"stafford_generalized"` | [`StaffordGeneralized`][pysmsim.spectrum.spreading.StaffordGeneralized] | `gamma1`, `r_t`, `gamma2`, `g_ratio`, `r_ref` = 1.0 |
| `"stafford_optimal"` | [`StaffordOptimal`][pysmsim.spectrum.spreading.StaffordOptimal] | `gamma1`, `r_t`, `gamma2`, `r_ref` = 1.0 |
| `"atkinson_2011"` | [`Atkinson2011`][pysmsim.spectrum.spreading.Atkinson2011] | none |
| `"boore_trial"` | [`BooreTrial`][pysmsim.spectrum.spreading.BooreTrial] | `r1`, `slope_lf`, `slope_hf`, `slope_far`, `ft1`, `ft2` |
| `"atkinson_2012"` | [`Atkinson2012`][pysmsim.spectrum.spreading.Atkinson2012] | `r_ref` = 1.0 |
| `"atkinson_boore_2014"` | [`AtkinsonBoore2014`][pysmsim.spectrum.spreading.AtkinsonBoore2014] | `h`, `r_ref` = 1.0, `fortran_compat` = false |
| `"atkinson_boore_2014_r10"` | [`AtkinsonBoore2014R10`][pysmsim.spectrum.spreading.AtkinsonBoore2014R10] | `h`, `r_ref` = 1.0, `fortran_compat` = false |

The hinged model takes a list of segments, each with `r_start` (km), `a`,
and optionally `b` and `m_ref`, for a slope a + b(M - m_ref) beyond
`r_start`:

```toml
[spreading]
model = "hinged"
r_ref = 1.0
segments = [
    { r_start = 1.0, a = -1.0 },
    { r_start = 40.0, a = -0.5 },
]
```

## Q

| `model` | Class | Keys |
| --- | --- | --- |
| `"simple"` | [`SimpleQ`][pysmsim.spectrum.attenuation.SimpleQ] | `c_q` = 3.5, `q0`, `eta` = 0.0 |
| `"piecewise"` | [`PiecewiseQ`][pysmsim.spectrum.attenuation.PiecewiseQ] | `c_q` = 3.5, `qr1`, `s1` = 0.0, `fr1` = 1.0, `qr2`, `s2` = 0.0, `fr2` = 1.0, `ft1` = 1.0, `ft2` = 1.0 |
| `"stafford_2022"` | [`Stafford2022Q`][pysmsim.spectrum.attenuation.Stafford2022Q] | `c_q` = 3.5, `q0`, `eta_a`, `eta_b`, `eta_g` |

`c_q` is the velocity (km/s) in the anelastic factor exp(-πfR/(c_q Q)).

## Path duration

```toml
[path_duration]
distances = [0.0, 7.0, 45.0, 125.0, 175.0, 270.0]   # km
durations = [0.0, 2.4, 8.4, 10.9, 17.4, 34.2]        # s
slope_beyond = 0.156                                 # s/km beyond the last distance
```

The path duration is linear between the points, `slope_beyond` (s/km) past
the last one, and the first duration below the first one. See
[`PathDuration`][pysmsim.params.PathDuration].

## Site

```toml
[site]
amp_frequencies = [0.01, 0.1, 1.0, 10.0]   # Hz
amplifications = [1.0, 1.1, 1.5, 2.5]
kappa0 = 0.035              # s
dkappa_dmag = 0.0           # change of kappa per magnitude unit
kappa_magnitude_ref = 0.0
fmax = 0.0                  # Hz; 0 for no fmax filter
```

The amplification is interpolated linearly in frequency and
logarithmically in amplitude between the points, as SMSIM's `spect_amp` does, and is constant
beyond the table. See
[`Site`][pysmsim.spectrum.site.Site].

## Filters

```toml
[filters.low_cut]      # acausal Butterworth
fcut = 0.0             # Hz; 0 for no low-cut filter
order = 4              # each of two passes: an even number (SMSIM nslope = 2 * order)

[filters.high_cut]
model = "none"
```

| `[filters.high_cut] model` | Class | Keys |
| --- | --- | --- |
| `"none"` | [`NoHighCut`][pysmsim.spectrum.filters.NoHighCut] | none |
| `"raised_cosine"` | [`RaisedCosineHighCut`][pysmsim.spectrum.filters.RaisedCosineHighCut] | `f1`, `f2`, `eta` = 1.0 |
| `"quarter_cosine"` | [`QuarterCosineHighCut`][pysmsim.spectrum.filters.QuarterCosineHighCut] | `f1`, `f2`, `eta` = 1.0 |

Both sections may be left out (no filters).

## Random vibration

```toml
[rv]
zup = 10.0                  # upper limit of the CL56 peak-factor integral
eps_int = 1.0e-5            # relative accuracy of the integrations
amp_cutoff = 1.0e-3         # sets the upper integration frequency fup
osc_correction = "bt15_wna"
bt15_tables = ["bt15_wna_acr", "bt15_ena_scr"]
```

`osc_correction` is the RMS-duration correction for oscillators: `"none"`
(Boore, 1983), `"bj84"` (Boore and Joyner, 1984), `"lp99"` (Liu and
Pezeshk, 1999), `"bt15_wna"` or `"bt15_ena"` (Boore and Thompson, 2015,
the first or second table of `bt15_tables`) or `"bt15_average"` (the
average of the two). The BT15 tables ship with pysmsim. See
[`RVConfig`][pysmsim.params.RVConfig].

## Time domain

```toml
[td]
window = "exponential"     # or "box"
taper = 0.05               # box window: fraction of the duration for the tapers
eps_w = 0.2                # exponential window shape
eta_w = 0.05
f_tb2te = 2.0
f_te_xtnd = 1.0
dur_fctr = 1.3             # series length, in motion durations after tshift
dt = 0.005                 # s
tshift = 50.0              # s before the motion starts
seed = 123
nsims = 800
noise = "gaussian"         # or "uniform"
```

Every key has a default, so the section may be left out. See
[`TDConfig`][pysmsim.params.TDConfig].

## Scenarios

```toml
[scenarios]
magnitudes = { start = 4.0, stop = 8.0, step = 2.0 }   # or a list
distances = [10.0, 20.0, 50.0]                         # km
distance_metric = "rjb"            # or "rrup"
rjb_to_rrup = "cy14_ztor"          # or "none", "magnitude_function"
mechanism = "strike_slip"          # or "normal", "reverse"
periods = { start = 0.01, stop = 100.0, count = 200 }   # s, log-spaced
pga = true
pgv = true
damping = 0.05
frequencies = { start = 0.01, stop = 100.0, count = 41 }   # Hz, for the FAS
kappa0 = [0.02, 0.05]              # optional: loop over kappa_0 (s)
stress = [100.0]                   # optional: loop over stress (bars)
```

Every combination of stress, kappa_0, magnitude and distance is a scenario,
computed in that loop order.

- `magnitudes` is a list or `{start, stop, step}`. `distances`, `periods`,
  `frequencies`, `kappa0` and `stress` are lists or `{start, stop, count}`
  for `count` values spaced evenly in log, as in SMSIM's control files.
- Joyner-Boore distances (`distance_metric = "rjb"`) are converted to
  Rrup = sqrt(Rjb² + h²), with h the Chiou and Youngs (2014) depth to the
  top of rupture for the `mechanism` (`"cy14_ztor"`), h = 0 (`"none"`) or
  h(M) from a `[scenarios.depth_function]` table (`"magnitude_function"`;
  see [`DepthFunction`][pysmsim.scenario.DepthFunction]).
- `kappa0` and `stress`, when given, replace the kappa_0 of `[site]` and
  the stress model of `[source.stress]`. Kappa then does not depend on
  magnitude, as in SMSIM's loop drivers.
- A run has at most 1,000 periods and 1,000 frequencies.

See [`ScenarioGrid`][pysmsim.scenario.ScenarioGrid].

## Output

```toml
[output]
path = "results.csv"   # relative to the run file's folder
format = "csv"         # or "parquet", "hdf5", "col"
```

`pysmsim run -o` and `-f` replace these (see
[Command line](cli.md#output-formats)).

## Reading and writing run files in Python

```python
from pysmsim.io.config import dumps_config, load_config

config = load_config("examples/wna_bt15_example.toml")
config.params        # Params: the model
config.scenarios     # ScenarioGrid
config.output        # OutputSettings
text = dumps_config(config)   # back to TOML
```

`RunFile.model_json_schema()` in `pysmsim.io.config` gives the JSON schema
of the file, for editors that check TOML against a schema.
