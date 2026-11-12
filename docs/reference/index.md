# API reference

These pages are generated from the docstrings. Every physics step is a public
function or a small dataclass, so any of them can be imported and called on
its own from a notebook.

| Package | What it holds |
| --- | --- |
| [Parameters and scenarios](params.md) | `pysmsim.params` (all model parameters) and `pysmsim.scenario` (the scenario grid) |
| [Fourier spectrum](spectrum.md) | `pysmsim.spectrum`: source, stress, finite fault, spreading, Q, site, filters, responses and the spectrum itself |
| [Random vibration](rv.md) | `pysmsim.rv`: spectral moments, peak factors, durations and `rv_motion` |
| [Time domain](td.md) | `pysmsim.td`: time grid, window, noise, series, measures, oscillators and `td_motion` |
| [Drivers](drivers.md) | `pysmsim.drivers`: loops over scenarios, combined runs and single-scenario details |
| [Input and output](io.md) | `pysmsim.io`: run files, SMSIM files, BT15 tables and result files |

Units are cgs: cm, s and g/cm³, with distances in km, stress in bars and
g = 981 cm/s².
