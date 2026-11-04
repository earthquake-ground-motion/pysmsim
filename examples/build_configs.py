"""Build and modify pysmsim run configurations from Python.

Three ways to do it, all ending in a validated RunConfig that can be saved
as a TOML run file:

1. Build the dataclasses directly (no files involved).
2. Load an existing run file and change parts of it with
   ``dataclasses.replace`` (the dataclasses are frozen, so changes make new
   objects and the original is left untouched).
3. Edit the file as a plain dictionary and validate it with
   ``parse_config``, which gives the same checks and error messages as
   reading a TOML file.
"""

import tomllib
from dataclasses import replace
from pathlib import Path

from pysmsim.io.config import RunConfig, dumps_config, load_config, parse_config
from pysmsim.params import (
    FiniteFault,
    FiniteFaultModel,
    Medium,
    OscillatorCorrection,
    Params,
    PathDuration,
    Q,
    RVConfig,
    Site,
    Spreading,
    SpreadingSegment,
)
from pysmsim.scenario import OutputSettings, ScenarioGrid
from pysmsim.spectrum.source import SingleCorner
from pysmsim.spectrum.stress import LogLinearStress

HERE = Path(__file__).parent

# --- 1. Build a configuration from scratch ----------------------------------

params = Params(
    title="Generic ENA rock",
    medium=Medium(density=2.8, shear_velocity=3.6),
    source=SingleCorner(stress=LogLinearStress(stress_ref=150.0, magnitude_ref=6.5)),
    finite_fault=FiniteFault(
        model=FiniteFaultModel.TWO_LINES_SMOOTHED,
        c1=-1.8276,
        c2=0.43,
        c3=-0.5126,
        c4=0.235,
        delta_m=2.0,
    ),
    spreading=Spreading(
        segments=(
            SpreadingSegment(r_start=1.0, a=-1.0),
            SpreadingSegment(r_start=50.0, a=-0.5),
        )
    ),
    q=Q(qr1=410.0, s1=0.5, qr2=410.0, s2=0.5),
    path_duration=PathDuration(
        distances=(0.0, 15.0, 35.0, 50.0), durations=(0.0, 2.6, 17.5, 25.1), slope_beyond=0.111
    ),
    site=Site(amp_frequencies=(0.01, 1.0, 10.0), amplifications=(1.0, 1.5, 2.5), kappa0=0.006),
    rv=RVConfig(osc_correction=OscillatorCorrection.BT15_ENA),
)
grid = ScenarioGrid(
    magnitudes=(5.0, 6.0, 7.0),
    distances=(10.0, 30.0, 100.0),
    periods=(0.1, 0.2, 1.0),
)
new_config = RunConfig(params=params, scenarios=grid, output=OutputSettings(path="ena.csv"))

# Invalid values fail as soon as the object is built, e.g.
#   Site(amp_frequencies=(1.0, 0.5), amplifications=(1.0, 1.0))
#   -> ValueError: amp_frequencies must be strictly increasing

# --- 2. Load an existing file and modify it ---------------------------------

base = load_config(HERE / "wna_bt15_example.toml")

# Change one value deep inside: replace each level on the way down.
higher_kappa = replace(
    base,
    params=replace(base.params, site=replace(base.params.site, kappa0=0.05)),
)

# Make a family of variants, for example a stress sweep, each with its own output.
# The source is a single-corner model; give it a new stress model each time.
source = base.params.source
assert isinstance(source, SingleCorner)
variants = {
    stress: replace(
        base,
        params=replace(
            base.params,
            source=replace(source, stress=LogLinearStress(stress_ref=stress, magnitude_ref=6.5)),
        ),
        scenarios=replace(base.scenarios, stress=None),  # use the params stress
        output=replace(base.output, path=f"wna_stress_{stress:g}.csv"),
    )
    for stress in (50.0, 100.0, 200.0)
}

# The original is unchanged: frozen dataclasses cannot be modified in place.
assert base.params.site.kappa0 == 0.035

# --- 3. Edit the file contents as a dictionary -------------------------------
# Handy for many small edits, or when the values come from another tool
# (a spreadsheet, a database): change keys, then validate the whole thing.

with (HERE / "wna_bt15_example.toml").open("rb") as f:
    data = tomllib.load(f)
data["site"]["kappa0"] = 0.05
data["scenarios"]["magnitudes"] = [5.0, 5.5, 6.0]
data["scenarios"]["distances"] = {"start": 1.0, "stop": 200.0, "count": 10}
from_dict = parse_config(data)  # raises ConfigError naming any bad key

# --- Save any of them as a TOML run file -------------------------------------

if __name__ == "__main__":
    out = Path("generated_configs")
    out.mkdir(exist_ok=True)
    (out / "ena_generic.toml").write_text(dumps_config(new_config))
    (out / "wna_kappa_0.05.toml").write_text(dumps_config(higher_kappa))
    for stress, config in variants.items():
        (out / f"wna_stress_{stress:g}.toml").write_text(dumps_config(config))

    # A saved file reads back to an equal configuration.
    assert load_config(out / "ena_generic.toml") == new_config
    print(f"wrote {len(list(out.glob('*.toml')))} run files to {out}/")
