"""Read and write TOML run files.

One TOML file describes a whole run: the model parameters (what an SMSIM
``.params`` file holds) and the scenarios to compute (what a loop-driver
``.ctl`` file holds). Its sections are::

    schema_version = 1
    title = "..."
    [medium]  [source]  [source.stress]  [finite_fault]  [spreading]
    [q]  [path_duration]  [site]  [filters.low_cut]  [filters.high_cut]
    [rv]  [td]  [scenarios]  [output]

The keys in each section are the field names of the matching dataclass in
:mod:`pysmsim.params` or :mod:`pysmsim.scenario`, and options are the enum
values (for example ``osc_correction = "bt15_wna"`` in ``[rv]``). In
``[source]``, ``[source.stress]``, ``[finite_fault]``, ``[spreading]``,
``[q]`` and ``[filters.high_cut]``, ``model`` names one of the classes in the modules of
:mod:`pysmsim.spectrum` (for example ``model = "single_corner"``), and the
other keys are the fields of that class; a key the named model does not use
is an error. In ``[scenarios]``,
``magnitudes`` may be a table ``{start, stop, step}`` and ``distances``,
``periods``, ``frequencies``, ``kappa0`` and ``stress`` a table ``{start, stop, count}`` for
log-spaced values, as in the SMSIM control files. ``examples/`` has complete
files. The file is checked with pydantic: unknown keys, wrong types and
inconsistent settings are reported with the key that caused them.

Use :func:`load_config` to read a file and :func:`dumps_config` to write
one.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pysmsim.params import (
    Medium,
    NoiseDistribution,
    OscillatorCorrection,
    Params,
    PathDuration,
    RVConfig,
    TDConfig,
    WindowShape,
)
from pysmsim.scenario import (
    DepthFunction,
    DistanceMetric,
    Mechanism,
    OutputFormat,
    OutputSettings,
    RjbToRrup,
    ScenarioGrid,
    check_spectral_points,
)
from pysmsim.spectrum.attenuation import Q_MODELS, QModel
from pysmsim.spectrum.filters import HIGH_CUT_MODELS, Filters, HighCutFilter, LowCutFilter
from pysmsim.spectrum.finite_fault import FINITE_FAULT_MODELS, FiniteFaultFactor, RpsCombination
from pysmsim.spectrum.site import Site
from pysmsim.spectrum.source import SOURCE_MODELS, SourceSpectrum
from pysmsim.spectrum.spreading import SPREADING_MODELS, GeometricalSpreading, SpreadingSegment
from pysmsim.spectrum.stress import STRESS_MODELS, StressModel

__all__ = [
    "SCHEMA_VERSION",
    "ConfigError",
    "RunConfig",
    "RunFile",
    "dumps_config",
    "load_config",
    "parse_config",
]

SCHEMA_VERSION = 1
"""Version of the run-file layout that this module reads and writes."""


class ConfigError(ValueError):
    """A run file is not valid. The message names each offending key."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Everything a run needs: model parameters, scenarios and output."""

    params: Params
    scenarios: ScenarioGrid
    output: OutputSettings = field(default_factory=OutputSettings)


# --- Schema ---------------------------------------------------------------
# One pydantic model per section. The field names and defaults match the
# dataclasses; the models check types and ranges and then build the
# dataclasses, whose own checks catch inconsistent combinations.


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class MediumSection(_Section):
    density: float = Field(gt=0, description="Density near the source (g/cm^3)")
    shear_velocity: float = Field(gt=0, description="Shear-wave velocity (km/s)")
    partition: float = Field(0.707, gt=0)
    radiation: float = Field(0.55, gt=0)
    free_surface: float = Field(2.0, gt=0)

    def build(self) -> Medium:
        return Medium(**self.model_dump())


# The source and stress models: one section per model class, picked by the
# value of ``model`` (a discriminated union), so that each accepts only the
# keys of its own class.


class _ModelSection(_Section):
    """A section that builds the class named by its ``model`` key."""

    registry: ClassVar[Mapping[str, type[Any]]]
    model: str

    def build(self) -> Any:
        kwargs = {
            key: _build(getattr(self, key)) for key in type(self).model_fields if key != "model"
        }
        return self.registry[self.model](**kwargs)


def _build(value: Any) -> Any:
    """Build nested sections (alone or in a tuple); leave other values as they are."""
    if isinstance(value, tuple):
        return tuple(_build(v) for v in value)
    build = getattr(value, "build", None)
    return build() if isinstance(value, _Section) and build is not None else value


class _StressSection(_ModelSection):
    registry = STRESS_MODELS


class LogLinearStressSection(_StressSection):
    model: Literal["log_linear"]
    stress_ref: float = Field(gt=0, description="Stress (bars) at magnitude_ref")
    dlog10_stress_dm: float = 0.0
    magnitude_ref: float = 6.5


class HoerlStressSection(_StressSection):
    model: Literal["hoerl"]
    c1: float
    c2: float
    c3: float


class Stafford2022StressSection(_StressSection):
    model: Literal["stafford_2022"]
    c1: float
    c2: float


StressSection = Annotated[
    LogLinearStressSection | HoerlStressSection | Stafford2022StressSection,
    Field(discriminator="model"),
]
"""The ``[source.stress]`` table."""


class _SourceSection(_ModelSection):
    registry = SOURCE_MODELS


class _WeightedSourceSection(_SourceSection):
    duration_weight_fa: float = Field(0.5, ge=0)
    duration_weight_fb: float = Field(0.5, ge=0)


class SingleCornerSection(_WeightedSourceSection):
    model: Literal["single_corner"]
    stress: StressSection
    pf_a: float = Field(2.0, gt=0)
    pd_a: float = Field(1.0, gt=0)


class Joyner1984Section(_WeightedSourceSection):
    model: Literal["joyner_1984"]
    stress: StressSection
    magnitude_hinge: float
    fb_over_fa: float = Field(4.0, gt=0)


class TwoCornerMultiplicativeSection(_WeightedSourceSection):
    model: Literal["two_corner_multiplicative"]
    stress: StressSection
    log10_fa: tuple[float, float, float]
    pf_a: float = Field(2.0, gt=0)
    pd_a: float = Field(0.5, gt=0)
    pf_b: float = Field(2.0, gt=0)
    pd_b: float = Field(0.5, gt=0)


class TwoCornerAdditiveSection(_WeightedSourceSection):
    model: Literal["two_corner_additive"]
    stress: StressSection
    log10_fa: tuple[float, float, float]
    log10_eps: tuple[float, float, float]
    pf_a: float = Field(2.0, gt=0)
    pd_a: float = Field(1.0, gt=0)
    pf_b: float = Field(2.0, gt=0)
    pd_b: float = Field(1.0, gt=0)
    fortran_compat: bool = False


# The models whose corner frequencies are fixed functions of magnitude have
# no parameters besides the duration weights


class Atkinson1993Section(_WeightedSourceSection):
    model: Literal["atkinson_1993"]


class AtkinsonSilva1997Section(_WeightedSourceSection):
    model: Literal["atkinson_silva_1997"]


class Haddon1996Section(_WeightedSourceSection):
    model: Literal["haddon_1996"]


class AtkinsonBoore1998CaliforniaSection(_WeightedSourceSection):
    model: Literal["atkinson_boore_1998_california"]


class BoatwrightChoySection(_WeightedSourceSection):
    model: Literal["boatwright_choy"]


class JoynerENASection(_WeightedSourceSection):
    model: Literal["joyner_ena"]


class AtkinsonSilva2000Section(_WeightedSourceSection):
    model: Literal["atkinson_silva_2000"]


class Atkinson2005Section(_WeightedSourceSection):
    model: Literal["atkinson_2005"]


class JiArchuleta2020Section(_SourceSection):
    model: Literal["ji_archuleta_2020"]


SourceSection = Annotated[
    SingleCornerSection
    | Joyner1984Section
    | Atkinson1993Section
    | AtkinsonSilva1997Section
    | Haddon1996Section
    | AtkinsonBoore1998CaliforniaSection
    | BoatwrightChoySection
    | JoynerENASection
    | AtkinsonSilva2000Section
    | Atkinson2005Section
    | TwoCornerMultiplicativeSection
    | TwoCornerAdditiveSection
    | JiArchuleta2020Section,
    Field(discriminator="model"),
]
"""The ``[source]`` table."""


class _FiniteFaultSection(_ModelSection):
    registry = FINITE_FAULT_MODELS


class NoFiniteFaultSection(_FiniteFaultSection):
    model: Literal["none"]


class _CombinedSection(_FiniteFaultSection):
    combination: RpsCombination = RpsCombination.QUADRATURE


class OneLineSection(_CombinedSection):
    model: Literal["one_line"]
    c1: float
    c2: float


class TwoLinesSection(_CombinedSection):
    model: Literal["two_lines"]
    c1: float
    c2: float
    c3: float
    c4: float


class TwoLinesSmoothedSection(_CombinedSection):
    model: Literal["two_lines_smoothed"]
    c1: float
    c2: float
    c3: float
    c4: float
    delta_m: float = Field(gt=0, description="Width of the transition (magnitude units)")


class GeneralizedSection(_CombinedSection):
    model: Literal["generalized"]
    c1: float
    c2: float
    c3: float
    c4: float
    h_ratio: float = Field(gt=0, description="h at the transition over the lines' value")


class Stafford2022FiniteFaultSection(_CombinedSection):
    model: Literal["stafford_2022"]
    h_alpha: float
    h_beta: float
    h_gamma: float
    h_delta: float
    h_epsilon: float


FiniteFaultSection = Annotated[
    NoFiniteFaultSection
    | OneLineSection
    | TwoLinesSection
    | TwoLinesSmoothedSection
    | GeneralizedSection
    | Stafford2022FiniteFaultSection,
    Field(discriminator="model"),
]
"""The ``[finite_fault]`` table."""


class SegmentSection(_Section):
    r_start: float = Field(gt=0, description="Start of the segment (km)")
    a: float
    b: float = 0.0
    m_ref: float = 0.0

    def build(self) -> SpreadingSegment:
        return SpreadingSegment(**self.model_dump())


class _SpreadingSection(_ModelSection):
    registry = SPREADING_MODELS


class HingedSection(_SpreadingSection):
    model: Literal["hinged"]
    segments: tuple[SegmentSection, ...] = Field(min_length=1)
    r_ref: float = Field(1.0, gt=0)


class Atkinson2011Section(_SpreadingSection):
    model: Literal["atkinson_2011"]


class BooreTrialSection(_SpreadingSection):
    model: Literal["boore_trial"]
    r1: float = Field(gt=1, description="Distance where the far slope starts (km)")
    slope_lf: float
    slope_hf: float
    slope_far: float
    ft1: float = Field(gt=0)
    ft2: float = Field(gt=0)


class Atkinson2012Section(_SpreadingSection):
    model: Literal["atkinson_2012"]
    r_ref: float = Field(1.0, gt=0)


class AtkinsonBoore2014Section(_SpreadingSection):
    model: Literal["atkinson_boore_2014"]
    h: float = Field(gt=0, lt=50)
    r_ref: float = Field(1.0, gt=0)
    fortran_compat: bool = False


class AtkinsonBoore2014R10Section(_SpreadingSection):
    model: Literal["atkinson_boore_2014_r10"]
    h: float = Field(gt=0, lt=50)
    r_ref: float = Field(1.0, gt=0)
    fortran_compat: bool = False


class StaffordGeneralizedSection(_SpreadingSection):
    model: Literal["stafford_generalized"]
    gamma1: float
    r_t: float = Field(gt=0, description="Transition distance (km)")
    gamma2: float
    g_ratio: float = Field(gt=0)
    r_ref: float = Field(1.0, gt=0)


class StaffordOptimalSection(_SpreadingSection):
    model: Literal["stafford_optimal"]
    gamma1: float
    r_t: float = Field(gt=0, description="Transition distance (km)")
    gamma2: float
    r_ref: float = Field(1.0, gt=0)


SpreadingSection = Annotated[
    HingedSection
    | Atkinson2011Section
    | BooreTrialSection
    | Atkinson2012Section
    | AtkinsonBoore2014Section
    | AtkinsonBoore2014R10Section
    | StaffordGeneralizedSection
    | StaffordOptimalSection,
    Field(discriminator="model"),
]
"""The ``[spreading]`` table."""


class _QSection(_ModelSection):
    registry = Q_MODELS
    c_q: float = Field(3.5, gt=0, description="Velocity in the path factor (km/s)")


class PiecewiseQSection(_QSection):
    model: Literal["piecewise"]
    qr1: float = Field(gt=0)
    s1: float = 0.0
    fr1: float = Field(1.0, gt=0)
    qr2: float = Field(gt=0)
    s2: float = 0.0
    fr2: float = Field(1.0, gt=0)
    ft1: float = Field(1.0, gt=0)
    ft2: float = Field(1.0, gt=0)


class Stafford2022QSection(_QSection):
    model: Literal["stafford_2022"]
    q0: float = Field(gt=0)
    eta_a: float
    eta_b: float
    eta_g: float


class SimpleQSection(_QSection):
    model: Literal["simple"]
    q0: float = Field(gt=0, description="Q at 1 Hz")
    eta: float = 0.0


QSection = Annotated[
    PiecewiseQSection | Stafford2022QSection | SimpleQSection, Field(discriminator="model")
]
"""The ``[q]`` table."""


class PathDurationSection(_Section):
    distances: tuple[float, ...] = Field(min_length=1)
    durations: tuple[float, ...] = Field(min_length=1)
    slope_beyond: float = 0.0

    def build(self) -> PathDuration:
        return PathDuration(**self.model_dump())


class SiteSection(_Section):
    amp_frequencies: tuple[float, ...] = Field(min_length=1)
    amplifications: tuple[float, ...] = Field(min_length=1)
    kappa0: float = Field(0.0, ge=0)
    dkappa_dmag: float = 0.0
    kappa_magnitude_ref: float = 0.0
    fmax: float = Field(0.0, ge=0)

    def build(self) -> Site:
        return Site(**self.model_dump())


class LowCutSection(_Section):
    fcut: float = Field(0.0, ge=0, description="Corner frequency (Hz); 0 = no filter")
    order: int = Field(4, ge=2, description="Order of each of the two passes (even)")

    def build(self) -> LowCutFilter:
        return LowCutFilter(**self.model_dump())


class _HighCutSection(_ModelSection):
    registry = HIGH_CUT_MODELS


class NoHighCutSection(_HighCutSection):
    model: Literal["none"]


class _CosineTaperSection(_HighCutSection):
    f1: float = Field(gt=0, description="Frequency (Hz) where the taper starts")
    f2: float = Field(gt=0, description="Frequency (Hz) where the taper ends")
    eta: float = Field(1.0, gt=0, description="Power of the cosine")


class RaisedCosineHighCutSection(_CosineTaperSection):
    model: Literal["raised_cosine"]


class QuarterCosineHighCutSection(_CosineTaperSection):
    model: Literal["quarter_cosine"]


HighCutSection = Annotated[
    NoHighCutSection | RaisedCosineHighCutSection | QuarterCosineHighCutSection,
    Field(discriminator="model"),
]
"""The ``[filters.high_cut]`` table."""


class FiltersSection(_Section):
    low_cut: LowCutSection = Field(default_factory=LowCutSection)
    high_cut: HighCutSection = Field(default_factory=lambda: NoHighCutSection(model="none"))

    def build(self) -> Filters:
        return Filters(low_cut=self.low_cut.build(), high_cut=self.high_cut.build())


class RVSection(_Section):
    zup: float = Field(10.0, gt=0)
    eps_int: float = Field(1.0e-5, gt=0)
    amp_cutoff: float = Field(1.0e-3, gt=0, lt=1)
    osc_correction: OscillatorCorrection = OscillatorCorrection.BT15_WNA
    bt15_tables: tuple[str, str] = ("bt15_wna_acr", "bt15_ena_scr")

    def build(self) -> RVConfig:
        return RVConfig(**self.model_dump())


class TDSection(_Section):
    window: WindowShape = WindowShape.EXPONENTIAL
    taper: float = Field(0.05, ge=0, lt=1)
    eps_w: float = Field(0.2, gt=0, lt=1)
    eta_w: float = Field(0.05, gt=0, lt=1)
    f_tb2te: float = Field(2.0, gt=0)
    f_te_xtnd: float = Field(1.0, gt=0)
    dur_fctr: float = Field(1.3, gt=0)
    dt: float = Field(0.005, gt=0)
    tshift: float = Field(50.0, ge=0)
    seed: int = 123
    nsims: int = Field(800, ge=1)
    noise: NoiseDistribution = NoiseDistribution.GAUSSIAN

    def build(self) -> TDConfig:
        return TDConfig(**self.model_dump())


class DepthFunctionSection(_Section):
    m1: float = 3.75
    h1: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    m2: float = 7.5
    h2: float = 0.0

    def build(self) -> DepthFunction:
        return DepthFunction(**self.model_dump())


class LogSpacedSection(_Section):
    """``count`` values spaced evenly in log between ``start`` and ``stop``."""

    start: float = Field(gt=0)
    stop: float = Field(gt=0)
    count: int = Field(ge=1)

    def values(self) -> tuple[float, ...]:
        if self.count == 1:
            return (self.start,)
        ratio = self.stop / self.start
        return tuple(self.start * ratio ** (i / (self.count - 1)) for i in range(self.count))


class LinearSection(_Section):
    """Values from ``start`` to ``stop`` (inclusive) in steps of ``step``."""

    start: float
    stop: float
    step: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> LinearSection:
        if self.stop < self.start:
            raise ValueError("stop must not be less than start")
        return self

    def values(self) -> tuple[float, ...]:
        count = round((self.stop - self.start) / self.step) + 1
        return tuple(self.start + i * self.step for i in range(count))


_Values = tuple[float, ...] | LogSpacedSection
"""A list of values, or a table ``{start, stop, count}`` for log spacing."""


def _expand(values: _Values | LinearSection) -> tuple[float, ...]:
    if isinstance(values, tuple):
        return values
    return values.values()


class ScenariosSection(_Section):
    magnitudes: tuple[float, ...] | LinearSection
    distances: _Values
    distance_metric: DistanceMetric = DistanceMetric.RJB
    rjb_to_rrup: RjbToRrup = RjbToRrup.CY14_ZTOR
    mechanism: Mechanism = Mechanism.STRIKE_SLIP
    depth_function: DepthFunctionSection = Field(default_factory=DepthFunctionSection)
    periods: _Values = ()
    pga: bool = True
    pgv: bool = True
    damping: float = Field(0.05, gt=0, lt=1)
    frequencies: _Values = ()
    kappa0: _Values | None = None
    stress: _Values | None = None

    @model_validator(mode="after")
    def _check_sizes(self) -> ScenariosSection:
        # Before the values are expanded, so that a huge count fails at once
        for name in ("periods", "frequencies"):
            values = getattr(self, name)
            count = values.count if isinstance(values, LogSpacedSection) else len(values)
            check_spectral_points(name, count)
        return self

    def build(self) -> ScenarioGrid:
        return ScenarioGrid(
            magnitudes=_expand(self.magnitudes),
            distances=_expand(self.distances),
            distance_metric=self.distance_metric,
            rjb_to_rrup=self.rjb_to_rrup,
            mechanism=self.mechanism,
            depth_function=self.depth_function.build(),
            periods=_expand(self.periods),
            pga=self.pga,
            pgv=self.pgv,
            damping=self.damping,
            frequencies=_expand(self.frequencies),
            kappa0=None if self.kappa0 is None else _expand(self.kappa0),
            stress=None if self.stress is None else _expand(self.stress),
        )


class OutputSection(_Section):
    path: str = "results.csv"
    format: OutputFormat = OutputFormat.CSV

    def build(self) -> OutputSettings:
        return OutputSettings(**self.model_dump())


class RunFile(_Section):
    """The whole run file. ``RunFile.model_json_schema()`` gives its JSON schema."""

    schema_version: Literal[1]
    title: str = ""
    medium: MediumSection
    source: SourceSection
    finite_fault: FiniteFaultSection = Field(
        default_factory=lambda: NoFiniteFaultSection(model="none")
    )
    spreading: SpreadingSection
    q: QSection
    path_duration: PathDurationSection
    site: SiteSection
    filters: FiltersSection = Field(default_factory=FiltersSection)
    rv: RVSection = Field(default_factory=RVSection)
    td: TDSection = Field(default_factory=TDSection)
    scenarios: ScenariosSection
    output: OutputSection = Field(default_factory=OutputSection)

    def build(self) -> RunConfig:
        """Build the :class:`RunConfig` dataclasses from the validated file."""
        params = Params(
            title=self.title,
            medium=self.medium.build(),
            source=self.source.build(),
            finite_fault=self.finite_fault.build(),
            spreading=self.spreading.build(),
            q=self.q.build(),
            path_duration=self.path_duration.build(),
            site=self.site.build(),
            filters=self.filters.build(),
            rv=self.rv.build(),
            td=self.td.build(),
        )
        return RunConfig(
            params=params, scenarios=self.scenarios.build(), output=self.output.build()
        )


# --- Reading and writing --------------------------------------------------


def parse_config(data: Mapping[str, Any]) -> RunConfig:
    """Build a :class:`RunConfig` from the parsed contents of a run file.

    Parameters
    ----------
    data : mapping
        The run file as nested dictionaries, as returned by ``tomllib``.

    Returns
    -------
    RunConfig

    Raises
    ------
    ConfigError
        If the contents do not follow the schema. The message lists every
        problem with the key it concerns.
    """
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SCHEMA_VERSION} (this version of pysmsim), got {version!r}"
        )
    try:
        return RunFile.model_validate(data).build()
    except ValidationError as err:
        lines = [
            f"  {'.'.join(str(p) for p in e['loc']) or '(top level)'}: {e['msg']}"
            for e in err.errors()
        ]
        raise ConfigError("invalid run file:\n" + "\n".join(lines)) from err
    except ValueError as err:  # a consistency check in a dataclass
        raise ConfigError(f"invalid run file: {err}") from err


def load_config(path: str | Path) -> RunConfig:
    """Read a TOML run file.

    Parameters
    ----------
    path : str or Path
        The run file.

    Returns
    -------
    RunConfig

    Raises
    ------
    ConfigError
        If the file is not valid TOML or does not follow the schema.
    """
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"{path}: not valid TOML: {err}") from err
    try:
        return parse_config(data)
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err


_MODEL_TYPES = (
    SourceSpectrum,
    StressModel,
    FiniteFaultFactor,
    GeometricalSpreading,
    QModel,
    HighCutFilter,
)
"""Classes written as a table that starts with their ``model`` name."""

_COMMON_KEYS = ("c_q", "duration_weight_fa", "duration_weight_fb")
"""Keys written after a model's own parameters."""


def _plain(value: Any) -> Any:
    """Turn dataclass contents into TOML-writable values."""
    if isinstance(value, _MODEL_TYPES):
        return _model_table(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name))
            for f in fields(value)
            if getattr(value, f.name) is not None
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _model_table(obj: Any) -> dict[str, Any]:
    """Turn a model object into a table that starts with its ``model`` name."""
    table: dict[str, Any] = {"model": obj.name}
    # The model's own parameters first, then those every model of its kind has
    names = [f.name for f in fields(obj)]
    common = [n for n in names if n in _COMMON_KEYS]
    for name in [n for n in names if n not in common] + common:
        table[name] = _plain(getattr(obj, name))
    return table


def dumps_config(config: RunConfig) -> str:
    """Write a :class:`RunConfig` as the text of a TOML run file.

    Reading the text back with :func:`parse_config` gives an equal
    ``RunConfig``.
    """
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    doc.update(_plain(config.params))
    doc["scenarios"] = _plain(config.scenarios)
    doc["output"] = _plain(config.output)
    return tomli_w.dumps(doc)
