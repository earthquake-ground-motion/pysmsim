from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest

from pysmsim.io.config import ConfigError, dumps_config, load_config, parse_config
from pysmsim.params import OscillatorCorrection
from pysmsim.scenario import DistanceMetric, RjbToRrup
from pysmsim.spectrum.attenuation import PiecewiseQ, SimpleQ, Stafford2022Q
from pysmsim.spectrum.filters import NoHighCut, QuarterCosineHighCut, RaisedCosineHighCut
from pysmsim.spectrum.finite_fault import (
    Generalized,
    NoFiniteFault,
    RpsCombination,
    TwoLinesSmoothed,
)
from pysmsim.spectrum.source import AtkinsonSilva2000, SingleCorner, TwoCornerAdditive
from pysmsim.spectrum.spreading import AtkinsonBoore2014, Hinged, StaffordOptimal
from pysmsim.spectrum.stress import HoerlStress, LogLinearStress

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"


@pytest.fixture
def example_data() -> dict[str, Any]:
    with EXAMPLE.open("rb") as f:
        return tomllib.load(f)


def test_load_example() -> None:
    config = load_config(EXAMPLE)
    params = config.params
    assert params.title == "Western North America params"
    assert params.medium.density == 2.72
    assert params.source == SingleCorner(
        stress=LogLinearStress(stress_ref=250.0, magnitude_ref=6.5)
    )
    assert params.finite_fault == TwoLinesSmoothed(
        c1=-1.72, c2=0.43, c3=-0.405, c4=0.235, delta_m=2.0
    )
    assert isinstance(params.spreading, Hinged)
    assert [s.r_start for s in params.spreading.segments] == [1.0, 40.0]
    assert params.q == PiecewiseQ(qr1=180.0, s1=0.45, qr2=180.0, s2=0.45, c_q=3.5)
    assert len(params.site.amp_frequencies) == 25
    assert params.rv.osc_correction is OscillatorCorrection.BT15_WNA

    grid = config.scenarios
    assert grid.magnitudes == (4.0, 6.0, 8.0)
    assert grid.distances == (10.0,)
    assert grid.distance_metric is DistanceMetric.RJB
    assert grid.rjb_to_rrup is RjbToRrup.CY14_ZTOR
    assert len(grid.periods) == 200
    assert grid.periods[0] == pytest.approx(0.01)
    assert grid.periods[-1] == pytest.approx(100.0)
    assert grid.kappa0 == (0.02, 0.05)


def test_round_trip() -> None:
    config = load_config(EXAMPLE)
    text = dumps_config(config)
    assert parse_config(tomllib.loads(text)) == config


def test_defaults_fill_optional_sections(example_data: dict[str, Any]) -> None:
    for section in ("finite_fault", "filters", "rv", "td", "output"):
        del example_data[section]
    config = parse_config(example_data)
    assert config.params.finite_fault == NoFiniteFault()
    assert config.params.filters.low_cut.fcut == 0.0
    assert config.params.filters.high_cut == NoHighCut()


def test_dataclasses_are_frozen() -> None:
    config = load_config(EXAMPLE)
    with pytest.raises(AttributeError):
        config.params.medium.density = 3.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: d.update(schema_version=2), "schema_version must be 1"),
        (lambda d: d["medium"].update(densty=2.7), "medium.densty"),
        (lambda d: d["source"].update(model="one_corner"), "source: Input tag 'one_corner'"),
        (lambda d: d["source"].update(pf_b=2.0), "source.single_corner.pf_b"),
        (lambda d: d["source"].pop("stress"), "source.single_corner.stress"),
        (lambda d: d["source"]["stress"].update(c1=1.0), "stress.log_linear.c1"),
        (lambda d: d["medium"].update(density=-1.0), "medium.density"),
        (lambda d: d["q"].update(qr1=0.0), "q.piecewise.qr1: Input should be greater than 0"),
        (
            lambda d: d["spreading"].update(segments=[]),
            "spreading.hinged.segments: Tuple should have at least 1 item",
        ),
        (lambda d: d["path_duration"].update(durations=[0.0, 2.4]), "same length"),
        (lambda d: d["site"].update(amp_frequencies=[1.0] * 25), "strictly increasing"),
        (lambda d: d["filters"]["low_cut"].update(order=3), "even number"),
        (lambda d: d["scenarios"].update(periods=[0.0, 1.0]), "periods must be positive"),
        (lambda d: d["source"]["stress"].update(stress_ref=0.0), "stress_ref: Input should be"),
    ],
)
def test_invalid_files_name_the_problem(
    example_data: dict[str, Any], change: Any, expected: str
) -> None:
    data = copy.deepcopy(example_data)
    change(data)
    with pytest.raises(ConfigError, match=expected):
        parse_config(data)


def test_bad_toml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("schema_version = \n")
    with pytest.raises(ConfigError, match=r"bad\.toml: not valid TOML"):
        load_config(path)


def test_log_spaced_and_linear_values(example_data: dict[str, Any]) -> None:
    example_data["scenarios"]["distances"] = {"start": 1.0, "stop": 100.0, "count": 3}
    example_data["scenarios"]["magnitudes"] = {"start": 5.0, "stop": 6.0, "step": 0.5}
    grid = parse_config(example_data).scenarios
    assert grid.distances == pytest.approx((1.0, 10.0, 100.0))
    assert grid.magnitudes == pytest.approx((5.0, 5.5, 6.0))


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"model": "atkinson_silva_2000"}, AtkinsonSilva2000()),
        (
            {
                "model": "two_corner_additive",
                "log10_fa": [2.181, -0.496, 0.0],
                "log10_eps": [0.605, -0.255, 0.0],
                "fortran_compat": True,
                "stress": {"model": "hoerl", "c1": 26900.0, "c2": -5.057, "c3": 0.652},
            },
            TwoCornerAdditive(
                stress=HoerlStress(c1=26900.0, c2=-5.057, c3=0.652),
                log10_fa=(2.181, -0.496, 0.0),
                log10_eps=(0.605, -0.255, 0.0),
                fortran_compat=True,
            ),
        ),
    ],
)
def test_source_models_by_name(
    example_data: dict[str, Any], table: dict[str, Any], expected: object
) -> None:
    example_data["source"] = table
    config = parse_config(example_data)
    assert config.params.source == expected
    # and back through a TOML file
    assert parse_config(tomllib.loads(dumps_config(config))) == config


@pytest.mark.parametrize(
    ("section", "table", "expected"),
    [
        ("finite_fault", {"model": "none"}, NoFiniteFault()),
        (
            "finite_fault",
            {
                "model": "generalized",
                "combination": "sum",
                "c1": -1.72,
                "c2": 0.43,
                "c3": -0.405,
                "c4": 0.235,
                "h_ratio": 0.8939,
            },
            Generalized(
                c1=-1.72,
                c2=0.43,
                c3=-0.405,
                c4=0.235,
                h_ratio=0.8939,
                combination=RpsCombination.SUM,
            ),
        ),
        (
            "spreading",
            {"model": "stafford_optimal", "gamma1": -1.1611, "r_t": 50.0, "gamma2": -0.5},
            StaffordOptimal(gamma1=-1.1611, r_t=50.0, gamma2=-0.5),
        ),
        (
            "spreading",
            {"model": "atkinson_boore_2014", "h": 10.0, "fortran_compat": True},
            AtkinsonBoore2014(h=10.0, fortran_compat=True),
        ),
        (
            "q",
            {
                "model": "stafford_2022",
                "q0": 205.4,
                "eta_a": 0.6884,
                "eta_b": 0.1354,
                "eta_g": 5.1278,
            },
            Stafford2022Q(q0=205.4, eta_a=0.6884, eta_b=0.1354, eta_g=5.1278),
        ),
        ("q", {"model": "simple", "q0": 180.0, "eta": 0.45}, SimpleQ(q0=180.0, eta=0.45)),
        ("q", {"model": "simple", "q0": 180.0, "c_q": 3.6}, SimpleQ(q0=180.0, eta=0.0, c_q=3.6)),
    ],
)
def test_path_models_by_name(
    example_data: dict[str, Any], section: str, table: dict[str, Any], expected: object
) -> None:
    example_data[section] = table
    config = parse_config(example_data)
    assert getattr(config.params, section) == expected
    assert parse_config(tomllib.loads(dumps_config(config))) == config


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: d["spreading"].update(gamma1=-1.0), "spreading.hinged.gamma1"),
        (lambda d: d["q"].update(q0=200.0), "q.piecewise.q0"),
        (
            lambda d: d["finite_fault"].update(h_ratio=0.9),
            "finite_fault.two_lines_smoothed.h_ratio",
        ),
        (lambda d: d["finite_fault"].update(model="three_lines"), "finite_fault: Input tag"),
    ],
)
def test_path_keys_of_other_models_rejected(
    example_data: dict[str, Any], change: Any, expected: str
) -> None:
    change(example_data)
    with pytest.raises(ConfigError, match=expected):
        parse_config(example_data)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"model": "none"}, NoHighCut()),
        (
            {"model": "raised_cosine", "f1": 25.0, "f2": 50.0},
            RaisedCosineHighCut(f1=25.0, f2=50.0),
        ),
        (
            {"model": "quarter_cosine", "f1": 5.0, "f2": 40.0, "eta": 0.5},
            QuarterCosineHighCut(f1=5.0, f2=40.0, eta=0.5),
        ),
    ],
)
def test_high_cut_models_by_name(
    example_data: dict[str, Any], table: dict[str, Any], expected: object
) -> None:
    example_data["filters"]["high_cut"] = table
    config = parse_config(example_data)
    assert config.params.filters.high_cut == expected
    assert parse_config(tomllib.loads(dumps_config(config))) == config


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"model": "none", "f1": 25.0}, "filters.high_cut.none.f1"),
        (
            {"model": "raised_cosine", "f1": 25.0},
            "filters.high_cut.raised_cosine.f2: Field required",
        ),
        ({"model": "raised_cosine", "f1": 50.0, "f2": 25.0}, "0 < f1 < f2"),
        ({"model": "half_cosine"}, "filters.high_cut: Input tag"),
    ],
)
def test_invalid_high_cut_rejected(
    example_data: dict[str, Any], table: dict[str, Any], expected: str
) -> None:
    example_data["filters"]["high_cut"] = table
    with pytest.raises(ConfigError, match=expected):
        parse_config(example_data)
