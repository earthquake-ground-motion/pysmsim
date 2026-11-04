from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest

from pysmsim.io.config import ConfigError, dumps_config, load_config, parse_config
from pysmsim.params import (
    FiniteFaultModel,
    OscillatorCorrection,
    QModel,
    SpreadingModel,
)
from pysmsim.scenario import DistanceMetric, RjbToRrup
from pysmsim.spectrum.source import AtkinsonSilva2000, SingleCorner, TwoCornerAdditive
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
    assert params.finite_fault.model is FiniteFaultModel.TWO_LINES_SMOOTHED
    assert params.spreading.model is SpreadingModel.HINGED
    assert [s.r_start for s in params.spreading.segments] == [1.0, 40.0]
    assert params.q.model is QModel.PIECEWISE
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
    assert config.params.finite_fault.model is FiniteFaultModel.NONE
    assert config.params.filters.low_cut.fcut == 0.0


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
        (lambda d: d["q"].update(qr1=0.0), "qr1 > 0"),
        (lambda d: d["spreading"].update(segments=[]), "at least one segment"),
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
