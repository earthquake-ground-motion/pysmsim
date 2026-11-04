"""Source spectrum against the Fortran oracle (double-precision build)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from oracle_files import (
    GOLDEN,
    SHIPPED_SOURCES,
    SYNTHETIC_SHEAR_VELOCITY,
    read_factor_file,
    read_source_cases,
)

from pysmsim.spectrum.source import SOURCE_MODELS, SourceSpectrum, TwoCornerAdditive

pytestmark = pytest.mark.golden

# Agreement with the double-precision build; the differences are rounding
RTOL = 1e-12

SOURCE_CASES = read_source_cases()

# Cases where SMSIM's quirk changes the result; they match only with
# fortran_compat=True
QUIRK_FILES = {"s12_quirk_eps_ref.M3.0.txt"}


def _check(path: Path, source: SourceSpectrum, beta: float) -> None:
    golden = read_factor_file(path)
    h = golden.header
    terms = source.terms(h["m"], beta)

    assert terms.m0 == pytest.approx(h["m0"], rel=RTOL)
    assert terms.fa == pytest.approx(h["fa"], rel=RTOL)
    assert terms.fb == pytest.approx(h["fb"], rel=RTOL)
    assert terms.eps == pytest.approx(h["am0b_m0"], rel=RTOL)
    if terms.stress is None:  # corners fixed by magnitude: SMSIM prints 0
        assert h["stress"] == 0.0
    else:
        assert terms.stress == pytest.approx(h["stress"], rel=RTOL)
    assert source.duration(terms) == pytest.approx(h["dur_source"], rel=RTOL)

    shape = source.shape(golden.columns["f"], terms)
    assert shape == pytest.approx(golden.columns["source"], rel=RTOL)


@pytest.mark.parametrize("path", sorted((GOLDEN / "source").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_source_cases(path: Path) -> None:
    source = SOURCE_CASES[path.name.split(".M")[0]]
    if path.name in QUIRK_FILES:
        assert isinstance(source, TwoCornerAdditive)
        source = replace(source, fortran_compat=True)
    _check(path, source, SYNTHETIC_SHEAR_VELOCITY)


@pytest.mark.parametrize("path", sorted((GOLDEN / "factors").glob("*.txt")), ids=lambda p: p.stem)
def test_shipped_params_files(path: Path) -> None:
    beta, source = SHIPPED_SOURCES[path.name.split(".M")[0]]
    _check(path, source, beta)


def test_all_source_models_covered() -> None:
    assert {type(source) for source in SOURCE_CASES.values()} == set(SOURCE_MODELS.values())


def test_quirk_changes_the_result() -> None:
    path = GOLDEN / "source" / "s12_quirk_eps_ref.M3.0.txt"
    source = SOURCE_CASES["s12_quirk_eps_ref"]
    fixed = source.terms(3.0, SYNTHETIC_SHEAR_VELOCITY)
    golden = read_factor_file(path).header
    # pysmsim: eps = 10**(0.605 - 0.255*(3 - 2)) > 1, so M 3 is below the
    # magnitude where eps = 1 and the model is single-corner
    assert fixed.eps == 1.0
    assert fixed.fa == fixed.fb
    assert golden["am0b_m0"] > 1.0
