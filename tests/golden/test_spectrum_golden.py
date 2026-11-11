"""The whole Fourier spectrum (spect_amp) and the responses against the Fortran oracle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from oracle_files import (
    GOLDEN,
    SHIPPED_MEDIA,
    SHIPPED_PATHS,
    SHIPPED_SOURCES,
    params_from_parts,
    read_factor_file,
    read_path_cases,
    read_shipped_sites,
    read_site_cases,
    read_source_cases,
    shipped_params,
)

from pysmsim.params import Params
from pysmsim.spectrum.fas import fourier_spectrum, scenario_terms, spectrum_factors
from pysmsim.spectrum.response import (
    GroundMotion,
    MotionQuantity,
    Oscillator,
    WoodAnderson,
    WWSSNShortPeriod,
)
from pysmsim.spectrum.source import TwoCornerAdditive
from pysmsim.spectrum.spreading import AtkinsonBoore2014, AtkinsonBoore2014R10

pytestmark = pytest.mark.golden

# Agreement with the double-precision build; the differences are rounding
RTOL = 1e-12

OFR = "ofr"
SHIPPED_SITES = read_shipped_sites()


def _stem(path: Path) -> str:
    return path.name.split(".M")[0]


def _check(path: Path, params: Params) -> None:
    golden = read_factor_file(path)
    h, col = golden.header, golden.columns
    m, rrup, f = h["m"], h["rrup"], col["f"]

    terms = scenario_terms(params, m, rrup)
    assert terms.radiation_constant == pytest.approx(h["const"], rel=RTOL)
    assert terms.rps == pytest.approx(h["rps"], rel=RTOL)

    factors = spectrum_factors(f, params, m, rrup)
    assert factors.product() == pytest.approx(col["product"], rel=RTOL, abs=1e-300)
    fds = fourier_spectrum(f, params, m, rrup)
    assert fds == pytest.approx(col["fds"], rel=RTOL, abs=1e-300)
    acc = fourier_spectrum(f, params, m, rrup, GroundMotion(MotionQuantity.ACCELERATION))
    assert acc == pytest.approx(col["fas"], rel=RTOL, abs=1e-300)


@pytest.mark.parametrize("path", sorted((GOLDEN / "factors").glob("*.txt")), ids=lambda p: p.stem)
def test_shipped_params_files(path: Path) -> None:
    _check(path, shipped_params(_stem(path)))


def _ofr(**parts: object) -> Params:
    kwargs: dict[str, object] = {
        "medium": SHIPPED_MEDIA[OFR],
        "source": SHIPPED_SOURCES[OFR][1],
        "path": SHIPPED_PATHS[OFR],
        "site": SHIPPED_SITES[OFR],
    }
    kwargs.update(parts)
    return params_from_parts(**kwargs)  # type: ignore[arg-type]


SOURCE_CASES = read_source_cases()
PATH_CASES = read_path_cases()
SITE_CASES = read_site_cases()


@pytest.mark.parametrize("path", sorted((GOLDEN / "source").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_source_cases(path: Path) -> None:
    source = SOURCE_CASES[_stem(path)]
    if "quirk" in path.name and isinstance(source, TwoCornerAdditive):
        source = replace(source, fortran_compat=True)
    _check(path, _ofr(source=source))


@pytest.mark.parametrize("path", sorted((GOLDEN / "path").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_path_cases(path: Path) -> None:
    models = PATH_CASES[_stem(path)]
    if isinstance(models.spreading, AtkinsonBoore2014 | AtkinsonBoore2014R10):
        models = replace(models, spreading=replace(models.spreading, fortran_compat=True))
    _check(path, _ofr(path=models))


@pytest.mark.parametrize("path", sorted((GOLDEN / "site").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_site_cases(path: Path) -> None:
    _check(path, _ofr(site=SITE_CASES[_stem(path)]))


# --- Responses ------------------------------------------------------------


def _read_responses(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    header: dict[str, float] = {}
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("!"):
            key, sep, value = line[1:].partition("=")
            if sep and key.strip() != "params":
                header[key.strip()] = float(value)
        elif not names:
            names = line.split()
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    return header, {n: data[:, i] for i, n in enumerate(names)}


@pytest.mark.parametrize("path", sorted((GOLDEN / "response").glob("*.txt")), ids=lambda p: p.stem)
def test_responses(path: Path) -> None:
    header, col = _read_responses(path)
    params = shipped_params(_stem(path))
    m, rrup, f = header["m"], header["rrup"], col["f"]
    responses = {
        "dis": GroundMotion(),
        "vel": GroundMotion(MotionQuantity.VELOCITY),
        "acc": GroundMotion(MotionQuantity.ACCELERATION),
        "wa": WoodAnderson(),
        "wwssn_sp": WWSSNShortPeriod(),
    }
    for name in col:
        if name.startswith("osc_"):
            responses[name] = Oscillator(period=float(name[4:]), damping=header["damp"])
    assert len(responses) == len(col) - 1
    for name, response in responses.items():
        expected = col[name]
        assert fourier_spectrum(f, params, m, rrup, response) == pytest.approx(
            expected, rel=RTOL, abs=1e-300
        ), name
