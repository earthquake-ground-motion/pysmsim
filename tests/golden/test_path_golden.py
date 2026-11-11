"""Finite-fault factor, geometrical spreading and Q against the Fortran oracle."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from oracle_files import (
    GOLDEN,
    SHIPPED_PATHS,
    PathModels,
    read_factor_file,
    read_path_cases,
)

from pysmsim.spectrum.attenuation import Q_MODELS, SimpleQ
from pysmsim.spectrum.finite_fault import FINITE_FAULT_MODELS
from pysmsim.spectrum.spreading import (
    SPREADING_MODELS,
    AtkinsonBoore2014,
    AtkinsonBoore2014R10,
    FrequencyIndependentSpreading,
)

pytestmark = pytest.mark.golden

# Agreement with the double-precision build; the differences are rounding
RTOL = 1e-12

PATH_CASES = read_path_cases()

# Radiation constant of ofr.params (rho 2.8, beta 3.6, prtitn 0.71, rtp 0.55,
# fs 2) for r_ref = 1 km, as in SMSIM's const_am0_gsprd
OFR_CONST = 0.71 * 0.55 * 2.0 * 1e-20 / (4.0 * math.pi * 2.8 * 3.6**3)


def _compat(models: PathModels) -> PathModels:
    """Reproduce SMSIM's 1.570796 for pi/2 in the AB14 spreading models."""
    if isinstance(models.spreading, AtkinsonBoore2014 | AtkinsonBoore2014R10):
        return replace(models, spreading=replace(models.spreading, fortran_compat=True))
    return models


def _check(path: Path, models: PathModels) -> None:
    golden = read_factor_file(path)
    h = golden.header
    m, rrup, f = h["m"], h["rrup"], golden.columns["f"]

    assert models.finite_fault.h(m) == pytest.approx(h["f_ff"], rel=RTOL, abs=0.0)
    rps = models.finite_fault.rps(rrup, m)
    assert rps == pytest.approx(h["rps"], rel=RTOL)

    spreading = models.spreading
    gsprd = spreading.factor(f, rps, rrup, m)
    assert gsprd == pytest.approx(golden.columns["gsprd"], rel=RTOL)
    if isinstance(spreading, FrequencyIndependentSpreading):
        assert spreading.value(rps, rrup, m) == pytest.approx(h["gsprd"], rel=RTOL)

    q = models.q
    assert q.q(f, m) == pytest.approx(golden.columns["q"], rel=RTOL)
    assert q.attenuation(f, m, rps, rrup) == pytest.approx(golden.columns["q_path"], rel=RTOL)


@pytest.mark.parametrize("path", sorted((GOLDEN / "path").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_path_cases(path: Path) -> None:
    models = _compat(PATH_CASES[path.name.split(".M")[0]])
    _check(path, models)
    # The radiation constant divides by the spreading model's r_ref
    const = read_factor_file(path).header["const"]
    assert const == pytest.approx(OFR_CONST / models.spreading.reference_distance(), rel=RTOL)


@pytest.mark.parametrize("path", sorted((GOLDEN / "factors").glob("*.txt")), ids=lambda p: p.stem)
def test_shipped_params_files(path: Path) -> None:
    _check(path, SHIPPED_PATHS[path.name.split(".M")[0]])


def test_all_path_models_covered() -> None:
    models = list(PATH_CASES.values()) + list(SHIPPED_PATHS.values())
    assert {type(m.finite_fault) for m in models} == set(FINITE_FAULT_MODELS.values())
    assert {type(m.spreading) for m in models} == set(SPREADING_MODELS.values())
    # SimpleQ has no SMSIM form; the unit tests compare it with PiecewiseQ
    assert {type(m.q) for m in models} == set(Q_MODELS.values()) - {SimpleQ}


@pytest.mark.parametrize("case", ["p_gs_ab14", "p_gs_ab14_r10"])
def test_ab14_half_pi_quirk(case: str) -> None:
    # SMSIM uses 1.570796 for pi/2; the difference is tiny but measurable
    spreading = PATH_CASES[case].spreading
    compat = replace(spreading, fortran_compat=True)  # type: ignore[type-var]
    f = np.logspace(-2, 2, 21)
    for r in (0.5, 5.0, 30.0):
        ratio = spreading.factor(f, r, r, 6.0) / compat.factor(f, r, r, 6.0)
        assert np.max(np.abs(ratio - 1.0)) < 1e-6
        assert np.max(np.abs(ratio - 1.0)) > 1e-12
