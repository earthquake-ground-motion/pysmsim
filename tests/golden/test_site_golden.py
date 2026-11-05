"""Crustal amplification, kappa, fmax and the filters against the Fortran oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from oracle_files import (
    GOLDEN,
    SHIPPED_PATHS,
    SiteModels,
    read_factor_file,
    read_shipped_sites,
    read_site_cases,
)

from pysmsim.spectrum.filters import HIGH_CUT_MODELS, LowCutFilter
from pysmsim.spectrum.site import Site

pytestmark = pytest.mark.golden

# Agreement with the double-precision build; the differences are rounding
RTOL = 1e-12

SITE_CASES = read_site_cases()
SHIPPED_SITES = read_shipped_sites()


def _check(path: Path, models: SiteModels) -> None:
    golden = read_factor_file(path)
    m, f, col = golden.header["m"], golden.columns["f"], golden.columns
    site, filters = models.site, models.filters

    assert site.kappa(m) == pytest.approx(golden.header["kappa"], rel=RTOL, abs=1e-15)
    assert site.amplification(f) == pytest.approx(col["site"], rel=RTOL)
    assert site.diminution(f, m) == pytest.approx(col["kappa"] * col["fmax"], rel=RTOL)
    assert filters.low_cut.factor(f) == pytest.approx(col["lowcut"], rel=RTOL)
    assert filters.high_cut.factor(f) == pytest.approx(col["highcut"], rel=RTOL, abs=1e-300)
    assert filters.factor(f) == pytest.approx(col["lowcut"] * col["highcut"], rel=RTOL)


@pytest.mark.parametrize("path", sorted((GOLDEN / "site").glob("*.txt")), ids=lambda p: p.stem)
def test_synthetic_site_cases(path: Path) -> None:
    _check(path, SITE_CASES[path.name.split(".M")[0]])


@pytest.mark.parametrize("path", sorted((GOLDEN / "factors").glob("*.txt")), ids=lambda p: p.stem)
def test_shipped_params_files(path: Path) -> None:
    stem = path.name.split(".M")[0]
    models = SHIPPED_SITES[stem]
    _check(path, models)
    # dimin is the Q path factor times the site diminution
    golden = read_factor_file(path)
    m, rrup, f = golden.header["m"], golden.header["rrup"], golden.columns["f"]
    path_models = SHIPPED_PATHS[stem]
    rps = path_models.finite_fault.rps(rrup, m)
    dimin = path_models.q.attenuation(f, m, rps, rrup) * models.site.diminution(f, m)
    assert dimin == pytest.approx(golden.columns["dimin"], rel=RTOL)


def test_shipped_site_lines_cover_the_factor_files() -> None:
    stems = {p.name.split(".M")[0] for p in (GOLDEN / "factors").glob("*.txt")}
    assert stems == set(SHIPPED_SITES)


def test_all_high_cut_shapes_covered() -> None:
    shapes = {type(c.filters.high_cut) for c in SITE_CASES.values()}
    assert shapes == set(HIGH_CUT_MODELS.values())


def test_site_cases_cover_the_options() -> None:
    sites: list[Site] = [c.site for c in SITE_CASES.values()]
    assert any(len(s.amp_frequencies) == 1 for s in sites)
    assert any(s.dkappa_dmag != 0 for s in sites)
    assert any(s.fmax > 0 and s.kappa0 == 0 for s in sites)
    low_cuts: list[LowCutFilter] = [c.filters.low_cut for c in SITE_CASES.values()]
    assert {lc.order for lc in low_cuts if lc.fcut > 0} == {2, 4, 6}


def test_low_cut_nslope_quirk() -> None:
    # SMSIM rounds nslope 6 down to 4 (order 2); pysmsim only accepts
    # orders that need no rounding, so the case is built with order 2
    case = SITE_CASES["s_lowcut_quirk_nslope6"]
    assert case.nslope == 6
    assert case.filters.low_cut.order == 2
    golden = read_factor_file(GOLDEN / "site" / "s_lowcut_quirk_nslope6.M6.0.txt")
    f = golden.columns["f"]
    nslope6 = 1.0 / (1.0 + (case.filters.low_cut.fcut / f) ** 6)
    assert not np.allclose(golden.columns["lowcut"], nslope6, rtol=1e-3)
    with pytest.raises(ValueError, match="even number"):
        LowCutFilter(fcut=0.2, order=3)
