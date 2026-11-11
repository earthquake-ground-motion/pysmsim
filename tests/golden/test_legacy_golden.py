"""Converting SMSIM input files, checked against the oracle's own readers.

The shipped ``.params`` files and the synthetic cases of ``oracle/inputs``
are read with :func:`pysmsim.io.legacy.read_params` and compared with the
parameters the golden tests use (``oracle_files.py``), which are built
independently from the same lines. The synthetic files are made here as
``oracle/make_golden.sh`` makes them, by replacing value lines of
``ofr.params``.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from oracle_files import (
    ORACLE_INPUTS,
    SHIPPED_MEDIA,
    read_path_cases,
    read_rv_cases,
    read_site_cases,
    read_source_cases,
    shipped_params,
)
from test_rv_golden import EXAMPLE_GRID, GRID

from pysmsim.io.config import parse_config
from pysmsim.io.legacy import convert, read_params
from pysmsim.params import Params, TDConfig

pytestmark = pytest.mark.golden

SMSIM_FILES = Path(__file__).parents[1] / "data" / "smsim"
OFR = (SMSIM_FILES / "ofr.params").read_text()
EXAMPLE_PARAMS = "wna.scf.bt15acr_fff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms"

_COMMENT = re.compile(r"^ *!")


def _as_oracle(params: Params) -> Params:
    """The parameters with the title and [td] the oracle's readers use."""
    return replace(params, title="golden", td=TDConfig())


def _cases(name: str) -> Iterator[list[str]]:
    """The fields of each case line of oracle/inputs/<name>."""
    for line in (ORACLE_INPUTS / name).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            yield line.split("|")


def _edit(edit: Callable[[int, str, list[str]], bool], keep_comment: Callable[[int], bool]) -> str:
    """ofr.params with value lines replaced, as the awk scripts of make_golden.sh do.

    ``edit(n, line, out)`` gets each value line with its number ``n`` (from
    1) and returns True when it wrote the replacement into ``out``.
    """
    out: list[str] = []
    n = 0
    for line in OFR.splitlines():
        if _COMMENT.match(line):
            if keep_comment(n):
                out.append(line)
            continue
        n += 1
        if not edit(n, line, out):
            out.append(line)
    return "\n".join(out) + "\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.params"
    path.write_text(text)
    return path


# --- Parameter files ----------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(SHIPPED_MEDIA))
def test_shipped_params(stem: str) -> None:
    params, notes = read_params(SMSIM_FILES / f"{stem}.params")
    assert _as_oracle(params) == shipped_params(stem)
    assert notes == []
    assert params.title and params.td.nsims == 800


@pytest.mark.parametrize("fields", list(_cases("source_cases.txt")), ids=lambda f: f[0].strip())
def test_source_cases(tmp_path: Path, fields: list[str]) -> None:
    name, shape, stress, scaling = (f.strip() for f in fields)
    lines = {4: shape, 5: stress, 6: scaling}

    def edit(n: int, line: str, out: list[str]) -> bool:
        if n in lines:
            out.append(f" {lines[n]}")
            return True
        return False

    params, _ = read_params(_write(tmp_path, name, _edit(edit, lambda n: True)))
    expected = replace(shipped_params("ofr"), source=read_source_cases()[name])
    assert _as_oracle(params) == expected


@pytest.mark.parametrize("fields", list(_cases("path_cases.txt")), ids=lambda f: f[0].strip())
def test_path_cases(tmp_path: Path, fields: list[str]) -> None:
    name, ff, spreading, q = (f.strip() for f in fields)

    def edit(n: int, line: str, out: list[str]) -> bool:
        if n == 7:
            out.append(f" {ff}")
        elif n == 8:  # all the spreading lines, which replace value lines 8 to 14
            for i, part in enumerate(spreading.split(";"), start=1):
                if i in (2, 3):
                    out.append("!")
                out.append(f" {part}")
        elif n == 15:
            out.append(f" {q}")
        return 7 <= n <= 15

    text = _edit(edit, lambda n: n < 8 or n >= 14)
    params, _ = read_params(_write(tmp_path, name, text))
    case = read_path_cases()[name]
    expected = replace(
        shipped_params("ofr"), finite_fault=case.finite_fault, spreading=case.spreading, q=case.q
    )
    assert _as_oracle(params) == expected


@pytest.mark.parametrize("fields", list(_cases("site_cases.txt")), ids=lambda f: f[0].strip())
def test_site_cases(tmp_path: Path, fields: list[str]) -> None:
    name, amps, dimin, low_cut, high_cut = (f.strip() for f in fields)
    points = amps.split(";")
    namps = 0

    def edit(n: int, line: str, out: list[str]) -> bool:
        nonlocal namps
        if n == 23:
            out.append(f" {len(points)}")
            out.extend(f" {p}" for p in points)
            namps = int(line.split()[0])
        elif 23 < n <= 23 + namps:
            pass
        elif n in (24 + namps, 25 + namps, 26 + namps):
            out.append(" " + (dimin, low_cut, high_cut)[n - 24 - namps])
        else:
            return False
        return True

    params, notes = read_params(_write(tmp_path, name, _edit(edit, lambda n: True)))
    case = read_site_cases()[name]
    expected = replace(shipped_params("ofr"), site=case.site, filters=case.filters)
    assert _as_oracle(params) == expected
    # SMSIM rounds nslope down to a multiple of 4, and the note says so
    assert any("nslope" in note for note in notes) == (
        case.nslope != 2 * params.filters.low_cut.order
    )


@pytest.mark.parametrize("fields", list(_cases("rv_cases.txt")), ids=lambda f: f[0].strip())
def test_rv_cases(tmp_path: Path, fields: list[str]) -> None:
    name, dimin, rv_line, damping = (f.strip() for f in fields)

    def edit(n: int, line: str, out: list[str]) -> bool:
        if n in (29, 32):
            out.append(f" {dimin if n == 29 else rv_line}")
            return True
        return False

    params_path = _write(tmp_path, name, _edit(edit, lambda n: True))
    ctl = (ORACLE_INPUTS / "tmrsk_loop_rv_drvr.ctl").read_text()
    ctl = ctl.replace("@PARAMS@", params_path.name).replace("@COL@", f"{name}.col")
    ctl = ctl.replace("!FractionalDamping\n 0.05", f"!FractionalDamping\n {damping}")
    ctl_path = tmp_path / f"{name}.ctl"
    ctl_path.write_text(ctl)

    text, _ = convert(params_path, ctl_path)
    config = parse_config(tomllib.loads(text))
    case = read_rv_cases()[name]
    assert _as_oracle(config.params) == case.params
    assert config.scenarios == replace(GRID, damping=case.damping)
    assert config.output.path == f"{name}.csv"


# --- The shipped example ------------------------------------------------------------


def test_example_run() -> None:
    text, notes = convert(
        SMSIM_FILES / f"{EXAMPLE_PARAMS}.params", SMSIM_FILES / "example.tmrsk_loop_rv_drvr.ctl"
    )
    assert notes == []
    config = parse_config(tomllib.loads(text))
    assert _as_oracle(config.params) == shipped_params(EXAMPLE_PARAMS)
    grid = config.scenarios
    assert grid.periods == pytest.approx(EXAMPLE_GRID.periods, rel=1e-12)
    assert replace(grid, periods=EXAMPLE_GRID.periods) == EXAMPLE_GRID
    assert math.isclose(grid.periods[-1], 100.0)
