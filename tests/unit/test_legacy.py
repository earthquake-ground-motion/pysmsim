"""Reading SMSIM input files: Fortran list-directed input, quirks and errors.

The comparison of every shipped and synthetic file with the oracle is in
tests/golden/test_legacy_golden.py.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from pysmsim.cli import main
from pysmsim.io.config import load_config, parse_config
from pysmsim.io.legacy import (
    LegacyFormatError,
    _format_toml,
    _FortranReader,
    convert,
    read_params,
    read_rv_loop_ctl,
)
from pysmsim.params import TDConfig
from pysmsim.spectrum.attenuation import Stafford2022Q
from pysmsim.spectrum.source import Joyner1984
from pysmsim.spectrum.stress import HoerlStress

SMSIM_FILES = Path(__file__).parents[1] / "data" / "smsim"
OFR = (SMSIM_FILES / "ofr.params").read_text().replace("\r\n", "\n")
CTL = (SMSIM_FILES / "tmrsk_loop_rv_drvr.ctl").read_text().replace("\r\n", "\n")
_COMMENT = re.compile(r"^ *!")


def ofr_with(**lines: str) -> str:
    """ofr.params with value lines replaced: ``ofr_with(v5="1 80 0 6.5 4 0")``."""
    out = []
    n = 0
    for line in OFR.splitlines():
        if not _COMMENT.match(line):
            n += 1
            line = lines.get(f"v{n}", line)
        out.append(line)
    return "\n".join(out) + "\n"


def write(tmp_path: Path, text: str, name: str = "x.params") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def ctl_with(old: str, new: str) -> str:
    assert old in CTL
    return CTL.replace(old, new, 1)


# --- Fortran list-directed input ----------------------------------------------------


def test_list_directed_values() -> None:
    reader = _FortranReader("1, 2.5d0 3*4\n 5 ! six\n7\n8 9 10\n", "t")
    assert reader.floats(6, "a") == [1.0, 2.5, 4.0, 4.0, 4.0, 5.0]  # continues on line 2
    assert reader.ints(1, "b") == [7]
    assert reader.floats(2, "c") == [8.0, 9.0]  # the rest of the line is ignored
    assert reader.pos == 4


def test_counted_values() -> None:
    reader = _FortranReader("3 1.0 2.0\n 3.0 extra\n2 4 5\n", "t")
    assert reader.counted("x") == [1.0, 2.0, 3.0]
    assert reader.counted("y") == [4.0, 5.0]
    with pytest.raises(LegacyFormatError, match="the number of z must be at least 1"):
        _FortranReader("0\n", "t").counted("z")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1 / 2\n", "'/' ends the values"),
        ("1 x\n", "'x' is not a number"),
        ("1.5 2\n", "'1.5' is not a whole number"),
        ("1\n", "the file ends while reading a"),
    ],
)
def test_list_directed_errors(text: str, message: str) -> None:
    reader = _FortranReader(text, "t")
    with pytest.raises(LegacyFormatError, match=message):
        reader.ints(2, "a")


def test_skipcmnt() -> None:
    reader = _FortranReader("!a\n|b\n  ! c\n1\n2\n", "t")
    reader.skipcmnt()
    assert reader.ints(1, "x") == [1]
    assert reader.notes == []
    reader.skipcmnt()  # always skips one line, here a value
    assert reader.pos == 5
    assert "SMSIM skips this line, which is not a comment ('2')" in reader.notes[0]


# --- Parameter files ----------------------------------------------------------------


def test_read_params_values(tmp_path: Path) -> None:
    params, notes = read_params(write(tmp_path, OFR))
    assert params.title == "Sample data file **** NOT FOR A PARTICULAR APPLICATION **"
    assert params.td.f_tb2te == 2.12 and params.td.dt == 0.005
    assert notes == []
    # Carriage returns and \smsim\ in the .pars names, as in the distribution
    params_crlf, _ = read_params(SMSIM_FILES / "ofr.params")
    assert params_crlf == params


def test_quirk_notes(tmp_path: Path) -> None:
    text = ofr_with(
        v4=" 2 2.0 1.0 0.0 0.0",
        v5=" 2 26900.0 -5.057 0.652 0.5 0.0",
        v15=" -2.0 205.4 0.6884 0.1354 5.1278 0.0 0.0 0.0 3.5",
        v30=" 0.1 6",
    )
    params, notes = read_params(write(tmp_path, text))
    assert isinstance(params.source, Joyner1984)
    assert isinstance(params.source.stress, HoerlStress)
    assert isinstance(params.q, Stafford2022Q)
    assert params.filters.low_cut.order == 2
    assert len(notes) == 3
    assert "only when fr1 is exactly -1" in notes[0]
    assert "SMSIM uses the log-linear stress formula" in notes[1]
    assert "nslope = 6 is not 4, 8, 12, ...; SMSIM uses 4" in notes[2]


def test_missing_td_lines(tmp_path: Path) -> None:
    lines = OFR.splitlines()
    cut = next(i for i, line in enumerate(lines) if line.startswith("!WINDOW PARAMS"))
    params, notes = read_params(write(tmp_path, "\n".join(lines[:cut]) + "\n"))
    assert params.td == TDConfig()
    assert notes == ["no time-domain lines: the [td] section has pysmsim's defaults"]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ({"v1": "01/01/20"}, "has the date '01/01/20'; pysmsim reads 02/25/21"),
        ({"v4": " 14 2.0 1.0 0.0 0.0"}, "source number 14 is not one of SMSIM's"),
        ({"v5": " 4 80.0 0.0 6.5 4.0 0.0"}, "stress model 4 is not one of SMSIM's"),
        ({"v7": " 3 3 -1.72 0.43 -0.405 0.235 2.0"}, "iflag_f_ff = 3 is not 0, 1 or 2"),
        ({"v7": " 1 4 -1.72 0.43 -0.405 0.235 2.0"}, "nlines_f_ff = 4 is not"),
        ({"v8": " 8"}, "geometrical spreading option 8 is not 0 to 7"),
        ({"v31": " 3 25.0 50.0 1.0"}, "itype_hcfilt = 3 is not 0, 1 or 2"),
        ({"v32": " 10.0 0.00001 0.001 6"}, "osc_crrctn = 6 is not 0 to 5"),
        ({"v32": " 10.0 0.00001 0.001 3", "v33": " my_table.pars"}, "uses the table 'my_table"),
        ({"v35": " 2 0.05 0.2 0.05 2.12 1.0"}, "indxwind and iran_type must be 0 or 1"),
        ({"v3": " 2.8 -3.6 0.71 0.55 2.0"}, "x.params: "),  # a check of the dataclass
    ],
)
def test_params_errors(tmp_path: Path, lines: dict[str, str], message: str) -> None:
    with pytest.raises(LegacyFormatError, match=re.escape(message)):
        read_params(write(tmp_path, ofr_with(**lines)))


def test_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(LegacyFormatError, match="cannot read the file"):
        read_params(tmp_path / "missing.params")


# --- Control files ------------------------------------------------------------------


def test_read_ctl() -> None:
    control = read_rv_loop_ctl(SMSIM_FILES / "tmrsk_loop_rv_drvr.ctl")
    assert control.col_file == "tmrsk_loop_rv_drvr.col"
    s = control.scenarios
    assert s["periods"] == [0.01, 0.1, 0.2, 1.0, 2.0]
    assert not s["pga"] and not s["pgv"]  # the Y line is for log-spaced periods only
    assert s["magnitudes"] == {"start": 4.0, "stop": 8.0, "step": 4.0}
    assert s["kappa0"] == {"start": 0.0025, "stop": 0.16, "count": 7}
    assert s["distances"] == {"start": 1.0, "stop": 400.0, "count": 20}
    assert s["stress"] == {"start": 1.0, "stop": 1000.0, "count": 4}
    assert s["rjb_to_rrup"] == "cy14_ztor" and s["mechanism"] == "strike_slip"
    assert control.notes == []


def test_ctl_individual_values(tmp_path: Path) -> None:
    text = ctl_with(" 5 0.01, 0.1, 0.2, 1.0, 2.0", " 6 -1 0 -2 -3 0.1 1.0")
    text = text.replace(" 0\n!if individual magnitudes", " 1\n!if individual magnitudes", 1)
    text = text.replace(" 4.0 4 8.0", " 2 5.0 6.5")
    text = text.replace("\n 0\n!if from params file", "\n -1\n!if from params file")
    text = text.replace(" 2\n! Mech", " 1\n! Mech").replace("\n 1\n!Case 1", "\n 3\n!Case 1")
    text = text.replace(" 0\n!if log-spaced stresses", " -1\n!if log-spaced stresses")
    control = read_rv_loop_ctl(write(tmp_path, text, "x.ctl"))
    s = control.scenarios
    assert s["periods"] == [0.1, 1.0] and s["pga"] and s["pgv"]
    assert s["magnitudes"] == [5.0, 6.5]
    assert "kappa0" not in s and "stress" not in s
    assert s["rjb_to_rrup"] == "magnitude_function" and s["mechanism"] == "reverse"
    assert control.notes == [
        "periods -3, -2 (Wood-Anderson and WWSSN instruments) are left out: pysmsim's RV "
        "method does not compute them yet"
    ]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("03/03/23", "06/07/16", "has the date '06/07/16'"),
        (" 5 0.01, 0.1, 0.2, 1.0, 2.0", " 1001 " + "1.0 " * 1001, "1001 periods"),
        (" 20 1.0 400.0", " 20 0.0 400.0", "need a count of at least 1 and a start above 0"),
        (" 2\n! Mech", " 3\n! Mech", "h4rjb2rrup case 3 is not 0, 1 or 2"),
        ("\n 1\n!Case 1", "\n 4\n!Case 1", "mechanism 4 is not 1, 2 or 3"),
    ],
)
def test_ctl_errors(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(LegacyFormatError, match=re.escape(message)):
        read_rv_loop_ctl(write(tmp_path, ctl_with(old, new), "x.ctl"))


def test_ctl_log_spaced_single_value(tmp_path: Path) -> None:
    control = read_rv_loop_ctl(write(tmp_path, ctl_with(" 20 1.0 400.0", " 1 30.0 400.0"), "x.ctl"))
    assert control.scenarios["distances"] == [30.0]


# --- Run files ----------------------------------------------------------------------


def test_convert(tmp_path: Path) -> None:
    params_path = write(tmp_path, ofr_with(v29=" 25.0 0.03 0.01 6.0"))
    ctl = ctl_with("\n 0\n!if from params file", "\n -1\n!if from params file")
    text, notes = convert(params_path, write(tmp_path, ctl, "x.ctl"))
    assert text.startswith("# Converted by pysmsim from x.params and x.ctl.\n#\n# Notes:\n")
    assert "SMSIM's loop drivers ignore dkappadmag" in notes[0]
    assert all(len(line) <= 88 for line in text.splitlines())
    config = parse_config(tomllib.loads(text))
    assert config.params == read_params(params_path)[0]
    assert config.output.path == "tmrsk_loop_rv_drvr.csv"
    assert len(config.scenarios.periods) == 5


def test_convert_without_ctl(tmp_path: Path) -> None:
    text, notes = convert(write(tmp_path, OFR, "ofr.params"))
    assert notes == [
        "no control file given: [scenarios] holds example values; edit them before running"
    ]
    config = parse_config(tomllib.loads(text))
    assert config.output.path == "ofr.csv"
    assert config.scenarios.magnitudes == (4.0, 5.0, 6.0, 7.0, 8.0)


def test_convert_invalid_scenarios(tmp_path: Path) -> None:
    ctl = write(tmp_path, ctl_with(" 5 0.01, 0.1, 0.2, 1.0, 2.0", " 1 -2"), "x.ctl")
    with pytest.raises(LegacyFormatError, match="the converted run file is not valid"):
        convert(write(tmp_path, OFR), ctl)


def test_format_toml() -> None:
    doc = {
        "a": 1,
        "s": {
            "short": [1.0, 2.0],
            "long": [float(i) for i in range(40)],
            "inline": {"start": 1.0, "stop": 2.0, "count": 3},
            "rows": [{"x": 1.0}, {"x": 2.0}],
            "sub": {"model": "m", "values": [1.0]},
        },
        "only_subs": {"one": {"k": "a" * 80}},
    }
    text = _format_toml(doc)
    assert tomllib.loads(text) == doc
    assert "short = [1.0, 2.0]\n" in text
    assert "inline = { start = 1.0, stop = 2.0, count = 3 }\n" in text
    assert "rows = [\n    { x = 1.0 },\n    { x = 2.0 },\n]" in text
    assert "[s.sub]" in text and "[only_subs.one]" in text and "[only_subs]" not in text
    assert max(len(line) for line in text.splitlines()) <= 88


# --- The command --------------------------------------------------------------------


def test_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    params = SMSIM_FILES / "ofr.params"
    ctl = SMSIM_FILES / "tmrsk_loop_rv_drvr.ctl"
    assert main(["convert", str(params), str(ctl)]) == 0
    captured = capsys.readouterr()
    assert captured.out == convert(params, ctl)[0]
    assert captured.err == ""

    out = tmp_path / "run.toml"
    assert main(["convert", str(params), "-o", str(out)]) == 0
    assert load_config(out).params == read_params(params)[0]
    err = capsys.readouterr().err
    assert "note: no control file given" in err and f"Wrote {out}" in err

    assert main(["convert", str(params), "-o", str(out)]) == 1
    assert "exists; use --force to replace it" in capsys.readouterr().err
    assert main(["convert", str(params), "-o", str(out), "--force", "-q"]) == 0
    assert capsys.readouterr().err == ""

    assert main(["convert", str(tmp_path / "missing.params")]) == 1
    assert "cannot read the file" in capsys.readouterr().err
    assert main(["convert", str(params), "-o", str(tmp_path / "no" / "dir.toml")]) == 1
    assert "cannot write" in capsys.readouterr().err
