"""The pysmsim command and the results writers."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pysmsim import __version__
from pysmsim.cli import main
from pysmsim.drivers.run import run
from pysmsim.io.config import dumps_config, load_config
from pysmsim.io.hdf5 import read_hdf5, read_td_series
from pysmsim.io.results import (
    COL_SENTINEL,
    format_for_path,
    results_table,
    write_col,
    write_results,
)
from pysmsim.scenario import DistanceMetric, OutputFormat, ScenarioGrid

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
CONFIG = load_config(EXAMPLE)
GRID = ScenarioGrid(
    magnitudes=(5.0, 7.0),
    distances=(10.0, 50.0),
    periods=(0.1, 1.0),
    frequencies=(0.1, 1.0, 10.0),
    stress=(50.0, 100.0),
)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A small run file in its own folder, writing results.csv there."""
    path = tmp_path / "run.toml"
    config = type(CONFIG)(params=CONFIG.params, scenarios=GRID, output=CONFIG.output)
    text = dumps_config(config).replace('path = "wna_bt15_example.csv"', 'path = "out/res.csv"')
    path.write_text(text)
    return path


# --- Results writers ----------------------------------------------------------------


def test_format_for_path() -> None:
    assert format_for_path("a.CSV") is OutputFormat.CSV
    assert format_for_path("a/b.h5") is OutputFormat.HDF5
    assert format_for_path("a.hdf5") is OutputFormat.HDF5
    assert format_for_path("a.parquet") is OutputFormat.PARQUET
    assert format_for_path("a.col") is OutputFormat.COL
    assert format_for_path("a.txt") is None


def test_results_table_and_formats(tmp_path: Path) -> None:
    result = run(CONFIG.params, GRID)
    table = results_table(result)
    assert result.fas is not None and result.rv is not None
    assert list(table.columns) == [*result.scenarios, *result.rv, *result.fas]
    assert len(table) == 8
    write_results(result, tmp_path / "r.csv", "csv")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "r.csv"), table)
    write_results(result, tmp_path / "r.parquet", OutputFormat.PARQUET)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "r.parquet"), table)
    write_results(result, tmp_path / "r.h5", "hdf5")
    pd.testing.assert_frame_equal(results_table(read_hdf5(tmp_path / "r.h5")), table)
    with pytest.raises(ValueError, match="'xls' is not a valid OutputFormat"):
        write_results(result, tmp_path / "r.xls", "xls")


def test_write_col(tmp_path: Path) -> None:
    result = run(CONFIG.params, GRID)
    assert result.rv is not None
    path = tmp_path / "r.col"
    write_col(result, path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith(" Parameter file: ")
    assert lines[1].split()[0] == "damp"
    rows = np.array([[float(v) for v in line.split()] for line in lines[2:]])
    # One row per scenario and measure: PGV, PGA, then the periods
    assert rows.shape == (8 * 4, 14)
    np.testing.assert_allclose(rows[:4, 1], [-1.0, 0.0, 0.1, 1.0])
    np.testing.assert_allclose(rows[:4, 2], [-1.0, 999.99, 10.0, 1.0])
    assert (rows[:, 10] == 1).all()  # numsource of the single-corner model
    np.testing.assert_allclose(rows[0::4, 13], result.rv.pgv, rtol=1e-6)
    np.testing.assert_allclose(rows[0::4, 12], result.rv.pgv, rtol=1e-6)
    np.testing.assert_allclose(rows[1::4, 13], result.rv.pga, rtol=1e-6)
    np.testing.assert_allclose(rows[1::4, 12], result.rv.pga / 981.0, rtol=1e-6)
    np.testing.assert_allclose(rows[3::4, 13], result.rv["psa(1)"], rtol=1e-6)
    np.testing.assert_allclose(rows[:16, 11], 50.0)


def test_write_col_sentinels_and_errors(tmp_path: Path) -> None:
    # M 9.5 is beyond the BT15 table, so PSA is NaN there
    grid = ScenarioGrid(
        magnitudes=(6.0, 9.5),
        distances=(10.0,),
        distance_metric=DistanceMetric.RRUP,
        pgv=False,
        pga=False,
        periods=(1.0,),
        frequencies=(1.0,),
    )
    result = run(CONFIG.params, grid)
    assert result.rv is not None
    assert not math.isnan(result.rv["psa(1)"][0]) and math.isnan(result.rv["psa(1)"][1])
    write_col(result, tmp_path / "r.col")
    rows = [
        [float(v) for v in line.split()]
        for line in (tmp_path / "r.col").read_text().splitlines()[2:]
    ]
    assert rows[0][4] == COL_SENTINEL  # Rjb, not given
    assert rows[0][-1] > 0
    assert rows[1][-2:] == [COL_SENTINEL, COL_SENTINEL]
    with pytest.raises(ValueError, match="needs the RV results"):
        write_col(run(CONFIG.params, grid, rv=False), tmp_path / "x.col")


# --- The command --------------------------------------------------------------------


def test_check(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(config_file), str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert f"{config_file}: OK" in out
    assert "scenarios: 8 (2 magnitudes x 2 distances x 2 stress values)" in out
    assert "FAS: 3 frequencies" in out
    assert "scenarios: 6 (3 magnitudes x 1 distances x 2 kappa_0 x 1 stress values)" in out


def test_check_reports_invalid_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(EXAMPLE.read_text().replace("damping = 0.05", "damping = 2.0"))
    missing = tmp_path / "missing.toml"
    assert main(["check", str(EXAMPLE), str(bad), str(missing)]) == 1
    captured = capsys.readouterr()
    assert f"{EXAMPLE}: OK" in captured.out
    assert f"{bad}: invalid" in captured.err and "scenarios.damping" in captured.err
    assert f"{missing}: cannot read the file" in captured.err
    assert captured.err.count(f"{bad}:") == 1
    assert main(["check", "-q", str(EXAMPLE)]) == 0
    assert capsys.readouterr().out == ""


def test_run_uses_the_output_section(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(config_file)]) == 0
    # The path of the run file is relative to its folder
    out = config_file.parent / "out" / "res.csv"
    table = pd.read_csv(out)
    expected = results_table(run(CONFIG.params, GRID))
    pd.testing.assert_frame_equal(table, expected)
    err = capsys.readouterr().err
    assert f"Wrote {out} (csv), 8 scenarios" in err


@pytest.mark.parametrize(
    ("name", "args", "fmt"),
    [
        ("r.h5", [], "hdf5"),
        ("r.parquet", [], "parquet"),
        ("r.col", [], "col"),
        ("r.dat", ["--format", "csv"], "csv"),
        ("r.dat", [], "csv"),  # unknown suffix: the run file's format
    ],
)
def test_run_output_options(
    config_file: Path, tmp_path: Path, name: str, args: list[str], fmt: str
) -> None:
    out = tmp_path / name
    assert main(["run", str(config_file), "-o", str(out), "-q", *args]) == 0
    assert out.exists()
    if fmt == "hdf5":
        assert read_hdf5(out).grid == GRID
    elif fmt == "parquet":
        assert len(pd.read_parquet(out)) == 8
    elif fmt == "csv":
        assert len(pd.read_csv(out)) == 8
    else:
        assert len(out.read_text().splitlines()) == 2 + 8 * 4


def test_run_options(config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "fas.h5"
    assert main(["run", str(config_file), "-o", str(out), "--no-rv", "-q"]) == 0
    assert read_hdf5(out).rv is None
    out = tmp_path / "compat.h5"
    assert main(["run", str(config_file), "-o", str(out), "--fortran-compat", "-q"]) == 0
    assert read_hdf5(out).fortran_compat
    assert capsys.readouterr().err == ""
    assert main(["run", str(config_file), "-o", str(tmp_path / "x.col"), "--no-rv"]) == 1
    assert "cannot be used with --no-rv" in capsys.readouterr().err


def test_run_td(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "td.toml"
    params = dataclasses.replace(
        CONFIG.params, td=dataclasses.replace(CONFIG.params.td, nsims=3, dt=0.01, tshift=5.0)
    )
    grid = dataclasses.replace(GRID, frequencies=(), stress=None)
    path.write_text(dumps_config(type(CONFIG)(params=params, scenarios=grid, output=CONFIG.output)))
    out = tmp_path / "td.h5"
    assert main(["run", str(path), "-o", str(out), "--td", "--no-rv"]) == 0
    err = capsys.readouterr().err
    assert "TD (--td): 3 simulations of each scenario, exponential window, dt 0.01 s" in err
    result = read_hdf5(out)
    assert result.rv is None and result.td is not None and len(result.td) == 4
    assert main(["run", str(path), "-o", str(tmp_path / "td.csv"), "--td", "-q"]) == 0
    assert {"pga", "td_pga", "td_psa(1)"} <= set(pd.read_csv(tmp_path / "td.csv").columns)
    series = tmp_path / "series.h5"
    assert main(["run", str(path), "-o", str(series), "--td", "--td-series", "2", "-q"]) == 0
    assert read_td_series(series).acceleration.shape[0] == 2
    with h5py.File(series) as h5:  # the defaults: whole series, 32-bit
        assert h5["td/series/acceleration"].dtype == np.float32
        assert h5["td/series/times"].attrs["tolerance"] == 0.0
    args = [
        "run",
        str(path),
        "-o",
        str(series),
        "--td",
        "--td-series",
        "1",
        "--series-tolerance",
        "1e-4",
        "--series-precision",
        "64",
    ]
    assert main([*args, "-q"]) == 0
    assert read_td_series(series).tolerance == 1e-4
    with h5py.File(series) as h5:
        assert h5["td/series/acceleration"].dtype == np.float64
    assert read_hdf5(series).td is not None
    assert main(["run", str(path), "-o", str(series), "--no-rv", "--td-series", "-q"]) == 1
    assert main(["run", str(path), "-o", str(tmp_path / "s.csv"), "--td-series"]) == 1
    assert "--td-series needs the hdf5 format" in capsys.readouterr().err
    assert main(["run", str(path), "-o", str(tmp_path / "x.col"), "--td"]) == 1
    assert "cannot be used with --no-rv or --td" in capsys.readouterr().err
    assert main(["check", str(path)]) == 0
    assert "TD (--td): 3 simulations" in capsys.readouterr().out


def test_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "td.toml"
    params = dataclasses.replace(
        CONFIG.params, td=dataclasses.replace(CONFIG.params.td, nsims=3, dt=0.01, tshift=5.0)
    )
    path.write_text(dumps_config(type(CONFIG)(params=params, scenarios=GRID, output=CONFIG.output)))
    out = tmp_path / "out"
    assert (
        main(
            ["scenario", str(path), "-m", "6", "-r", "20", "--td", "--series", "2", "-o", str(out)]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "M 6, Rrup 20 km: f_ff" in captured.out
    assert "Random vibration:" in captured.out and "Fourier spectra:" in captured.out
    assert "Time domain, 3 simulations" in captured.out
    assert "Wrote rv.csv, td.csv, td_series.csv, fas.csv" in captured.err
    rv = pd.read_csv(out / "rv.csv")
    assert list(rv.measure) == ["pgd", "pgv", "pga", "arias", "psa", "psa"]
    assert (rv.rrup == 20.0).all()
    assert len(pd.read_csv(out / "fas.csv")) == 3
    assert (pd.read_csv(out / "td.csv").nsims == 3).all()
    series = pd.read_csv(out / "td_series.csv")
    assert list(series.columns) == ["time", "acc", "vel", "dis"]

    # Rjb, converted as the run file says; and only the Fourier spectra
    assert (
        main(
            [
                "scenario",
                str(path),
                "-m",
                "6",
                "--rjb",
                "10",
                "--no-rv",
                "-q",
                "-o",
                str(tmp_path / "rjb"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    fas = pd.read_csv(tmp_path / "rjb" / "fas.csv")
    assert fas.rrup.iloc[0] > 10.0
    assert not (tmp_path / "rjb" / "rv.csv").exists()

    errors = {
        ("--series", "1"): "--series needs --td, -o",
        ("--td", "--series", "4", "-o", str(out)): "is more than the number of simulations",
        ("--nsims", "2"): "--nsims needs --td",
    }
    for extra, message in errors.items():
        assert main(["scenario", str(path), "-m", "6", "-r", "20", *extra]) == 1
        assert message in capsys.readouterr().err
    assert main(["scenario", str(path), "-m", "6", "-r", "0"]) == 1
    assert "must be positive" in capsys.readouterr().err
    no_fas = tmp_path / "no_fas.toml"
    grid = dataclasses.replace(GRID, frequencies=())
    no_fas.write_text(
        dumps_config(type(CONFIG)(params=params, scenarios=grid, output=CONFIG.output))
    )
    assert main(["scenario", str(no_fas), "-m", "6", "-r", "20", "--no-rv"]) == 1
    assert "--no-rv needs --td" in capsys.readouterr().err
    assert main(["scenario", str(no_fas), "-m", "6", "-r", "20", "--td", "--nsims", "2"]) == 0
    assert "Time domain, 2 simulations" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["scenario", str(path), "-m", "6"])  # a distance is required


def test_run_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    no_freq = tmp_path / "no_freq.toml"
    no_freq.write_text(EXAMPLE.read_text().replace("frequencies = {", "# frequencies = {"))
    assert main(["run", str(no_freq), "--no-rv"]) == 1
    assert "needs --td or [scenarios] frequencies" in capsys.readouterr().err
    bad = tmp_path / "bad.toml"
    bad.write_text("schema_version = 1\n")
    assert main(["run", str(bad)]) == 1
    assert "invalid" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        main(["run"])
    assert exit_info.value.code == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"pysmsim {__version__}"


def test_python_m(config_file: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "pysmsim", "check", "-q", str(config_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("count = 200 }", "count = 300000000 }", "periods"),
        ("count = 41 }", "count = 1001 }", "frequencies"),
        (
            "frequencies = { start = 0.01, stop = 100.0, count = 41 }",
            "frequencies = [" + ", ".join(str(0.01 * (i + 1)) for i in range(1001)) + "]",
            "frequencies",
        ),
    ],
)
def test_too_many_periods_or_frequencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], old: str, new: str, name: str
) -> None:
    # Refused while the file is read, before the values are expanded
    path = tmp_path / "big.toml"
    path.write_text(EXAMPLE.read_text().replace(old, new))
    assert main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "at most 1000 periods and 1000 frequencies" in err
    assert f"reduce the number of {name}" in err


def test_grid_limits() -> None:
    ScenarioGrid(magnitudes=(5.0,), distances=(10.0,), periods=(1.0,) * 1000)
    with pytest.raises(ValueError, match="reduce the number of periods"):
        ScenarioGrid(magnitudes=(5.0,), distances=(10.0,), periods=(1.0,) * 1001)
    with pytest.raises(ValueError, match="reduce the number of frequencies"):
        ScenarioGrid(magnitudes=(5.0,), distances=(10.0,), frequencies=(1.0,) * 1001)
