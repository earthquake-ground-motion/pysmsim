"""The combined FAS and RV run and its HDF5 files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pysmsim.drivers.loop_fas import fas_loop
from pysmsim.drivers.loop_rv import rv_loop
from pysmsim.drivers.run import RV_COLUMNS, SCENARIO_COLUMNS, RunResult, run
from pysmsim.io.config import load_config
from pysmsim.io.hdf5 import FORMAT, UNITS, read_hdf5, write_hdf5
from pysmsim.scenario import DistanceMetric, ScenarioGrid

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
PARAMS = load_config(EXAMPLE).params
GRID = ScenarioGrid(
    magnitudes=(5.0, 7.0),
    distances=(10.0, 50.0),
    periods=(0.1, 1.0, 3.0),
    kappa0=(0.02, 0.04),
    stress=(50.0, 100.0),
)
F = np.array([0.0, 0.1, 1.0, 10.0])


@pytest.fixture(scope="module")
def result() -> RunResult:
    return run(PARAMS, GRID, F)


def test_run_tables_share_rows(result: RunResult) -> None:
    n = 2 * 2 * 2 * 2
    assert tuple(result.scenarios.columns) == SCENARIO_COLUMNS
    assert result.fas is not None and result.rv is not None
    assert len(result.scenarios) == len(result.fas) == len(result.rv) == n
    assert list(result.fas.columns) == ["fas(0)", "fas(0.1)", "fas(1)", "fas(10)"]
    motions = ["pgv", "pga", "psa(0.1)", "psa(1)", "psa(3)"]
    assert list(result.rv.columns) == [*RV_COLUMNS, *motions]
    # Order: stress, kappa_0, magnitude, distance
    assert result.scenarios.stress.tolist() == [50.0] * 8 + [100.0] * 8
    assert result.scenarios.kappa0.tolist()[:8] == [0.02] * 4 + [0.04] * 4
    assert result.scenarios.magnitude.tolist()[:4] == [5.0, 5.0, 7.0, 7.0]
    assert result.scenarios.rjb.tolist()[:4] == [10.0, 50.0, 10.0, 50.0]


def test_run_agrees_with_the_loops(result: RunResult) -> None:
    assert result.fas is not None and result.rv is not None
    # The scenario and RV tables together are the RV loop's table
    joined = result.scenarios.join(result.rv)
    pd.testing.assert_frame_equal(joined, rv_loop(PARAMS, GRID))
    # The FAS loop orders kappa_0 before stress
    fas = fas_loop(PARAMS, GRID, F)
    order = fas.sort_values(["stress", "kappa0"], kind="stable").index
    expected = fas.loc[order].reset_index(drop=True)
    for name in ("magnitude", "rjb", "h", "rrup", "f_ff", "rps", "kappa0", "stress"):
        np.testing.assert_array_equal(result.scenarios[name], expected[name], err_msg=name)
    np.testing.assert_array_equal(result.fas, expected[list(result.fas.columns)])


def test_run_without_fas_or_rv() -> None:
    grid = replace(GRID, distance_metric=DistanceMetric.RRUP, kappa0=None, stress=None)
    only_rv = run(PARAMS, grid)
    assert only_rv.fas is None and only_rv.frequencies.size == 0
    assert only_rv.rv is not None and len(only_rv.rv) == 4
    assert only_rv.scenarios.rjb.isna().all()
    only_fas = run(PARAMS, grid, F, rv=False)
    assert only_fas.rv is None and only_fas.fas is not None


def test_run_rejects_bad_frequencies() -> None:
    for f in ([1.0, 1.0000001], np.ones((2, 2)), [-1.0]):
        with pytest.raises(ValueError, match="frequencies"):
            run(PARAMS, GRID, f)


def _assert_same(a: RunResult, b: RunResult) -> None:
    assert a.params == b.params and a.grid == b.grid
    assert a.fortran_compat == b.fortran_compat
    pd.testing.assert_frame_equal(a.scenarios, b.scenarios)
    np.testing.assert_array_equal(a.frequencies, b.frequencies)
    for x, y in ((a.fas, b.fas), (a.rv, b.rv)):
        if x is None or y is None:
            assert x is None and y is None
        else:
            pd.testing.assert_frame_equal(x, y)


@pytest.mark.parametrize("compression", ["gzip", "lzf", None])
def test_hdf5_round_trip(result: RunResult, tmp_path: Path, compression: str | None) -> None:
    path = tmp_path / "run.h5"
    write_hdf5(result, path, compression=compression)
    _assert_same(read_hdf5(path), result)


def test_hdf5_round_trip_without_fas_or_pgv(tmp_path: Path) -> None:
    grid = replace(GRID, pgv=False, periods=(), kappa0=None)
    result = run(PARAMS, grid, fortran_compat=True)
    path = tmp_path / "run.h5"
    write_hdf5(result, path)
    back = read_hdf5(path)
    _assert_same(back, result)
    assert back.rv is not None and "pgv" not in back.rv and "pga" in back.rv


def test_hdf5_layout(result: RunResult, tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    write_hdf5(result, path)
    assert result.fas is not None and result.rv is not None
    with h5py.File(path) as h5:
        assert h5.attrs["format"] == FORMAT
        assert set(h5) == {"scenarios", "fas", "rv"}
        assert set(h5["scenarios"]) == set(SCENARIO_COLUMNS)
        assert h5["fas/fas"].shape == (16, 4)
        assert h5["rv/psa"].shape == (16, 3)
        assert h5["rv/psa"].attrs["dimensions"] == "scenario, period"
        # Too small to compress
        assert h5["rv/psa"].compression is None
        assert h5["rv/period"].attrs["damping"] == 0.05
        for name, dataset in h5["rv"].items():
            assert dataset.attrs["units"] == UNITS[name]
        np.testing.assert_array_equal(h5["rv/psa"][:, 1], result.rv["psa(1)"])
        np.testing.assert_array_equal(h5["fas/fas"][:, 2], result.fas["fas(1)"])


def test_hdf5_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "something else"
    with pytest.raises(ValueError, match="not a pysmsim results file"):
        read_hdf5(path)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = FORMAT
        h5.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="format version 99"):
        read_hdf5(path)


def test_hdf5_compresses_large_datasets(tmp_path: Path) -> None:
    grid = replace(GRID, periods=tuple(np.geomspace(0.01, 10.0, 100)), kappa0=None, stress=None)
    result = run(PARAMS, grid, np.geomspace(0.01, 100.0, 300), rv=False)
    path = tmp_path / "run.h5"
    write_hdf5(result, path)
    with h5py.File(path) as h5:
        assert h5["fas/fas"].compression == "gzip"
        assert h5["scenarios/magnitude"].compression is None
    _assert_same(read_hdf5(path), result)
