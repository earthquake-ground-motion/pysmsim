"""Running the loops in several processes (pysmsim.drivers._parallel)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from pysmsim.drivers._parallel import map_tasks, resolve_workers
from pysmsim.drivers.loop_rv import rv_loop_details
from pysmsim.drivers.loop_td import td_loop_details
from pysmsim.drivers.run import run
from pysmsim.io.config import load_config
from pysmsim.scenario import ScenarioGrid

EXAMPLE = Path(__file__).parents[2] / "examples" / "wna_bt15_example.toml"
CONFIG = load_config(EXAMPLE)
# Few, short simulations so that the tests stay quick without the compiled kernels
PARAMS = dataclasses.replace(
    CONFIG.params, td=dataclasses.replace(CONFIG.params.td, nsims=3, dt=0.01, tshift=5.0)
)
GRID = ScenarioGrid(
    magnitudes=(4.0, 6.0),
    distances=(10.0, 40.0),
    periods=(0.1, 1.0),
    stress=(50.0, 100.0),
    kappa0=(0.02, 0.04),
)


def test_resolve_workers() -> None:
    assert resolve_workers(3) == 3
    assert resolve_workers(None) == resolve_workers(0) == (os.cpu_count() or 1)
    with pytest.raises(ValueError, match="workers must be >= 0"):
        resolve_workers(-1)


def test_map_tasks_in_process() -> None:
    # One task, or one worker, needs no other process (a local function would not pickle)
    def add(state: int, task: int) -> int:
        return state + task

    assert map_tasks(add, 10, [1, 2, 3]) == [11, 12, 13]
    assert map_tasks(add, 10, [5], workers=4) == [15]
    assert map_tasks(add, 10, [], workers=4) == []


def test_rv_loop_workers() -> None:
    serial = rv_loop_details(PARAMS, GRID)
    pd.testing.assert_frame_equal(
        rv_loop_details(PARAMS, GRID, workers=2), serial, check_exact=True
    )


def test_td_loop_workers() -> None:
    grid = dataclasses.replace(GRID, stress=None, kappa0=None)
    serial = td_loop_details(PARAMS, grid)
    pd.testing.assert_frame_equal(
        td_loop_details(PARAMS, grid, workers=2), serial, check_exact=True
    )


def test_run_workers() -> None:
    grid = dataclasses.replace(GRID, stress=None, kappa0=None, magnitudes=(5.0,))
    serial = run(PARAMS, grid, td=True)
    parallel = run(PARAMS, grid, td=True, workers=2)
    for name in ("rv", "td", "td_details"):
        pd.testing.assert_frame_equal(
            getattr(parallel, name), getattr(serial, name), check_exact=True
        )
