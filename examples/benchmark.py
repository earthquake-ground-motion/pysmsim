"""Time the FAS, RV and TD loops of a run file.

Run from the repository's root folder::

    python examples/benchmark.py
    python examples/benchmark.py examples/wna_bt15_example.toml --workers 1 4 --nsims 100

Each loop runs once to warm up (Numba loads or compiles its kernels then)
and then ``--repeat`` times (the TD loop once); the script prints the fastest
time of each. With several ``--workers`` values it also checks that every
value gives the same table. ``--nsims`` sets the number of TD simulations
(default: the run file's), and ``--no-td`` leaves the TD loop out.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from pysmsim.drivers.loop_fas import fas_loop
from pysmsim.drivers.loop_rv import rv_loop_details
from pysmsim.drivers.loop_td import td_loop_details
from pysmsim.io.config import load_config

EXAMPLE = Path(__file__).parent / "wna_bt15_example.toml"


def best_time(func: Callable[[], pd.DataFrame], repeat: int) -> tuple[float, pd.DataFrame]:
    """Return the fastest of ``repeat`` timed calls after one untimed call, and the table."""
    table = func()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        table = func()
        times.append(time.perf_counter() - start)
    return min(times), table


def main() -> None:
    """Parse the command line and time the loops."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, nargs="?", default=EXAMPLE, help="TOML run file")
    parser.add_argument("--workers", type=int, nargs="+", default=[1], help="process counts")
    parser.add_argument("--nsims", type=int, help="TD simulations of each scenario")
    parser.add_argument("--repeat", type=int, default=3, help="timed calls of each loop")
    parser.add_argument("--no-td", action="store_true", help="leave out the TD loop")
    args = parser.parse_args()

    config = load_config(args.config)
    params, grid = config.params, config.scenarios
    if args.nsims is not None:
        params = dataclasses.replace(params, td=dataclasses.replace(params.td, nsims=args.nsims))
    nscen = (
        len(grid.stress or (None,))
        * len(grid.kappa0 or (None,))
        * len(grid.magnitudes)
        * len(grid.distances)
    )
    print(
        f"{args.config}: {nscen} scenarios, {len(grid.periods)} periods, "
        f"{params.td.nsims} TD simulations; {numba.get_num_threads()} Numba threads"
    )

    frequencies = np.asarray(grid.frequencies or np.geomspace(0.01, 100.0, 200))
    seconds, _ = best_time(lambda: fas_loop(params, grid, frequencies), args.repeat)
    print(f"FAS ({frequencies.size} frequencies): {seconds:8.3f} s")

    loops: list[tuple[str, Callable[[int], pd.DataFrame]]] = [
        ("RV", lambda w: rv_loop_details(params, grid, workers=w))
    ]
    if not args.no_td:
        loops.append(("TD", lambda w: td_loop_details(params, grid, workers=w)))
    for name, loop in loops:
        first: pd.DataFrame | None = None
        for workers in args.workers:
            repeat = args.repeat if name == "RV" else 1
            seconds, table = best_time(lambda f=loop, w=workers: f(w), repeat)
            if first is None:
                first = table
            else:
                pd.testing.assert_frame_equal(table, first, check_exact=True)
            print(f"{name} (workers {workers}): {seconds:8.3f} s")


if __name__ == "__main__":
    main()
