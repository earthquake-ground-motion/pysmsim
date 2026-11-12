"""Run the scenarios of a loop in several processes.

The loop drivers split their work into tasks (blocks of scenarios) and call
:func:`map_tasks`, which runs them in this process when ``workers`` is 1 and
otherwise in a pool of worker processes. Each task is computed exactly as it
would be in one process, so the results do not depend on ``workers``.

The workers are started with ``spawn``, which is safe with Numba's threads
and the same on every platform. Each receives the shared state (the model
parameters and the grid) once, when it starts, and the tasks are small
tuples that name the scenarios. Each worker uses its share of this process's Numba threads.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numba

__all__ = ["map_tasks", "resolve_workers"]

_STATE: Any = None


def resolve_workers(workers: int | None) -> int:
    """Return the number of processes for ``workers``: None or 0 means one per CPU.

    Raises
    ------
    ValueError
        If ``workers`` is negative.
    """
    if workers is None or workers == 0:
        return os.cpu_count() or 1
    if workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    return workers


def _start(state: Any, threads: int) -> None:
    global _STATE
    _STATE = state
    numba.set_num_threads(threads)


def _call(job: tuple[Callable[[Any, Any], Any], Any]) -> Any:
    func, task = job
    return func(_STATE, task)


def map_tasks[S, T, R](
    func: Callable[[S, T], R], state: S, tasks: Sequence[T], workers: int | None = 1
) -> list[R]:
    """Return ``[func(state, task) for task in tasks]``, possibly in parallel.

    Parameters
    ----------
    func : callable
        A module-level function (it is sent to the workers by name).
    state : object
        Shared data, sent to each worker once; it must be picklable.
    tasks : sequence
        The tasks, small picklable objects.
    workers : int or None
        Number of processes; 1 (the default) computes in this process,
        None or 0 uses one per CPU.

    Returns
    -------
    list
        The results, in the order of the tasks.
    """
    n = min(resolve_workers(workers), len(tasks))
    if n <= 1:
        return [func(state, task) for task in tasks]
    threads = max(1, numba.get_num_threads() // n)
    chunk = max(1, len(tasks) // (4 * n))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n, mp_context=context, initializer=_start, initargs=(state, threads)
    ) as pool:
        return list(pool.map(_call, [(func, task) for task in tasks], chunksize=chunk))
