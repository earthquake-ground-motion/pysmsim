# Speed

## How long things take

On a 4-core machine, after the first run (which compiles the kernels):

| Work | Time |
| --- | --- |
| `fourier_spectrum` at 200 frequencies | about 50 µs |
| `rv_motion`, one motion | about 0.5 ms |
| The example run file by RV (6 scenarios, PGV, PGA and 200 periods) | about 0.7 s |
| `td_motion`, 800 simulations and a few periods | about 0.3 s per scenario |
| `td_motion`, 800 simulations and 200 periods | about 2.5 s per scenario |

The time of a TD scenario grows with `nsims`, with the number of points of
its series (longer for large magnitudes and distances, and for small `dt`)
and with the number of periods. `td_time_grids` gives the number of points
of every scenario of a grid without simulating (see
[Time domain](td.md#the-loop-over-a-grid-td_loop)).

## Numba

The inner loops are compiled by Numba: the spectrum factors, the
time-domain measures of each series and the oscillator response. The
compiled code is cached in the package's `__pycache__` folders, so only the
first run after installing or changing pysmsim compiles it (a few seconds).

The TD kernels run in parallel on every core. `NUMBA_NUM_THREADS` limits
the number of threads, for example to share a machine:

```bash
NUMBA_NUM_THREADS=2 pysmsim run run.toml --td
```

`NUMBA_DISABLE_JIT=1` runs the same code as plain Python. It gives the same
results, much more slowly, and is useful for debugging.

## Processes: `workers`

The RV method computes one scenario at a time on one core. `workers`
shares the scenarios of a grid among several processes:

```python
from pysmsim.drivers.loop_rv import rv_loop
from pysmsim.io.config import load_config

config = load_config("examples/wna_bt15_example.toml")
if __name__ == "__main__":
    table = rv_loop(config.params, config.scenarios, workers=2)
```

`workers=0` (or None) uses one process per CPU. `rv_loop`,
`rv_loop_details`, `td_loop`, `td_loop_details` and `run` take it, and so
does `pysmsim run --workers N`.

- The results are identical to one process, to the last digit.
- Starting the processes takes a second or two, so it pays off only for
  grids that take longer than that. On 4 cores, a grid of 340 scenarios
  with 200 periods took 71 s with one process and 21 s with four.
- The TD method already uses every core through Numba, so `workers` does
  not make it faster on one machine.
- The processes are started with `spawn`, so a script that uses `workers`
  must call the loop under `if __name__ == "__main__":`, as above.
  Notebooks need nothing special.

## Measuring

`examples/benchmark.py` times the FAS, RV and TD loops of a run file, and
checks that every number of workers gives the same tables:

```bash
python examples/benchmark.py
python examples/benchmark.py my_run.toml --workers 1 4 --nsims 100
```
