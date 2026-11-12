"""The ``pysmsim`` command.

``pysmsim run`` computes the scenarios of a TOML run file and writes the
results; ``pysmsim scenario`` gives every detail of one magnitude and
distance with the model of a run file; ``pysmsim check`` validates run
files without computing anything; ``pysmsim convert`` turns SMSIM input
files into a run file::

    pysmsim check run.toml
    pysmsim run run.toml
    pysmsim run run.toml -o results.h5
    pysmsim run run.toml -o results.col --fortran-compat
    pysmsim run run.toml --td -o results.h5
    pysmsim run run.toml --td --workers 0 -o results.h5
    pysmsim run run.toml --td --td-series 10 -o results.h5
    pysmsim scenario run.toml -m 6 -r 20
    pysmsim scenario run.toml -m 6 --rjb 15 --td --series 1 -o m6_rjb15
    pysmsim convert model.params tmrsk_loop_rv_drvr.ctl -o run.toml

``python -m pysmsim`` does the same. The output file and format come from
the ``[output]`` section of the run file unless ``-o`` or ``--format``
replaces them; a relative path in the run file is relative to the run
file's folder, and one given with ``-o`` to the current folder. Without
``--format``, the format follows the suffix of the ``-o`` file (``.csv``,
``.parquet``, ``.h5`` or ``.hdf5``, ``.col``) when it has one of these.

``pysmsim scenario`` prints its results; with ``-o DIR`` it also writes
them as CSV files in DIR (``rv.csv``, ``td.csv``, ``fas.csv`` and
``td_series.csv``; see :mod:`pysmsim.drivers.single` for the columns).

The exit status is 0 on success, 1 when a run file is invalid or the run
fails, and 2 for wrong command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pysmsim import __version__
from pysmsim.drivers.run import run
from pysmsim.drivers.single import fas_scenario, rv_scenario, td_scenario
from pysmsim.io.config import ConfigError, RunConfig, load_config
from pysmsim.io.hdf5 import SERIES_TOLERANCE, write_td_series
from pysmsim.io.legacy import LegacyFormatError, convert
from pysmsim.io.results import format_for_path, write_results
from pysmsim.scenario import OutputFormat, ScenarioGrid
from pysmsim.spectrum.distance import rjb_to_rrup_depth

__all__ = ["main"]


def _workers(text: str) -> int:
    """Parse the number of processes of ``--workers``, an integer >= 0."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysmsim",
        description="Stochastic ground-motion simulation (SMSIM) in Python.",
    )
    parser.add_argument("--version", action="version", version=f"pysmsim {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    run_cmd = commands.add_parser(
        "run",
        help="compute the scenarios of a run file and write the results",
        description=(
            "Compute the random-vibration peak motions (PGV, PGA, PSA) and, when the run "
            "file gives [scenarios] frequencies, the Fourier spectra of every scenario. "
            "With --td, also simulate [td] nsims time series of every scenario."
        ),
    )
    run_cmd.add_argument("config", type=Path, help="TOML run file")
    run_cmd.add_argument(
        "-o", "--output", type=Path, help="output file (default: [output] path of the run file)"
    )
    run_cmd.add_argument(
        "-f",
        "--format",
        choices=[f.value for f in OutputFormat],
        help="output format (default: from the -o suffix, else [output] format)",
    )
    run_cmd.add_argument(
        "--no-rv",
        action="store_true",
        help="do not compute the random-vibration peak motions",
    )
    run_cmd.add_argument(
        "--td",
        action="store_true",
        help="also compute the peak motions from time-domain simulations (slower)",
    )
    run_cmd.add_argument(
        "--td-series",
        type=int,
        nargs="?",
        const=0,
        metavar="NSIMS",
        help=(
            "also write the time-domain acceleration series to the HDF5 output (/td/series): "
            "the first NSIMS simulations of each scenario, or all of them"
        ),
    )
    run_cmd.add_argument(
        "--series-tolerance",
        type=float,
        default=SERIES_TOLERANCE,
        metavar="FRACTION",
        help=(
            "with --td-series, leave out the start and end of each series below this fraction "
            f"of its peak (default {SERIES_TOLERANCE:g}: the whole series)"
        ),
    )
    run_cmd.add_argument(
        "--series-precision",
        type=int,
        choices=(32, 64),
        default=32,
        metavar="BITS",
        help="with --td-series, store 32-bit (default) or 64-bit floats",
    )
    run_cmd.add_argument(
        "--fortran-compat",
        action="store_true",
        help="reproduce SMSIM's behaviour where pysmsim differs (see the documentation)",
    )
    run_cmd.add_argument(
        "-w",
        "--workers",
        type=_workers,
        default=1,
        metavar="N",
        help=(
            "share the RV and TD scenarios among N processes (default 1; 0: one per CPU); "
            "the results are the same, and it pays off for runs longer than a few seconds"
        ),
    )
    run_cmd.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")

    scenario_cmd = commands.add_parser(
        "scenario",
        help="compute every detail of one scenario",
        description=(
            "Compute PGD, PGV, PGA, the Arias intensity and the response spectrum of one "
            "magnitude and distance by random vibration, with their moments, durations and "
            "peak factors (SMSIM's gm_rv_drvr), using the model, periods and damping of a "
            "run file. With --td, also their averages over [td] nsims time-domain "
            "simulations (gm_td_drvr). When the run file gives [scenarios] frequencies, "
            "also the Fourier spectra and their factors (FAS_DRVR). The magnitudes, "
            "distances and kappa_0 and stress lists of [scenarios] are not used."
        ),
    )
    scenario_cmd.add_argument("config", type=Path, help="TOML run file")
    scenario_cmd.add_argument(
        "-m", "--magnitude", type=float, required=True, help="moment magnitude"
    )
    distance = scenario_cmd.add_mutually_exclusive_group(required=True)
    distance.add_argument("-r", "--rrup", type=float, help="rupture distance Rrup (km)")
    distance.add_argument(
        "--rjb",
        type=float,
        help="Joyner-Boore distance (km), converted to Rrup as the run file's [scenarios] says",
    )
    scenario_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="DIR",
        help="also write the results as CSV files in DIR",
    )
    scenario_cmd.add_argument(
        "--no-rv", action="store_true", help="do not compute the random-vibration results"
    )
    scenario_cmd.add_argument(
        "--td", action="store_true", help="also simulate time series (slower)"
    )
    scenario_cmd.add_argument(
        "--nsims", type=int, metavar="N", help="with --td, the number of simulations ([td] nsims)"
    )
    scenario_cmd.add_argument(
        "--series",
        type=int,
        metavar="K",
        help=(
            "with --td and -o, write the acceleration, velocity and displacement of "
            "simulation K (counted from 1) to td_series.csv"
        ),
    )
    scenario_cmd.add_argument(
        "--fortran-compat",
        action="store_true",
        help="reproduce SMSIM's behaviour where pysmsim differs (see the documentation)",
    )
    scenario_cmd.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")

    check_cmd = commands.add_parser(
        "check",
        help="validate run files and summarize them",
        description="Validate run files without computing anything.",
    )
    check_cmd.add_argument("configs", type=Path, nargs="+", metavar="config", help="TOML run file")
    check_cmd.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")

    convert_cmd = commands.add_parser(
        "convert",
        help="convert SMSIM input files into a run file",
        description=(
            "Convert an SMSIM parameter file (dated 02/25/21) and, optionally, a control file "
            "of tmrsk_loop_rv_drvr or tmrsk_loop_td_drvr into a TOML run file. Notes on where "
            "SMSIM would behave differently are printed and written at the top of the run file."
        ),
    )
    convert_cmd.add_argument("params", type=Path, help="SMSIM .params file")
    convert_cmd.add_argument(
        "ctl",
        type=Path,
        nargs="?",
        help=(
            "tmrsk_loop_rv_drvr or tmrsk_loop_td_drvr control file for the scenarios "
            "(default: example scenarios)"
        ),
    )
    convert_cmd.add_argument(
        "-o", "--output", type=Path, help="run file to write (default: print it)"
    )
    convert_cmd.add_argument(
        "--force", action="store_true", help="replace the -o file if it exists"
    )
    convert_cmd.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")
    return parser


def _count(values: tuple[float, ...] | None) -> int:
    return 1 if values is None else len(values)


def _n_scenarios(grid: ScenarioGrid) -> int:
    return len(grid.magnitudes) * len(grid.distances) * _count(grid.kappa0) * _count(grid.stress)


def _summary(config: RunConfig, td: bool = False) -> str:
    grid = config.scenarios
    params = config.params
    loops = [f"{len(grid.magnitudes)} magnitudes", f"{len(grid.distances)} distances"]
    if grid.kappa0 is not None:
        loops.append(f"{len(grid.kappa0)} kappa_0")
    if grid.stress is not None:
        loops.append(f"{len(grid.stress)} stress values")
    measures = [name for name, wanted in (("PGV", grid.pgv), ("PGA", grid.pga)) if wanted]
    if grid.periods:
        measures.append(f"PSA at {len(grid.periods)} periods (damping {grid.damping:g})")
    lines = [
        f"  title: {params.title or '(none)'}",
        f"  model: source {params.source.name}, spreading {params.spreading.name}, "
        f"Q {params.q.name}, oscillator correction {params.rv.osc_correction.value}",
        f"  scenarios: {_n_scenarios(grid)} ({' x '.join(loops)}), "
        f"distances as {grid.distance_metric.value}",
        f"  RV: {', '.join(measures)}",
        f"  FAS: {len(grid.frequencies)} frequencies" if grid.frequencies else "  FAS: none",
    ]
    if td:
        lines.append(
            f"  TD (--td): {params.td.nsims} simulations of each scenario, "
            f"{params.td.window.value} window, dt {params.td.dt:g} s"
        )
    lines += [
        f"  output: {config.output.path} ({config.output.format.value})",
    ]
    return "\n".join(lines)


def _load(path: Path) -> RunConfig | None:
    """Read a run file, or print why it cannot be read and return None."""
    try:
        return load_config(path)
    except ConfigError as err:  # its message starts with the path
        print(err, file=sys.stderr)
    except OSError as err:
        print(f"{path}: cannot read the file: {err.strerror or err}", file=sys.stderr)
    return None


def _check(args: argparse.Namespace) -> int:
    status = 0
    for path in args.configs:
        config = _load(path)
        if config is None:
            status = 1
            continue
        if not args.quiet:
            print(f"{path}: OK\n{_summary(config, td=True)}")
    return status


def _output(args: argparse.Namespace, config: RunConfig) -> tuple[Path, OutputFormat]:
    if args.output is not None:
        path = Path(args.output)
    else:
        path = Path(config.output.path)
        if not path.is_absolute():
            path = Path(args.config).parent / path
    if args.format is not None:
        fmt = OutputFormat(args.format)
    elif args.output is not None:
        fmt = format_for_path(path) or config.output.format
    else:
        fmt = config.output.format
    return path, fmt


def _run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if config is None:
        return 1
    path, fmt = _output(args, config)
    if fmt is OutputFormat.COL and (args.no_rv or args.td):
        print(
            "pysmsim: the col format holds the random-vibration peak motions only, so it "
            "cannot be used with --no-rv or --td",
            file=sys.stderr,
        )
        return 1
    if args.td_series is not None and (fmt is not OutputFormat.HDF5 or args.td_series < 0):
        print(
            "pysmsim: --td-series needs the hdf5 format and a number of simulations >= 0",
            file=sys.stderr,
        )
        return 1
    grid = config.scenarios
    if args.no_rv and not args.td and not grid.frequencies:
        print(
            "pysmsim: --no-rv needs --td or [scenarios] frequencies in the run file",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        summary = _summary(config, td=args.td)
        print(f"pysmsim {__version__}: {args.config}\n{summary}", file=sys.stderr)
    start = time.perf_counter()
    try:
        result = run(
            config.params,
            grid,
            rv=not args.no_rv,
            td=args.td,
            fortran_compat=args.fortran_compat,
            workers=args.workers,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_results(result, path, fmt)
        if args.td_series is not None:
            write_td_series(
                path,
                config.params,
                grid,
                nsims=args.td_series or None,
                tolerance=args.series_tolerance,
                dtype=np.float32 if args.series_precision == 32 else np.float64,
                fortran_compat=args.fortran_compat,
            )
    except (ValueError, RuntimeError, OSError) as err:
        print(f"pysmsim: {err}", file=sys.stderr)
        return 1
    if not args.quiet:
        missing = 0 if result.rv is None else int(result.rv.isna().to_numpy().sum())
        note = f"; {missing} values NaN outside the BT15 tables" if missing else ""
        elapsed = time.perf_counter() - start
        print(
            f"Wrote {path} ({fmt.value}), {len(result.scenarios)} scenarios in "
            f"{elapsed:.1f} s{note}",
            file=sys.stderr,
        )
    return 0


_RV_PRINT = {
    "measure": "measure",
    "period": "T (s)",
    "value": "value",
    "psv": "PSV (cm/s)",
    "sd": "SD (cm)",
    "psa_pga": "PSA/PGA",
    "dominant_frequency": "f_dom (Hz)",
    "dur_rms": "Drms (s)",
    "n_zero_crossings": "n_zero",
    "rms": "rms",
    "peak_factor": "peak/rms",
}
_TD_PRINT = {
    "measure": "measure",
    "period": "T (s)",
    "mean": "mean",
    "sem": "sem",
    "gmean": "gmean",
    "gfactor": "gfactor",
}
_FAS_PRINT = {"f": "f (Hz)", "fds": "FDS (cm-s)", "fvs": "FVS (cm)", "fas": "FAS (cm/s)"}
_UNITS = {
    "pgd": "cm",
    "pgv": "cm/s",
    "pga": "cm/s^2",
    "arias": "cm/s",
    "psa": "cm/s^2",
    "dur_75_05": "s",
    "dur_95_05": "s",
}


def _show(table: pd.DataFrame, columns: dict[str, str]) -> str:
    shown = table[list(columns)].rename(columns=columns)
    if "measure" in shown:
        shown["measure"] = [f"{m} ({_UNITS[m]})" for m in table.measure]
    return shown.to_string(index=False, float_format=lambda x: f"{x:.4g}", na_rep="")


def _scenario_heading(terms: dict[str, float], rjb: float | None) -> str:
    distance = (
        f"Rrup {terms['rrup']:g} km"
        if rjb is None
        else (f"Rjb {rjb:g} km, Rrup {terms['rrup']:.4g} km")
    )
    stress = "" if np.isnan(terms["stress"]) else f", stress {terms['stress']:.4g} bars"
    return (
        f"M {terms['magnitude']:g}, {distance}: f_ff {terms['f_ff']:.4g} km, "
        f"Rps {terms['rps']:.4g} km{stress}, fa {terms['fa']:.4g} Hz, "
        f"fb {terms['fb']:.4g} Hz, kappa {terms['kappa']:.4g} s"
    )


def _scenario(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if config is None:
        return 1
    grid = config.scenarios
    params = config.params
    problem = None
    if args.series is not None and (not args.td or args.output is None or args.series < 1):
        problem = "--series needs --td, -o and a simulation number >= 1"
    elif args.nsims is not None and (not args.td or args.nsims < 1):
        problem = "--nsims needs --td and a number >= 1"
    elif args.series is not None and args.series > (args.nsims or params.td.nsims):
        problem = f"--series {args.series} is more than the number of simulations"
    elif args.no_rv and not args.td and not grid.frequencies:
        problem = "--no-rv needs --td or [scenarios] frequencies in the run file"
    if problem is not None:
        print(f"pysmsim: {problem}", file=sys.stderr)
        return 1
    magnitude = args.magnitude
    rjb = args.rjb
    if rjb is not None:
        h = rjb_to_rrup_depth(
            magnitude,
            conversion=grid.rjb_to_rrup,
            mechanism=grid.mechanism,
            depth_function=grid.depth_function,
        )
        rrup = float(np.hypot(rjb, h))
    else:
        rrup = args.rrup
    if not rrup > 0 or (rjb is not None and rjb < 0):
        print("pysmsim: the distance must be positive", file=sys.stderr)
        return 1

    tables: dict[str, pd.DataFrame] = {}
    report: list[str] = []
    try:
        if not args.no_rv:
            rv = rv_scenario(
                params,
                magnitude,
                rrup,
                grid.periods,
                grid.damping,
                fortran_compat=args.fortran_compat,
            )
            tables["rv"] = rv.table()
            report += ["", "Random vibration:", _show(tables["rv"], _RV_PRINT)]
        if args.td:
            nsims = args.nsims or params.td.nsims
            td = td_scenario(
                params,
                magnitude,
                rrup,
                grid.periods,
                grid.damping,
                nsims=nsims,
                keep_series=args.series is not None,
            )
            tables["td"] = td.table()
            report += [
                "",
                f"Time domain, {nsims} simulations (sem: standard error of the mean; "
                "gfactor: geometric standard deviation):",
                _show(tables["td"], _TD_PRINT),
            ]
            if args.series is not None:
                tables["td_series"] = td.series(args.series - 1)
        if grid.frequencies:
            tables["fas"] = fas_scenario(
                params, magnitude, rrup, grid.frequencies, grid.periods, grid.damping
            )
            report += ["", "Fourier spectra:", _show(tables["fas"], _FAS_PRINT)]
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            for name, table in tables.items():
                table.to_csv(args.output / f"{name}.csv", index=False)
    except (ValueError, RuntimeError, OSError) as err:
        print(f"pysmsim: {err}", file=sys.stderr)
        return 1
    if not args.quiet:
        first = next(iter(tables.values()))
        terms = {
            k: float(first[k].iloc[0])
            for k in ("magnitude", "rrup", "f_ff", "rps", "stress", "fa", "fb", "kappa")
        }
        print(f"pysmsim {__version__}: {args.config}")
        print(_scenario_heading(terms, rjb))
        print("\n".join(report))
        if args.output is not None:
            files = ", ".join(f"{name}.csv" for name in tables)
            print(f"\nWrote {files} in {args.output}", file=sys.stderr)
    return 0


def _convert(args: argparse.Namespace) -> int:
    output: Path | None = args.output
    if output is not None and output.exists() and not args.force:
        print(f"pysmsim: {output} exists; use --force to replace it", file=sys.stderr)
        return 1
    try:
        text, notes = convert(args.params, args.ctl)
    except LegacyFormatError as err:
        print(f"pysmsim: {err}", file=sys.stderr)
        return 1
    if output is None:
        sys.stdout.write(text)
    else:
        try:
            output.write_text(text)
        except OSError as err:
            print(f"pysmsim: cannot write {output}: {err.strerror or err}", file=sys.stderr)
            return 1
    if not args.quiet:
        for note in notes:
            print(f"note: {note}", file=sys.stderr)
        if output is not None:
            print(f"Wrote {output}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``pysmsim`` command; return its exit status."""
    args = _parser().parse_args(argv)
    if args.command == "check":
        return _check(args)
    if args.command == "convert":
        return _convert(args)
    if args.command == "scenario":
        return _scenario(args)
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
