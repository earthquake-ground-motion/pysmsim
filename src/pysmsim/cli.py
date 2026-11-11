"""The ``pysmsim`` command.

``pysmsim run`` computes the scenarios of a TOML run file and writes the
results; ``pysmsim check`` validates run files without computing anything;
``pysmsim convert`` turns SMSIM input files into a run file::

    pysmsim check run.toml
    pysmsim run run.toml
    pysmsim run run.toml -o results.h5
    pysmsim run run.toml -o results.col --fortran-compat
    pysmsim convert model.params tmrsk_loop_rv_drvr.ctl -o run.toml

``python -m pysmsim`` does the same. The output file and format come from
the ``[output]`` section of the run file unless ``-o`` or ``--format``
replaces them; a relative path in the run file is relative to the run
file's folder, and one given with ``-o`` to the current folder. Without
``--format``, the format follows the suffix of the ``-o`` file (``.csv``,
``.parquet``, ``.h5`` or ``.hdf5``, ``.col``) when it has one of these.

The exit status is 0 on success, 1 when a run file is invalid or the run
fails, and 2 for wrong command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pysmsim import __version__
from pysmsim.drivers.run import run
from pysmsim.io.config import ConfigError, RunConfig, load_config
from pysmsim.io.legacy import LegacyFormatError, convert
from pysmsim.io.results import format_for_path, write_results
from pysmsim.scenario import OutputFormat, ScenarioGrid

__all__ = ["main"]


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
            "file gives [scenarios] frequencies, the Fourier spectra of every scenario."
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
        help="compute only the Fourier spectra, not the peak motions",
    )
    run_cmd.add_argument(
        "--fortran-compat",
        action="store_true",
        help="reproduce SMSIM's behaviour where pysmsim differs (see the documentation)",
    )
    run_cmd.add_argument("-q", "--quiet", action="store_true", help="print nothing but errors")

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
            "of tmrsk_loop_rv_drvr into a TOML run file. Notes on where SMSIM would behave "
            "differently are printed and written at the top of the run file."
        ),
    )
    convert_cmd.add_argument("params", type=Path, help="SMSIM .params file")
    convert_cmd.add_argument(
        "ctl",
        type=Path,
        nargs="?",
        help="tmrsk_loop_rv_drvr control file for the scenarios (default: example scenarios)",
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


def _summary(config: RunConfig) -> str:
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
            print(f"{path}: OK\n{_summary(config)}")
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
    if fmt is OutputFormat.COL and args.no_rv:
        print(
            "pysmsim: the col format holds peak motions, so it cannot be used with --no-rv",
            file=sys.stderr,
        )
        return 1
    grid = config.scenarios
    if args.no_rv and not grid.frequencies:
        print("pysmsim: --no-rv needs [scenarios] frequencies in the run file", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"pysmsim {__version__}: {args.config}\n{_summary(config)}", file=sys.stderr)
    start = time.perf_counter()
    try:
        result = run(config.params, grid, rv=not args.no_rv, fortran_compat=args.fortran_compat)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_results(result, path, fmt)
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
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
