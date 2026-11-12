# Development

## Set-up

```bash
uv sync --extra dev --extra docs
```

## Tests and checks

```bash
uv run pytest                          # all tests
uv run pytest -m "not slow"            # without the long time-domain statistics
NUMBA_DISABLE_JIT=1 uv run pytest      # the same code as plain Python
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
```

- `tests/unit/` test each module on its own.
- `tests/golden/` compare pysmsim with SMSIM's own programs, through the
  golden files below. They never need a Fortran compiler.
- `tests/docs/` run the Python snippets of this documentation's guide.
- The `slow` marker labels the tests that run hundreds of time-domain
  simulations; the `golden` marker labels the comparisons with SMSIM.
- With `NUMBA_DISABLE_JIT=1`, the few tests that need the compiled kernels
  to finish in reasonable time are skipped.

Every compiled kernel has a plain NumPy twin, and the unit tests check that
the two agree.

## The Fortran oracle and the golden files

`oracle/` holds scripts that build SMSIM's programs from the SMSIM
distribution (which is never modified: it is copied and patched in
`oracle/build/`) and run them on the inputs in `oracle/inputs/`. Their
outputs are committed in `tests/golden/double/`, with the compiler, flags
and patches in `BUILD_INFO.txt`. Regenerating them needs gfortran:

```bash
oracle/build.sh /path/to/smsim.2023-03-03_version
oracle/make_golden.sh /path/to/smsim.2023-03-03_version double
```

`oracle/README.md` describes the programs and patches, and
`tests/golden/README.md` every golden file and the cases they cover.

## Documentation

```bash
uv run mkdocs serve        # preview at http://127.0.0.1:8000, rebuilt on every change
uv run mkdocs build        # write the site to site/
```

The pages are Markdown files in `docs/`, and `mkdocs.yml` sets their order.
The API reference is generated from the docstrings by mkdocstrings. The
docstrings are NumPy style and use Sphinx roles such as
``:func:`rv_motion` ``; `docs/_scripts/griffe_rst.py` turns those into
links when the site is built.

## Conventions

- Every physics step is a public function with plain arguments that
  returns NumPy arrays or a small frozen dataclass; the drivers only
  compose them.
- Each model with its own parameters is its own class, chosen by its
  `model` name in a run file.
- SMSIM quirks are fixed by default, with the original behaviour behind a
  `fortran_compat` flag where it is a convention rather than an error (see
  [Comparison with SMSIM](smsim.md)).
- Units are cgs, with distances in km and g = 981 cm/s².

`CLAUDE.md` in the repository records the project's decisions in full.
