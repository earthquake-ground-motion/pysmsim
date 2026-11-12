# Installation

pysmsim needs Python 3.12 or later. It is installed from a clone of the
repository:

```bash
git clone https://github.com/earthquake-ground-motion/pysmsim.git
cd pysmsim
uv sync                      # or: pip install -e .
```

The extras add optional tools:

| Extra | Installs | For |
| --- | --- | --- |
| `notebook` | JupyterLab, matplotlib | exploring the functions in notebooks |
| `dev` | pytest, hypothesis, ruff, mypy | running the tests and the checks |
| `docs` | MkDocs, mkdocstrings | building this site |

```bash
uv sync --extra notebook --extra dev     # or: pip install -e ".[notebook,dev]"
```

## Dependencies

NumPy, SciPy, pandas, pydantic, h5py, pyarrow and tomli-w, and Numba,
which compiles the inner loops (the source and spectrum kernels, the
time-domain measures and the oscillator response) to machine code.

The compiled kernels are cached next to the source files, so only the
first run after installing or changing pysmsim pays for compiling them.
Setting the environment variable `NUMBA_DISABLE_JIT=1` runs the same code
as plain Python, which gives the same results more slowly; the tests run
both ways.

## Building this documentation

```bash
uv sync --extra docs
uv run mkdocs serve          # preview at http://127.0.0.1:8000
uv run mkdocs build          # write the site to site/
```

The API reference is generated from the docstrings, so it follows the code.
