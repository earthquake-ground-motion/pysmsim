"""The Python snippets of the documentation run as they are.

Each page's ```python blocks run in order in one namespace, as a reader
would run them in a notebook, in a scratch folder that holds a copy of
``examples/`` (the snippets read the example run file and write result
files). Blocks marked ```text are not run.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numba
import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
PAGES = sorted(p for p in DOCS.rglob("*.md") if "```python" in p.read_text())
# Pages that simulate time series: slow without the compiled kernels
TIME_DOMAIN = {"guide/td.md", "guide/runs.md"}
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _page_id(page: Path) -> str:
    return page.relative_to(DOCS).as_posix()


def _params() -> list[object]:
    params = []
    for page in PAGES:
        name = _page_id(page)
        marks = []
        if name in TIME_DOMAIN:
            marks = [
                pytest.mark.slow,
                pytest.mark.skipif(numba.config.DISABLE_JIT, reason="needs the compiled kernels"),
            ]
        params.append(pytest.param(page, id=name, marks=marks))
    return params


def test_pages_found() -> None:
    names = {_page_id(p) for p in PAGES}
    assert {"guide/fourier.md", "guide/rv.md", "guide/td.md"} <= names


@pytest.mark.parametrize("page", _params())
def test_snippets(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__main__"}
    for number, block in enumerate(_BLOCK.findall(page.read_text())):
        code = compile(block, f"{_page_id(page)}, block {number + 1}", "exec")
        exec(code, namespace)
