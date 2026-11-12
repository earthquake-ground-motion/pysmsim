"""Griffe extension: turn the reStructuredText of pysmsim's docstrings into Markdown.

The docstrings are NumPy style with Sphinx roles (``:func:`name```) and
literal blocks introduced by ``::``. mkdocstrings renders Markdown, so this
extension, loaded from ``mkdocs.yml``, rewrites each docstring before it is
rendered:

- a role naming a public pysmsim object becomes a cross-reference,
  ``[name][pysmsim.module.name]`` (a leading ``~`` shows only the last
  part, as in Sphinx); other roles become inline code;
- a line ending in ``::`` ends in ``:`` (the indented block after it is
  already a Markdown code block);
- asterisks outside code are escaped, since they are multiplication and
  powers in the formulas (``2*pi*f``, ``f**2``), not emphasis.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE = re.compile(r":(?:func|class|meth|attr|data|mod|obj):`(~?)([^`]+)`")
_LITERAL = re.compile(r"(\S)::$", re.MULTILINE)
_CODE_SPAN = re.compile(r"(``.*?``|`[^`]*`)", re.DOTALL)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _code_lines(lines: list[str]) -> list[bool]:
    """Mark the lines of indented code blocks.

    A block starts after a blank line, indented by 4 or more beyond the
    text before the blank line, and ends at the first line that is not
    indented beyond that text.
    """
    marks = []
    base = 0  # indent of the last text line outside a block
    block = False
    blank = False
    for line in lines:
        if not line.strip():
            marks.append(block)
            blank = True
            continue
        indent = _indent(line)
        if block and indent <= base:
            block = False
        if not block and blank and indent >= base + 4:
            block = True
        if not block:
            base = indent
        marks.append(block)
        blank = False
    return marks


def _escape_asterisks(text: str) -> str:
    """Escape ``*`` outside code spans and indented code blocks."""
    lines = text.split("\n")
    marks = _code_lines(lines)
    out: list[str] = []
    run: list[str] = []  # consecutive lines of text, which a code span may cross

    def flush() -> None:
        parts = _CODE_SPAN.split("\n".join(run))  # the odd items are code spans
        parts[::2] = [part.replace("*", r"\*") for part in parts[::2]]
        out.extend("".join(parts).split("\n"))
        run.clear()

    for line, code in zip(lines, marks, strict=True):
        if code:
            if run:
                flush()
            out.append(line)
        else:
            run.append(line)
    if run:
        flush()
    return "\n".join(out)


def _link(title: str, found: griffe.Object | griffe.Alias) -> str:
    """Link to an object the reference shows; private objects stay plain code.

    Attributes without a docstring of their own (dataclass fields, which
    the class docstring describes) link to their class.
    """
    if any(part.startswith("_") for part in found.path.split(".")):
        return f"`{title}`"
    target = found
    if target.is_attribute and target.docstring is None and target.parent is not None:
        target = target.parent
    return f"[`{title}`][{target.path}]"


class RstToMarkdown(griffe.Extension):
    """Rewrite Sphinx roles, literal-block markers and asterisks in every docstring."""

    def on_object(self, *, obj: griffe.Object, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        """Rewrite the docstring of one object, once the tree is complete."""
        if obj.docstring is None:
            return
        text = _escape_asterisks(obj.docstring.value)
        text = _ROLE.sub(lambda m: self._reference(m, obj, loader), text)
        obj.docstring.value = _LITERAL.sub(r"\1:", text)

    @staticmethod
    def _reference(match: re.Match[str], obj: griffe.Object, loader: griffe.GriffeLoader) -> str:
        short, target = match.group(1), match.group(2).lstrip(".")
        title = target.rsplit(".", 1)[-1] if short else target
        # Like Sphinx, look a relative name up in the current class, then the module
        scopes = [obj if obj.is_class or obj.is_module else obj.parent, obj.module, None]
        for scope in scopes:
            path = target if scope is None else f"{scope.path}.{target}"
            try:
                found = loader.modules_collection[path]
            except (KeyError, ValueError, griffe.AliasResolutionError):
                continue
            return _link(title, found)
        return f"`{title}`"
