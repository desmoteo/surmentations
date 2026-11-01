"""Document shell scripts and CI workflows from their own comments.

Sphinx has autodoc for Python and Hawkmoth for C; the repository's other
languages, POSIX shell and GitHub Actions YAML, are documented the way they
are written: a block of ``#`` comments at the top saying what the file is
for, and comments beside each step. The ``script`` directive renders the
first as text and the whole file, comments and all, as a listing::

    ```{script} native/check/run.sh
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


def header_comment(text: str) -> list[str]:
    """The paragraphs of a file's leading ``#`` comment block.

    A shebang is skipped; the block ends at the first line that is not a
    comment. Blank comment lines separate paragraphs.
    """
    paragraphs: list[list[str]] = [[]]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        if not stripped.startswith("#"):
            break
        body = stripped.lstrip("#").strip()
        if body:
            paragraphs[-1].append(body)
        elif paragraphs[-1]:
            paragraphs.append([])
    return [" ".join(lines) for lines in paragraphs if lines]


class ScriptDirective(SphinxDirective):
    """``script PATH``: a file's header comment, then the file as a listing."""

    required_arguments = 1
    option_spec = {"language": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        """Read the file at ``PATH``, relative to ``scriptdoc_root``, and render it."""
        relative = self.arguments[0]
        path = Path(self.config.scriptdoc_root) / relative
        self.env.note_dependency(str(path))
        text = path.read_text()
        language = self.options.get("language") or {".sh": "bash", ".yml": "yaml"}.get(
            path.suffix, "text"
        )
        section = nodes.section(ids=[nodes.make_id(relative)])
        section += nodes.title(text=relative)
        for paragraph in header_comment(text):
            section += nodes.paragraph(text=paragraph)
        listing = nodes.literal_block(text, text)
        listing["language"] = language
        section += listing
        return [section]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the directive and its one setting with Sphinx."""
    app.add_config_value("scriptdoc_root", ".", "env")
    app.add_directive("script", ScriptDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
