"""Sphinx configuration for the weathermentations code reference.

Build it from the repository root::

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs/site docs/site/_build/html

Every page is generated from the source: Python by autodoc, from the
docstrings; C by Hawkmoth, from the ``/** ... */`` comment before each
declaration, read through libclang; shell scripts and CI workflows by
``_ext/scriptdoc.py``, from their header comments. Nothing is written twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT, ROOT / "scripts", ROOT / "docs", Path(__file__).parent / "_ext"):
    sys.path.insert(0, str(path))

project = "weathermentations"
author = "the weathermentations authors"
copyright = "the weathermentations authors, MIT licence"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "hawkmoth",
    "scriptdoc",
]

# Python: every module's own members, private helpers included, in the order
# they are written. `__all__` lists what a module exports, not what it holds.
autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "ignore-module-all": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# C: the kernels are compiled once per instruction-set tier with WM_TIER set;
# the reference is of the portable spelling of each.
hawkmoth_root = str(ROOT / "native")


# The compiler whose header directories libclang is given, so that it finds
# <math.h> and <stddef.h>; the kernels are built with `cc`, not clang.
hawkmoth_compiler = "cc"

hawkmoth_clang = [
    "-std=c11",
    "-DWM_TIER=portable",
    "-DWM_NATIVE",
    f"-I{ROOT / 'native'}",
]

scriptdoc_root = str(ROOT)

myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "weathermentations reference"
