"""Every file is documented, in the form the reference site is generated from.

The site in ``docs/site`` is built from the source: Python by autodoc, C by
Hawkmoth, shell scripts and workflows by their header comments. What is not
written in the source is not on the site, so these tests hold the source to
it. A test function's name is its claim, written as a sentence, and needs no
docstring; closures inside tests are not documented either. Everything else
is.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PYTHON = sorted(
    [
        *ROOT.glob("src/surmentations/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("docs/*.py"),
        *ROOT.glob("docs/site/**/*.py"),
        ROOT / "hatch_build.py",
    ]
)
TESTS = sorted(ROOT.glob("tests/**/*.py"))
C_SOURCES = sorted([*ROOT.glob("native/*.c"), *ROOT.glob("native/*.h"), *ROOT.glob("native/*/*.c")])


def _undocumented(path: Path, test_file: bool) -> list[str]:
    """Every definition in ``path`` without a docstring, as ``line name``."""
    tree = ast.parse(path.read_text())
    missing = [] if ast.get_docstring(tree) else ["module"]

    def walk(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                is_function = not isinstance(child, ast.ClassDef)
                exempt = test_file and (
                    (is_function and child.name.startswith("test_") and not in_function)
                    or (is_function and in_function)
                )
                if not exempt and not ast.get_docstring(child):
                    missing.append(f"{child.lineno} {name}")
                walk(child, name + ".", in_function or is_function)
            else:
                walk(child, prefix, in_function)

    walk(tree, "", False)
    return missing


@pytest.mark.parametrize("path", PYTHON, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_python_definition_has_a_docstring(path: Path) -> None:
    assert _undocumented(path, test_file=False) == []


@pytest.mark.parametrize("path", TESTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_test_module_and_helper_has_a_docstring(path: Path) -> None:
    assert _undocumented(path, test_file=True) == []


# A top-level C declaration: a function definition, a type, a macro.
_C_DECLARATION = re.compile(
    r"^(static |SM_EXPORT |extern |typedef |#define |int |void |ptrdiff_t |const |float |double )"
)
# A member of the kernel table: a function pointer or the tier's name.
_C_MEMBER = re.compile(r"^    (int|void|ptrdiff_t|const char \*)\s*\(?\*?\w")


@pytest.mark.parametrize("path", C_SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_c_declaration_has_a_doc_comment(path: Path) -> None:
    """Hawkmoth documents a declaration from the ``/** */`` comment just before it.

    Declarations a macro's own definition spans (``SM_CAT_`` and the like) and
    include guards are exempt: they are spelling, not interface.
    """
    lines = path.read_text().split("\n")
    missing = []
    for index, line in enumerate(lines):
        is_member = path.suffix == ".h" and _C_MEMBER.match(line)
        if not (_C_DECLARATION.match(line) or is_member):
            continue
        is_extern_function = line.startswith("extern") and "(" in line
        if line.rstrip().endswith(";") and not is_member and not is_extern_function:
            continue  # a variable, or a declaration of one defined elsewhere
        if re.match(r"#define (SM_\w+_|SM_KERNELS_H)\b", line):
            continue
        above = index - 1
        while above >= 0 and lines[above].strip().startswith("#if"):
            above -= 1
        previous = lines[above].strip() if above >= 0 else ""
        if not (previous.endswith("*/") and previous != "/**/"):
            missing.append(f"{index + 1}: {line.strip()[:70]}")
            continue
        start = above
        while start > 0 and "/*" not in lines[start]:
            start -= 1
        if not lines[start].lstrip().startswith("/**"):
            missing.append(f"{index + 1}: {line.strip()[:70]} (a plain /* comment)")
    assert missing == []


def _modules(pattern: str, root: str) -> set[str]:
    """The dotted names of the modules ``pattern`` finds, counted from ``root``."""
    names = set()
    for path in ROOT.glob(pattern):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT / root).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        names.add(".".join(parts))
    return names


def test_every_module_has_a_page_on_the_site() -> None:
    pages = {path.stem for path in (ROOT / "docs/site/python").glob("*.rst")}
    assert _modules("src/surmentations/**/*.py", "src") == pages
    tests = {path.stem for path in (ROOT / "docs/site/tests").glob("*.rst")}
    assert _modules("tests/**/*.py", ".") == tests
