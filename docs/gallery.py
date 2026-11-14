"""Where the documentation's pictures are written, and where they are read from.

The pictures are not in the repository. Every one of them is rendered from
the code and the photographs that are, so CI renders them on each push to
``main`` and publishes them to GitHub Pages (``.github/workflows/gallery.yml``);
the pages link to them there. A picture kept in git is kept once per
regeneration, and the gallery was regenerated often enough to make the
history several hundred megabytes of superseded images.

Rendered locally, they land in the same places under ``docs/`` as the site
serves them from, and are ignored by git. :func:`url` maps one to the other.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DOCS = Path(__file__).resolve().parent

# The site's root serves what is under docs/: docs/examples/x.webp is
# <GALLERY_URL>examples/x.webp. Overridden for a fork's own Pages site.
GALLERY_URL = os.environ.get(
    "SURMENTATIONS_GALLERY_URL", "https://desmoteo.github.io/surmentations/"
).rstrip("/") + "/"

# WebP at this quality is indistinguishable from the PNGs it replaced at a
# full-size look, at a tenth of their size or less.
WEBP_QUALITY = 90


def url(path: Path) -> str:
    """The published address of a picture written under ``docs/``."""
    return GALLERY_URL + path.resolve().relative_to(DOCS).as_posix()


def local(address: str) -> Path | None:
    """The file under ``docs/`` a published address is rendered to; ``None`` for any other."""
    if not address.startswith(GALLERY_URL):
        return None
    return DOCS / address[len(GALLERY_URL) :]


def save(path: Path, rgb: np.ndarray, *, lossless: bool = False) -> None:
    """Write an RGB image as WebP, making its directory.

    ``lossless`` for a picture that is read back as an input, which has to
    come back with the pixels it was written with.
    """
    if path.suffix != ".webp":
        raise ValueError(f"the documentation's pictures are WebP, not {path.suffix}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV writes lossless WebP for a quality above 100.
    quality = 101 if lossless else WEBP_QUALITY
    if not cv2.imwrite(
        str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, quality]
    ):
        raise OSError(f"could not write {path}")


@functools.cache
def networks() -> dict[str, Any]:
    """Where to run the networks that read a scene: the GPU, or the CPU where there is none.

    The estimators refuse a CPU unless it is asked for by name, since it is
    far too slow for augmentation. For the few hundred pictures here it is
    slow but enough, and it is what CI has. Half precision is a GPU's.
    """
    import torch

    if torch.cuda.is_available():
        return {}
    print("No CUDA device: running the networks on the CPU, which is slow.")
    return {"device": "cpu", "half_precision": False}


# What the site holds besides the code reference: the folders under docs/
# that the pictures are rendered into, and which of their files to publish.
# The PNG is the banner's copy for GitHub's social preview, which takes no WebP.
PUBLISHED = ("examples", "assets")
SUFFIXES = (".webp", ".png")

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>surmentations gallery</title>
<style>
:root {{ --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; }}
}}
body {{ margin: 0 auto; max-width: 1400px; padding: 0 16px 48px; background: var(--bg);
       color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }}
h1 {{ margin: 24px 0 4px; }} h2 {{ margin-top: 40px; border-bottom: 1px solid var(--line); }}
h3 {{ margin: 20px 0 6px; font: 600 14px ui-monospace, monospace; }}
p, figcaption {{ color: var(--muted); }} a {{ color: inherit; }}
.grid {{ display: grid; gap: 6px; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }}
.row {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
figure {{ margin: 0; }} figcaption {{ font-size: 12px; overflow-wrap: anywhere; }}
img {{ width: 100%; height: auto; display: block; }}
</style>
</head>
<body>
<h1>surmentations</h1>
<p>Every picture of the documentation, rendered by CI from commit
<a href="{repository}/commit/{commit}"><code>{short}</code></a>.
The guides that show them are in <a href="{repository}">the repository</a>;
the <a href="reference/">code reference</a> is generated from its source.</p>
{body}
</body>
</html>
"""


def _figure(site: Path, path: Path, caption: str) -> str:
    """One picture, linked to itself at full size."""
    address = path.relative_to(site).as_posix()
    return (
        f'<figure><a href="{address}"><img src="{address}" loading="lazy" alt="{caption}"></a>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _index(site: Path, repository: str, commit: str) -> str:
    """The site's front page: the severity gallery a row to an effect, then every other folder."""
    import html
    import re

    examples = site / "examples"
    body = ['<h2 id="examples">Severity gallery</h2>',
            "<p>Each effect at severity 1 to 5, left to right, over the reference "
            "photograph; <code>_aware</code> rows are drawn again with a provider "
            "that reads the scene.</p>"]
    rows: dict[str, list[Path]] = {}
    for path in sorted(examples.glob("*_[0-9].webp")):
        rows.setdefault(re.sub(r"_\d$", "", path.stem), []).append(path)
    for row, paths in rows.items():
        cells = "".join(_figure(site, path, f"severity {int(path.stem[-1]) + 1}") for path in paths)
        body += [f'<h3 id="{row}">{html.escape(row)}</h3>', f'<div class="grid row">{cells}</div>']
    folders = sorted(
        {path.parent for path in site.rglob("*.webp")} - {examples, site / "reference"},
        key=lambda folder: folder.relative_to(site).as_posix(),
    )
    for folder in folders:
        if (site / "reference") in folder.parents:
            continue
        name = folder.relative_to(site).as_posix()
        cells = "".join(
            _figure(site, path, html.escape(path.stem)) for path in sorted(folder.glob("*.webp"))
        )
        body += [f'<h2 id="{name}">{html.escape(name)}</h2>', f'<div class="grid">{cells}</div>']
    return _PAGE.format(
        repository=repository, commit=commit, short=commit[:7], body="\n".join(body)
    )


def assemble(site: Path, repository: str, commit: str) -> int:
    """Copy the rendered pictures into ``site`` where :func:`url` says they are, and index them.

    Returns how many pictures were copied. The code reference, if it is
    wanted, goes in ``site/reference`` before or after this.
    """
    import shutil

    copied = 0
    for top in PUBLISHED:
        for path in sorted((DOCS / top).rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                target = site / path.relative_to(DOCS)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
    (site / "index.html").write_text(_index(site, repository, commit))
    (site / ".nojekyll").touch()
    return copied


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=assemble.__doc__)
    parser.add_argument("site", type=Path, help="the directory to assemble the site in")
    parser.add_argument("--repository", default="https://github.com/desmoteo/surmentations")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "HEAD"))
    args = parser.parse_args()
    count = assemble(args.site, args.repository.rstrip("/"), args.commit)
    print(f"assembled {count} pictures into {args.site}")
