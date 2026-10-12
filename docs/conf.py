"""Sphinx configuration for the ArbEPI-python documentation.

The prose is the repo's own READMEs, included (not copied) by the thin wrapper
pages in this directory, so it cannot drift from the code's docs. The API
reference is generated from docstrings by autosummary. Build:

    uv run --extra docs sphinx-build -b html docs docs/_build/html
"""

import posixpath
import re
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

project = "ArbEPI-python"
author = "Rex Fung"
copyright = "2026, Rex Fung"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# -- Markdown / notebooks ----------------------------------------------------
# .rst first: autosummary writes its generated pages with the first suffix.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 4  # GitHub-style #anchors in the READMEs resolve
# The demo notebooks need a GPU, scanner data or GERecon: show saved outputs.
nb_execution_mode = "off"

# Sphinx only reads files under the source directory, so the demo notebooks are
# copied next to the pages that link them (docs/_notebooks/, gitignored).
NOTEBOOKS = {
    "demo.ipynb": "sequence_demo.ipynb",
    "preprocess/demo.ipynb": "preprocess_demo.ipynb",
    "recon/demo.ipynb": "recon_demo.ipynb",
    "simulate_fmri/demo.ipynb": "simulate_fmri_demo.ipynb",
}
(DOCS / "_notebooks").mkdir(exist_ok=True)
for src, dst in NOTEBOOKS.items():
    shutil.copyfile(ROOT / src, DOCS / "_notebooks" / dst)

# -- API reference -----------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Docstrings use single backticks for code (`pp.scale_grad`); RST's default
# role would render those as italic title references.
default_role = "code"

# Only what a docs build cannot install. numba is deliberately not mocked: a
# mocked @njit turns pd_sample's functions into Mock objects.
autodoc_mock_imports = [
    "torch",
    "mirtorch",
    "sigpy",
    "GERecon",
    "snake",
    "ismrmrd",
    "h5py",
    "nibabel",
    "pywt",
    "brainextractor",
    "cupy",
]

# -- README pages ------------------------------------------------------------
# docs/<page>.md holds one line, `<!-- readme: <path from repo root> -->`, and
# the source-read hook below swaps in that file's text. {include} would resolve
# the READMEs' relative links against docs/ instead of the README's directory.
GITHUB = "https://github.com/rextlfung/ArbEPI-python/blob/main/"
README_PAGES = {
    "README.md": "overview",
    **{f"{d}/README.md": d for d in ("ge", "preprocess", "recon", "analyze", "simulate_fmri")},
}
_MARKER = re.compile(r"^<!-- readme: (\S+) -->")
_PICTURE = re.compile(r"<picture>.*?</picture>", re.S)
_ALT = re.compile(r'<img alt="([^"]*)"')
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def _link_fixer(readme_dir):
    def fix(m):
        text, target = m.groups()
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        path, _, frag = target.partition("#")
        frag = f"#{frag}" if frag else ""
        full = posixpath.normpath(posixpath.join(readme_dir, path))
        if full in README_PAGES:  # another README: its page
            return f"[{text}]({README_PAGES[full]}.md{frag})"
        if full.startswith(".."):  # sibling checkout (../ArbEPI): nothing to link
            return text
        return f"[{text}]({GITHUB}{full}{frag})"

    return fix


def _expand_readme(app, docname, source):
    m = _MARKER.match(source[0])
    if not m:
        return
    rel = m.group(1)
    text = (ROOT / rel).read_text()

    def picture(p):
        alt = _ALT.search(p.group(0))
        alt = alt.group(1) if alt else "pipeline"
        return "\n".join(
            f"```{{image}} pipeline/pipeline-{mode}.svg\n:alt: {alt}\n:class: only-{mode}\n```"
            for mode in ("light", "dark")
        )

    text = _PICTURE.sub(picture, text)
    source[0] = _LINK.sub(_link_fixer(posixpath.dirname(rel)), text)


def setup(app):
    app.connect("source-read", _expand_readme)


# Annotations such as `h5py.Dataset | h5py.Group` or `torch.device | None` are
# evaluated at import; autodoc's mock objects don't implement `|`.
from sphinx.ext.autodoc.mock import _MockObject  # noqa: E402

_MockObject.__or__ = lambda self, other: self
_MockObject.__ror__ = lambda self, other: self

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Output ------------------------------------------------------------------
# review-findings.md and TODO.md are worklogs, not user documentation.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "pipeline/make_pipeline.py",
    "review-findings.md",
    "TODO.md",
]
html_theme = "furo"
html_title = "ArbEPI-python"
html_theme_options = {
    "source_repository": "https://github.com/rextlfung/ArbEPI-python",
    "source_branch": "main",
    "source_directory": "docs/",
}
