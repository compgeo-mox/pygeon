"""Sphinx configuration of the PyGeoN documentation."""

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import bibtexparser

project = "PyGeoN"
author = "Wietse M. Boon, Alessio Fumagalli"
copyright = f"2025, {author}"

extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# the API pages are generated at each build from the package, one page per module
apidoc_modules = [
    {
        "path": "../src/pygeon",
        "destination": "api",
        "separate_modules": True,
        "module_first": True,
    }
]
# the table of contents of apidoc and the README of the tutorials are replaced by the
# toctrees in index.rst
exclude_patterns = [
    "_build",
    "api/modules.rst",
    "tutorials/README.md",
    "**/.ipynb_checkpoints",
]
# the arguments of the constructors are documented in the __init__ docstrings
autodoc_default_options = {"member-order": "bysource", "special-members": "__init__"}
# the types are written in the Args and Returns sections of the docstrings
autodoc_typehints = "none"

# mocked so that the documentation builds without them: they are needed to create
# and export the grids, not to import PyGeoN and read its docstrings
autodoc_mock_imports = ["gmsh", "meshio", "pygmsh", "pyvista", "vtkmodules"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
# PorePy and SymPy do not publish an inventory with the objects used by PyGeoN
nitpick_ignore_regex = [("py:class", r"(pp|porepy|sp)\..*")]

# the tutorials, linked from the repository, are rendered without being executed; as
# in Jupyter, the $$ display math is allowed inside a paragraph
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

html_theme = "furo"
html_title = "PyGeoN"


def resolve_alias(app, env, node, contnode):
    """
    Link the objects written with the import aliases of the code, as pg.Grid,
    pygeon.RT0.interpolate or np.ndarray: pg and pygeon are resolved in the documented
    objects, through the names exported by the package, np and sps by intersphinx.
    """
    prefix, _, name = node["reftarget"].partition(".")
    if prefix in ("np", "sps"):
        # rewritten for the intersphinx handler, called after this one
        node["reftarget"] = {"np": "numpy.", "sps": "scipy.sparse."}[prefix] + name
        return None

    # the first name is the one exported by the package, as a class or a function,
    # the rest are its attributes, as a method
    first, _, rest = name.partition(".")
    obj = getattr(sys.modules["pygeon"], first, None)
    if prefix in ("pg", "pygeon") and hasattr(obj, "__qualname__"):
        target = f"{obj.__module__}.{obj.__qualname__}" + (f".{rest}" if rest else "")
        return env.domains.python_domain.resolve_xref(
            env, node["refdoc"], app.builder, node["reftype"], target, node, contnode
        )
    return None


def copy_tutorial_figures(app, exception):
    """
    Copy the figures of the tutorials next to their pages, since the notebooks show
    them through raw html with paths relative to the notebook, as ../fig/darcy.png.
    """
    if exception is None and app.builder.format == "html":
        src, dst = app.srcdir / "tutorials/fig", app.outdir / "tutorials/fig"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied the figures of the tutorials from {src} to {dst}")


def write_papers(app):
    """
    Write papers.rst from papers.bib, the papers grouped by year and the titles linked
    to the DOI, so that adding a paper only takes its BibTeX entry.
    """
    entries = bibtexparser.parse_file(Path(app.confdir) / "papers.bib").entries
    clean = lambda s: " ".join(s.replace("{", "").replace("}", "").split())

    by_year = defaultdict(list)
    for e in entries:
        authors = "; ".join(a.strip() for a in clean(e["author"]).split(" and "))
        title = clean(e["title"]).rstrip(".")
        if "doi" in e:
            title = f"`{title} <https://doi.org/{e['doi']}>`__"
        journal = e["journal"] if "journal" in e else e["booktitle"]
        line = f"- {authors} ({e['year']}). {title}. *{journal}*."
        if "url" in e:
            line += f" `Code <{e['url']}>`__."
        by_year[e["year"]].append(line)

    years = sorted(by_year, reverse=True)
    out = Path(app.confdir) / "papers.rst"
    out.write_text(
        PAPERS_HEADER
        + "\n\n".join(f"{y}\n{'-' * len(y)}\n\n" + "\n".join(by_year[y]) for y in years)
        + "\n"
    )
    print(f"Written {len(entries)} papers in the years {years} to {out}")


PAPERS_HEADER = """Papers using PyGeoN
===================

If you use PyGeoN in your research, please cite it through its
`DOI <https://zenodo.org/badge/latestdoi/455087135>`_. To add your paper to this list,
open a pull request with its entry in ``docs/papers.bib``.

"""


def setup(app):
    app.connect("builder-inited", write_papers)
    app.connect("missing-reference", resolve_alias, priority=400)
    app.connect("build-finished", copy_tutorial_figures)
