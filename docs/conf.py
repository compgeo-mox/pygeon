"""Sphinx configuration of the PyGeoN documentation."""

import shutil
import sys

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

# the optional dependencies of PorePy are not needed to import PyGeoN
autodoc_mock_imports = ["gmsh", "meshio", "pygmsh", "pyvista", "vtkmodules"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
# PorePy and SymPy do not publish an inventory with the objects used by PyGeoN, and
# optional is the Google style mark of the arguments with a default
nitpick_ignore_regex = [("py:class", r"(pp|porepy|sp)\..*|optional")]

# the tutorials, linked from the repository, are rendered without being executed; as
# in Jupyter, the $$ display math is allowed inside a paragraph
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

html_theme = "furo"
html_title = "PyGeoN"


def resolve_alias(app, env, node, contnode):
    """
    Link the types written with the import aliases of the code, as pg.Grid or
    np.ndarray: pg and pygeon are resolved in the documented objects, through the
    names exported by the package, np and sps by intersphinx.
    """
    prefix, _, name = node["reftarget"].partition(".")
    if prefix in ("np", "sps"):
        # rewritten for the intersphinx handler, called after this one
        node["reftarget"] = {"np": "numpy.", "sps": "scipy.sparse."}[prefix] + name
    elif prefix in ("pg", "pygeon") and hasattr(sys.modules["pygeon"], name):
        obj = getattr(sys.modules["pygeon"], name)
        node["reftarget"] = f"{obj.__module__}.{obj.__qualname__}"
        return env.domains.python_domain.resolve_xref(
            env, node["refdoc"], app.builder, "class", node["reftarget"], node, contnode
        )


def copy_tutorial_figures(app, exception):
    """
    Copy the figures of the tutorials next to their pages, since the notebooks show
    them through raw html with paths relative to the notebook, as ../fig/darcy.png.
    """
    if exception is None and app.builder.format == "html":
        src, dst = app.srcdir / "tutorials/fig", app.outdir / "tutorials/fig"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied the figures of the tutorials from {src} to {dst}")


def setup(app):
    app.connect("missing-reference", resolve_alias, priority=400)
    app.connect("build-finished", copy_tutorial_figures)
