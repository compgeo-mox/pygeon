# tests/run_tutorials.py  (replace your current file with this)
import glob
import os
import sys

import nbformat
import pytest
from nbconvert import ScriptExporter
from nbconvert.preprocessors import Preprocessor

TUTORIAL_FILENAMES = glob.glob("tutorials/*.ipynb")


class IgnoreCommentPreprocessor(Preprocessor):
    """
    Remove any code cell whose first non-empty line starts with '# NBIGNORE'.

    Implementing preprocess_cell is often more reliable for exporters,
    and we'll also call preprocess explicitly as an extra guarantee.
    """

    def preprocess_cell(self, cell, resources, index):
        # Only act on code cells
        if cell.cell_type != "code":
            return cell, resources

        # find first non-empty line
        lines = [ln for ln in cell.source.splitlines() if ln.strip() != ""]
        if not lines:
            return cell, resources

        first = lines[0].lstrip()
        if first.startswith("# NBIGNORE"):
            # Signal that this cell should be skipped by returning an empty source
            # We'll mark it with a special metadata flag so the caller can drop it.
            # Alternatively we'll return a sentinel in resources, but here we set metadata.
            cell.metadata["nb_ignored_by_comment"] = True
            # return an empty cell (we'll filter later)
            cell.source = ""
        return cell, resources

    def preprocess(self, nb, resources):
        # run preprocess_cell on all cells and filter out marked ones
        new_cells = []
        for index, cell in enumerate(nb.cells):
            cell, resources = self.preprocess_cell(cell, resources, index)
            # exclude cells either explicitly marked, or now empty code cells
            if cell.cell_type == "code" and cell.metadata.get("nb_ignored_by_comment"):
                # skip it
                continue
            if cell.cell_type == "code" and cell.source.strip() == "":
                # skip empty code cells (safe to drop)
                continue
            new_cells.append(cell)
        nb.cells = new_cells
        return nb, resources


def edit_imports(filename: str):
    """Patch the generated script to avoid opening windows with Matplotlib/VTK."""
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()

    with open(filename, "w", encoding="utf-8") as f:
        # Disable VTK/PyVista offscreen rendering early
        f.write("import os\n")
        f.write("os.environ['PYVISTA_OFF_SCREEN'] = 'true'\n")
        f.write("os.environ['PYVISTA_USE_PANEL'] = 'false'\n")
        f.write("os.environ.pop('DISPLAY', None)\n")

        # Ensure we are in tutorials directory (so relative paths inside tutorials work)
        f.write("os.chdir('./tutorials')\n")

        # Disable matplotlib windows
        f.write("import matplotlib; matplotlib.use('template')\n")

        # Optional debug marker so you can check the generated file once
        # remove/comment out the following line in production:
        f.write("print('>>> patched for CI: offscreen mode set')\n\n")

        f.writelines(content)


@pytest.mark.tutorials
@pytest.mark.parametrize("tutorial_path", TUTORIAL_FILENAMES)
def test_run_tutorials(tutorial_path: str):
    """Run the tutorial and check that it didn't raise any error."""
    new_file = tutorial_path.removesuffix(".ipynb") + ".py"

    # Read the notebook
    nb = nbformat.read(tutorial_path, as_version=4)

    # Normalize to avoid MissingIDFieldWarning
    if hasattr(nbformat, "normalize"):
        nb = nbformat.normalize(nb)

    # --- Apply IgnoreCommentPreprocessor explicitly (guaranteed) ---
    preproc = IgnoreCommentPreprocessor()
    nb, _ = preproc.preprocess(nb, {})

    # --- Export to Python script using ScriptExporter ---
    exporter = ScriptExporter()
    body, _ = exporter.from_notebook_node(nb)

    # Write out the generated script
    with open(new_file, "w", encoding="utf-8") as f:
        f.write(body)

    # Patch imports to avoid opening windows
    edit_imports(new_file)

    # Run the generated Python script
    cmd_run = f"python {new_file}"
    status = os.system(cmd_run)
    assert status == 0

    # Clean up generated script (keep if you want to debug a failing test)
    os.remove(new_file)


if __name__ == "__main__":
    try:
        filenames = [sys.argv[1]]
    except IndexError:
        filenames = TUTORIAL_FILENAMES
    for tut_path in filenames:
        test_run_tutorials(tutorial_path=tut_path)
