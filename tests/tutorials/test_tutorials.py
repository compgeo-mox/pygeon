# Taken from PorePy
import glob
import os
import sys
import types

import nbformat
import pytest
from nbconvert import NotebookExporter
from nbconvert.preprocessors import Preprocessor

TUTORIAL_FILENAMES = glob.glob("tutorials/*.ipynb")


class IgnoreCommentPreprocessor(Preprocessor):
    """
    Remove any code cell whose first line starts with '# NBIGNORE'
    """

    def preprocess(self, nb, resources):
        new_cells = []
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.source.strip().startswith(
                "# NBIGNORE"
            ):
                continue
            new_cells.append(cell)
        nb.cells = new_cells
        return nb, resources


@pytest.mark.tutorials
@pytest.mark.parametrize("tutorial_path", TUTORIAL_FILENAMES)
def test_run_tutorials(tutorial_path: str):
    """Run the tutorial and check that it didn't raise any error."""
    new_file = tutorial_path.removesuffix(".ipynb") + ".py"

    # --- Convert notebook to script with IgnoreCommentPreprocessor ---
    nb = nbformat.read(tutorial_path, as_version=4)
    exporter = NotebookExporter()
    exporter.register_preprocessor(IgnoreCommentPreprocessor, enabled=True)
    body, _ = exporter.from_notebook_node(nb)

    with open(new_file, "w", encoding="utf-8") as f:
        f.write(body)

    # --- Patch imports and disable interactive plotting ---
    edit_imports(new_file)

    # --- Run the generated Python script ---
    cmd_run = f"python {new_file}"
    status = os.system(cmd_run)
    assert status == 0

    # --- Clean up generated script ---
    os.remove(new_file)


def edit_imports(filename: str):
    """Patch the generated script to avoid opening windows with Matplotlib/VTK."""
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()

    with open(filename, "w", encoding="utf-8") as f:
        # Disable VTK/PyVista offscreen rendering
        f.write("import os\n")
        f.write("os.environ['PYVISTA_OFF_SCREEN'] = 'true'\n")
        f.write("os.environ['PYVISTA_USE_PANEL'] = 'false'\n")
        f.write("os.environ.pop('DISPLAY', None)\n")

        # Ensure we are in tutorials directory
        f.write("os.chdir('./tutorials')\n")

        # Disable matplotlib windows
        f.write("import matplotlib; matplotlib.use('template')\n")

        f.writelines(content)


if __name__ == "__main__":
    try:
        filenames = [sys.argv[1]]
    except IndexError:
        filenames = TUTORIAL_FILENAMES
    for tut_path in filenames:
        test_run_tutorials(tutorial_path=tut_path)
