"""Tests for VTU visualizer."""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture(
    params=["unit_sd_2d", "unit_sd_3d", "octagon_sd_2d", "cart_sd_2d"], scope="module"
)
def grid_to_visualize(request: pytest.FixtureRequest):
    # resolve the underlying fixture by name
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def simple_vtu_file(grid_to_visualize):
    """Create a simple VTU data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some test data
        cell_scalar = np.random.rand(grid_to_visualize.num_cells)
        cell_vector = np.random.rand(3, grid_to_visualize.num_cells)
        point_scalar = np.random.rand(grid_to_visualize.num_nodes)
        point_vector = np.random.rand(3, grid_to_visualize.num_nodes)

        # Export to VTU
        file_name = "test_sol"
        save = pp.Exporter(grid_to_visualize, file_name, folder_name=str(tmpdir))
        save.write_vtu(
            [(f"cell_scalar", cell_scalar), (f"cell_vector", cell_vector)],
            data_pt=[(f"point_scalar", point_scalar), (f"point_vector", point_vector)],
        )

        file_name += "_" + str(grid_to_visualize.dim) + ".vtu"
        vis = pg.Visualizer(file_name, folder_name=str(tmpdir))

        fields = ["cell_scalar", "cell_vector", "point_scalar", "point_vector"]
        with mock.patch.object(vis.plotter, "show"):
            yield vis, grid_to_visualize, fields, tmpdir


def test_visualizer_initialization(simple_vtu_file):
    """Test Visualizer initialization."""
    vis, sd, fields, _ = simple_vtu_file

    # Check that meshes were loaded
    assert vis.mesh.n_cells == sd.num_cells
    assert vis.mesh.n_points == sd.num_nodes
    assert vis.mesh.n_arrays == 10

    for field in fields:
        if "cell" in field:
            assert field in vis.mesh.cell_data
        elif "point" in field:
            assert field in vis.mesh.point_data


def test_visualizer_scalar_field(simple_vtu_file):
    """Test scalar field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("cell_scalar", cmap="viridis")
    vis.show()


def test_visualizer_point_scalar_field(simple_vtu_file):
    """Test point scalar field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("point_scalar")
    vis.show()


def test_visualizer_scalar_field_with_label(simple_vtu_file):
    """Test scalar field with custom label."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("cell_scalar", field_label="Pressure")
    vis.show()


def test_visualizer_scalar_field_no_edges(simple_vtu_file):
    """Test scalar field without edges."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("cell_scalar", show_edges=False)
    vis.show()


def test_visualizer_vector_field(simple_vtu_file):
    """Test vector field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_vector_field("cell_vector", scaling_factor=0.1)
    vis.show()


def test_visualizer_point_vector_field(simple_vtu_file):
    """Test point vector field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_vector_field("point_vector", scaling_factor=0.1)
    vis.show()


def test_visualizer_contour_scalar_field(simple_vtu_file):
    """Test scalar field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_contour("cell_scalar")
    vis.show()


def test_visualizer_contour_point_scalar_field(simple_vtu_file):
    """Test point scalar field visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_contour("point_scalar", isosurfaces=5)
    vis.show()


def test_visualizer_combined_fields(simple_vtu_file):
    """Test combining scalar and vector fields."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_vector_field("cell_vector", scaling_factor=0.05)
    vis.plot_scalar_field("cell_scalar")
    vis.show()


def test_visualizer_show_mesh(simple_vtu_file):
    """Test mesh visualization."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_mesh(color="lightblue", show_edges=True)
    vis.show()


@pytest.mark.parametrize("view", ["xy", "xz", "yz", "iso"])
def test_visualizer_view_options(simple_vtu_file, view):
    """Test different view options."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("cell_scalar")
    vis.show(view=view)


def test_visualizer_with_title(simple_vtu_file):
    """Test visualization with title."""
    vis, _, _, _ = simple_vtu_file
    vis.plot_scalar_field("cell_scalar")
    vis.show(title="Test Visualization")


def test_visualizer_custom_scalar_bar_args(simple_vtu_file):
    """Test custom scalar bar arguments."""
    vis, _, _, _ = simple_vtu_file
    custom_bar_args = {
        "title": "Custom Pressure",
        "vertical": True,
        "position_x": 0.9,
        "position_y": 0.2,
    }
    vis.plot_scalar_field("cell_scalar", scalar_bar_args=custom_bar_args)
    vis.show()


def test_visualizer_missing_file():
    """Test error handling for missing PVD file."""
    with pytest.raises((FileNotFoundError, Exception)):
        pg.Visualizer("nonexistent_file.pvd")


@pytest.mark.parametrize("file_name", ["fig.png", "fig.eps", "fig.svg"])
def test_visualizer_save(simple_vtu_file, file_name):
    """Test save image."""

    vis, _, _, fig_path = simple_vtu_file
    vis.plot_scalar_field("cell_scalar")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vis.show(screenshot=tmpdir / file_name)

        # Check that file was created
        assert (tmpdir / file_name).stat().st_size > 0


# def test_visualizer_notebook_backend(simple_vtu_file):
#     """Test jupyter notebook visualization with static backend."""
#     vis, _, _, _ = simple_vtu_file
#     vis.plot_scalar_field("cell_scalar")

#     # Mock plotter.notebook to True to trigger jupyter_backend="static" path
#     with mock.patch.object(vis.plotter, "notebook", True):
#         vis.show()


# @pytest.mark.parametrize(
#     "latex_available, expected_family",
#     [(True, "times"), (False, "arial")],
# )
# def test_visualizer_latex_detection(monkeypatch, latex_available, expected_family):
#     """Ensure LaTeX detection sets the correct font family for both branches."""
#     import pygeon.viz.visualizer as viz_module

#     # Mock Latex availability
#     monkeypatch.setattr(
#         viz_module.shutil,
#         "which",
#         lambda _: "/usr/bin/latex" if latex_available else None,
#     )

#     # Minimal fake mesh to satisfy Visualizer.__init__
#     class _FakeMesh:
#         def GetMaxSpatialDimension(self):
#             return 3

#     # Mock pv.read to return fake mesh
#     monkeypatch.setattr(viz_module.pv, "read", lambda _: _FakeMesh())

#     # Construct Visualizer and verify font family
#     viz_module.Visualizer("dummy.vtu")
#     assert viz_module.pv.global_theme.font.family == expected_family
