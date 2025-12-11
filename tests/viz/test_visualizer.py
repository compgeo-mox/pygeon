"""Tests for PVD/VTU visualizer."""

import tempfile
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import porepy as pp
import pytest

import pygeon as pg

# Use non-interactive backend for testing
matplotlib.use("Agg")

# Check if PyVista is available
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


@pytest.fixture
def simple_pvd_file():
    """Create a simple PVD file with VTU data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a simple 2D grid
        mesh_size = 0.5
        dim = 2
        sd = pg.unit_grid(dim, mesh_size, as_mdg=False)
        sd.compute_geometry()

        # Create some test data
        cell_p = np.random.rand(sd.num_cells)
        cell_q = np.random.rand(3, sd.num_cells)
        point_u = np.random.rand(sd.num_nodes)

        # Export to VTU
        file_name = "test_sol"
        save = pp.Exporter(sd, file_name, folder_name=str(tmpdir))
        save.write_vtu(
            [(f"cell_p", cell_p), (f"cell_q", cell_q)], data_pt=[(f"point_u", point_u)]
        )

        yield tmpdir, file_name + "_2.vtu", ["cell_p", "cell_q", "point_u"]


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_initialization(simple_pvd_file):
    """Test Visualizer initialization."""
    tmpdir, file_name, fields = simple_pvd_file

    vis1 = pg.Visualizer(2, file_name, folder_name=str(tmpdir))
    vis2 = pg.Visualizer(2, str(tmpdir / file_name))

    for vis in [vis1, vis2]:
        # Check that meshes were loaded
        assert vis.mesh.n_cells == 14
        assert vis.mesh.n_points == 12
        assert vis.mesh.n_arrays == 9

        for field in fields:
            if "cell" in field:
                assert field in vis.mesh.cell_data
            elif "point" in field:
                assert field in vis.mesh.point_data


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_scalar_field(simple_pvd_file):
    """Test scalar field visualization."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    # Mock the show method to prevent display
    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p", cmap="viridis")
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_scalar_field_with_label(simple_pvd_file):
    """Test scalar field with custom label."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p", field_label="Pressure")
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_scalar_field_no_edges(simple_pvd_file):
    """Test scalar field without edges."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p", show_edges=False)
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_vector_field(simple_pvd_file):
    """Test vector field visualization."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_vector_field("cell_q", scaling_factor=0.1)
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_combined_fields(simple_pvd_file):
    """Test combining scalar and vector fields."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_vector_field("cell_q", scaling_factor=0.05)
        vis.plot_scalar_field("cell_p")
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_show_mesh(simple_pvd_file):
    """Test mesh visualization."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_mesh(color="lightblue", show_edges=True)
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_view_options(simple_pvd_file):
    """Test different view options."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p")

        # Test different views
        for view in ["xy", "xz", "yz", "iso"]:
            vis.show(view=view)
            vis.plot_scalar_field("cell_p")


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_with_title(simple_pvd_file):
    """Test visualization with title."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p")
        vis.show(title="Test Visualization")


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_custom_scalar_bar_args(simple_pvd_file):
    """Test custom scalar bar arguments."""
    tmpdir, file_name, _ = simple_pvd_file

    vis = pg.Visualizer(2, file_name, folder_name=str(tmpdir))

    custom_bar_args = {
        "title": "Custom Pressure",
        "vertical": True,
        "position_x": 0.9,
        "position_y": 0.2,
    }

    with mock.patch.object(vis.plotter, "show"):
        vis.plot_scalar_field("cell_p", scalar_bar_args=custom_bar_args)
        vis.show()


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not available")
def test_visualizer_missing_file():
    """Test error handling for missing PVD file."""
    with pytest.raises((FileNotFoundError, Exception)):
        pg.Visualizer("nonexistent_file.pvd")
