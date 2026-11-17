"""Module contains the tests of the Grid class."""

import numpy as np
import porepy as pp

import pygeon as pg


def test_grid_0d(ref_sd_0d):
    shape = ref_sd_0d.edge_nodes.shape
    assert shape == (ref_sd_0d.num_nodes, 0), f"edge_nodes has shape {shape}"
    assert len(ref_sd_0d.edge_tangents) == 0
    assert len(ref_sd_0d.edge_lengths) == 0
    assert ref_sd_0d.mesh_size == 0


def test_cartgrids(unit_cart_sd):
    h = unit_cart_sd.edge_lengths[0]
    assert np.allclose(unit_cart_sd.edge_lengths, h)
    assert np.allclose(unit_cart_sd.mesh_size, h)


def test_grid_2d_tris():
    sd = pp.StructuredTriangleGrid([1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 3))

    assert np.allclose(unique_lengths, known_lengths)
    assert np.allclose(sd.edge_lengths, sd.face_areas)
    assert np.allclose(np.sum(sd.edge_tangents * sd.face_normals, axis=0), 0)

    known_meshsize = (4 + np.sqrt(2)) / 5
    assert np.allclose(sd.mesh_size, known_meshsize)


def test_grid_3d_tets():
    sd = pp.StructuredTetrahedralGrid([1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    unique_lengths = np.unique(sd.edge_lengths)
    known_lengths = np.sqrt(np.arange(1, 4))

    assert np.allclose(unique_lengths, known_lengths)

    known_meshsize = (12 + 6 * np.sqrt(2) + np.sqrt(3)) / 19
    assert np.allclose(sd.mesh_size, known_meshsize)


def test_grid_1d_in_mdg(mdg):
    # The faces of the fracture grids are not numbered as their nodes, so that the
    # tangent of a cell goes from the node of its incoming face to the one of its
    # outgoing face
    for sd in mdg.subdomains(dim=1):
        start = sd.face_nodes.indices[sd.cell_faces.indices[sd.cell_faces.data < 0]]
        end = sd.face_nodes.indices[sd.cell_faces.indices[sd.cell_faces.data > 0]]
        edge_start = sd.edge_nodes.indices[sd.edge_nodes.data < 0]
        edge_end = sd.edge_nodes.indices[sd.edge_nodes.data > 0]
        assert np.array_equal(edge_start, start), f"Starts {edge_start} != {start}"
        assert np.array_equal(edge_end, end), f"Ends {edge_end} != {end}"

        diff = sd.edge_tangents - (sd.nodes[:, end] - sd.nodes[:, start])
        assert np.allclose(diff, 0), f"Tangents differ by {abs(diff).max()}"
        diff = sd.edge_lengths - sd.cell_volumes
        assert np.allclose(diff, 0), f"Lengths differ by {abs(diff).max()}"
