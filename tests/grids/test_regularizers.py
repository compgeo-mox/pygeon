"""Module testing the grid regularizers."""

import numpy as np
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture()
def sd_voronoi() -> pg.Grid:
    sd = pg.VoronoiGrid(30, seed=0)
    sd.compute_geometry()

    return sd


def aspect_ratio(sd: pg.Grid) -> float:
    return np.mean(sd.cell_diameters() ** 2 / sd.cell_volumes)


def test_lloyd(sd_voronoi):
    sd = pg.lloyd_regularization(sd_voronoi, 15)

    assert sd.num_cells == sd_voronoi.num_cells

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_graph_laplace(sd_voronoi):
    sd = pg.graph_laplace_regularization(sd_voronoi)
    sd.compute_geometry()

    # Topology is preserved
    assert (sd.face_ridges - sd_voronoi.face_ridges).nnz == 0
    assert (sd.cell_faces - sd_voronoi.cell_faces).nnz == 0

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_graph_laplace_cart_grids(unit_cart_sd):
    sd = pg.graph_laplace_regularization(unit_cart_sd, False)
    sd.compute_geometry()

    # The regularization leaves regular Cartesian grids intact
    assert np.allclose(sd.nodes, unit_cart_sd.nodes)
    assert np.allclose(sd.cell_volumes, unit_cart_sd.cell_volumes)


def test_graph_laplace_1d_faces_not_nodes():
    # A grid on [0, 1] with the faces numbered differently from their nodes: the
    # regularized nodes are equispaced
    x = np.array([0.0, 0.1, 0.5, 1.0])
    nodes = np.vstack((x, np.zeros((2, x.size))))
    face_to_node = np.array([2, 0, 3, 1])
    face_nodes = sps.csc_array(
        (np.ones(x.size, dtype=bool), (face_to_node, np.arange(x.size)))
    )
    node_to_face = np.argsort(face_to_node)
    cells = np.arange(x.size - 1)
    cell_faces = sps.csc_array(
        (
            np.tile([-1, 1], cells.size),
            (
                np.column_stack((node_to_face[:-1], node_to_face[1:])).ravel(),
                np.repeat(cells, 2),
            ),
        )
    )
    sd = pg.Grid(1, nodes, face_nodes, cell_faces, "faces not nodes")
    sd.compute_geometry()

    sd = pg.graph_laplace_regularization(sd, False)
    assert np.allclose(sd.nodes[0], np.linspace(0, 1, x.size))


def test_graph_laplace_dual(sd_voronoi):
    sd = pg.graph_laplace_dual_regularization(sd_voronoi)
    sd.compute_geometry()

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_elasticity_regulatization(sd_voronoi):
    sd = pg.elasticity_regularization(sd_voronoi, 10, sliding=False)
    sd.compute_geometry()

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_point_grid(ref_sd_0d):
    with pytest.raises(ValueError):
        pg.graph_laplace_regularization(ref_sd_0d)
