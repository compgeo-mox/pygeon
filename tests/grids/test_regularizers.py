"""Module testing the grid regularizers."""

import numpy as np
import pytest

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

    # Topology is preserved
    assert (sd.face_ridges - sd_voronoi.face_ridges).nnz == 0
    assert (sd.cell_faces - sd_voronoi.cell_faces).nnz == 0

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_graph_laplace_cart_grids(unit_cart_sd):
    sd = pg.graph_laplace_regularization(unit_cart_sd, False)

    # The regularization leaves regular Cartesian grids intact
    assert np.allclose(sd.nodes, unit_cart_sd.nodes)
    assert np.allclose(sd.cell_volumes, unit_cart_sd.cell_volumes)


def test_graph_laplace_dual(sd_voronoi):
    sd = pg.graph_laplace_dual_regularization(sd_voronoi)

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)


def test_elasticity_regulatization(sd_voronoi):
    sd = pg.elasticity_regularization(sd_voronoi, 10, sliding=False)

    # Aspect ratios have improved
    assert aspect_ratio(sd) < aspect_ratio(sd_voronoi)
