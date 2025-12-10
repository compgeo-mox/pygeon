"""Module testing the grid regularizers."""

import os

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture()
def sd_voronoi() -> pg.Grid:
    sd = pg.VoronoiGrid(30, seed=0)
    sd.compute_geometry()

    return sd


def test_lloyd(sd_voronoi):
    sd = pg.lloyd_regularization(sd_voronoi, 15)

    assert sd.num_cells == 30


def test_graph_laplace(sd_voronoi):
    sd = pg.graph_laplace_regularization(sd_voronoi)

    assert (sd.face_ridges - sd_voronoi.face_ridges).nnz == 0
    assert (sd.cell_faces - sd_voronoi.cell_faces).nnz == 0


def wip_test_graph_laplace_with_centers(sd_voronoi):
    sd = pg.graph_laplace_regularization_with_centers(sd_voronoi)

    assert (sd.face_ridges - sd_voronoi.face_ridges).nnz == 0
    assert (sd.cell_faces - sd_voronoi.cell_faces).nnz == 0


def wip_test_elasticity():
    sd = pg.VoronoiGrid(100, seed=0)
    sd.compute_geometry()

    cond_old = compute_cond(sd)

    pp.plot_grid(sd, alpha=0)
    for _ in np.arange(10):
        sd = pg.elasticity_regularization(sd, 5e-2)
    pp.plot_grid(sd, alpha=0)

    cond_new = compute_cond(sd)

    assert cond_old > cond_new


def wip_test_einstein():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, "einstein_svg_grids")
    file_name = os.path.join(folder, "H2.svg")

    sd = pg.EinSteinGrid(file_name)
    sd.compute_geometry()
    sd = pg.graph_laplace_regularization(sd, False)

    pass


def compute_cond(sd):
    discr = pg.VLagrange1("dumb")
    A = discr.assemble_stiff_matrix(sd)

    ew1 = sps.linalg.eigsh(A, 1, which="LM", return_eigenvectors=False)
    ew2 = sps.linalg.eigsh(A, 2, which="SM", return_eigenvectors=False).max()

    return ew1 / ew2
