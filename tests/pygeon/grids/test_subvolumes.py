"""Module contains sub volume tests."""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def test_quads(unit_cart_sd):
    sub_volumes, sub_simplices = unit_cart_sd.compute_subvolumes(True)

    assert np.allclose(unit_cart_sd.cell_volumes, np.sum(sub_volumes, 0))

    sub_simplices_known = unit_cart_sd.cell_faces.copy().astype(float)
    factor = [0, 2, 4, 6]
    sub_simplices_known.data[:] = (
        unit_cart_sd.cell_volumes[0] / factor[unit_cart_sd.dim]
    )

    assert np.allclose((sub_simplices - sub_simplices_known).data, 0)


def test_simplices(unit_sd):
    sub_volumes, sub_simplices = unit_sd.compute_subvolumes(True)

    assert np.allclose(unit_sd.cell_volumes, np.sum(sub_volumes, 0))
    assert np.allclose(unit_sd.cell_volumes, np.sum(sub_simplices, 0))


def test_poly_grids(unit_poly_sd):
    sub_volumes, sub_simplices = unit_poly_sd.compute_subvolumes(True)

    assert np.allclose(unit_poly_sd.cell_volumes, np.sum(sub_volumes, 0))
    assert np.allclose(unit_poly_sd.cell_volumes, np.sum(sub_simplices, 0))


def test_hitchhiker_pentagon(pentagon_sd):
    sub_volumes = pentagon_sd.compute_subvolumes()
    assert np.allclose(pentagon_sd.cell_volumes, np.sum(sub_volumes, 0))


def test_concave_quad():
    nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    face_nodes = sps.csc_array((np.ones(8), indices, np.arange(0, 9, 2)))
    cell_faces = sps.csc_array(np.ones((4, 1)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    sub_volumes = sd.compute_subvolumes()
    assert np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0))


def test_multiple_concave_quads():
    nodes = np.array(
        [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
    )
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0])
    face_nodes = sps.csc_array((np.ones(16), indices, np.arange(0, 17, 2)))
    cell_faces_j = np.repeat(np.arange(3), 4)
    cell_faces_i = np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7])
    cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
    cell_faces = sps.csc_array((cell_faces_v, (cell_faces_i, cell_faces_j)))

    sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    sub_volumes = sd.compute_subvolumes()

    known_sub_volumes = np.array(
        [
            [6.25e-02, 1.25e-01, 6.25e-02],
            [0, 1.25e-01, 0],
            [6.25e-02, 1.25e-01, 6.25e-02],
            [1.25e-01, 0, 0],
            [0, 1.25e-01, 0],
            [0, 0, 1.25e-01],
        ]
    )
    assert np.allclose(sub_volumes.todense(), known_sub_volumes)
