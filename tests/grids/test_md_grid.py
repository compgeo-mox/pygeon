"""Module contains tests to validate geometrical computations of mixed-dimensional
grid."""

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def test_faces_and_ridges(mdg_embedded_frac_2d):
    assert mdg_embedded_frac_2d.num_subdomain_faces() == 71
    assert mdg_embedded_frac_2d.num_subdomain_ridges() == 28


def test_remove_tip_faces(mdg_embedded_frac_2d):
    mdg = mdg_embedded_frac_2d
    P = pg.remove_tip_dofs(mdg, 1)

    assert np.allclose(P.data, 1)
    assert P.shape == (69, 71)


def test_remove_tip_cells(mdg_embedded_frac_2d):
    P = pg.remove_tip_dofs(mdg_embedded_frac_2d, 0)
    P_known = sps.eye_array(*P.shape)

    assert np.allclose((P - P_known).data, 0)
