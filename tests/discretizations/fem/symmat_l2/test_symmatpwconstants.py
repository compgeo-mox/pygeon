"""Module contains specific tests for the symmetric matrix P0 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.SymMatPwConstants("test")


def test_mat_invert(discr, unit_sd):

    fun = lambda _: np.array([[1, 2, 3], [2, 5, 6], [3, 6, 10]])
    mat = discr.interpolate(unit_sd, fun)
    proj = discr.eval_at_cell_centers(unit_sd)

    inv_mat = discr.mat_invert(unit_sd, mat)

    val_mat = proj @ mat
    val_inv_mat = proj @ inv_mat

    val_mat_reshaped = val_mat.reshape((pg.AMBIENT_DIM, pg.AMBIENT_DIM, -1))
    val_mat_reshaped = np.transpose(val_mat_reshaped, (2, 0, 1))

    val_inv_mat_reshaped = val_inv_mat.reshape((pg.AMBIENT_DIM, pg.AMBIENT_DIM, -1))
    val_inv_mat_reshaped = np.transpose(val_inv_mat_reshaped, (2, 0, 1))

    eye = np.diag([1] * unit_sd.dim + [0] * (pg.AMBIENT_DIM - unit_sd.dim))

    assert np.allclose(val_mat_reshaped @ val_inv_mat_reshaped, eye)
    assert np.allclose(val_inv_mat_reshaped @ val_mat_reshaped, eye)
