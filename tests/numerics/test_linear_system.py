"""Module contains tests to validate the linear system module."""

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def test_simple_problem():
    """Coverage test"""
    A = sps.eye(4, 4, format="csc")
    b = np.tile(np.arange(4), (3, 1)).T

    LS = pg.LinearSystem(A, b)
    sol = LS.solve()

    assert np.allclose(sol, b)

    ess_dofs = np.array([True, False, False, True])
    LS.flag_ess_bc(ess_dofs, np.ones_like(ess_dofs))

    sol = LS.solve()

    assert np.allclose(sol, np.tile([1, 1, 2, 1], (3, 1)).T)
