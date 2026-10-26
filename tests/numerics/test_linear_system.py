"""Module contains tests to validate the linear system module."""

import numpy as np
import scipy.sparse as sps

import pygeon as pg


def test_simple_problem():
    """Coverage test"""
    A = sps.eye(4, 4).tocsc()
    b = np.tile(np.arange(4), (3, 1)).T

    LS = pg.LinearSystem(A, b)
    sol = LS.solve()

    assert np.allclose(sol, b)

    ess_dofs = np.array([True, False, False, True])
    LS.flag_ess_bc(ess_dofs, np.ones_like(ess_dofs))

    sol = LS.solve()

    assert np.allclose(sol, np.tile([1, 1, 2, 1], (3, 1)).T)


def test_all_dofs_essential():
    # The reduced system is empty, the solver must not be called
    A = sps.eye_array(3).tocsc()
    LS = pg.LinearSystem(A, np.arange(3.0))
    LS.flag_ess_bc(np.ones(3, dtype=bool), np.array([1.0, 2.0, 3.0]))

    def solver(*_):
        raise AssertionError(
            "LinearSystem.solve called the solver on the empty reduced system, "
            "although all the degrees of freedom are essential"
        )

    assert np.allclose(LS.solve(solver), [1.0, 2.0, 3.0])
