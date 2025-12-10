"""Module contains specific tests for the Nedelec 0 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.Nedelec0:
    return pg.Nedelec0("test")


def test_ndof(discr, ref_sd_3d):
    assert discr.ndof(ref_sd_3d) == ref_sd_3d.num_ridges


def test_assemble_mass(discr, ref_sd_3d):
    M = discr.assemble_mass_matrix(ref_sd_3d)

    M_known = (
        np.array(
            [
                [10, 5, 5, 0, 0, 0],
                [5, 10, 5, 0, 0, 0],
                [5, 5, 10, 0, 0, 0],
                [0, 0, 0, 4, 1, -1],
                [0, 0, 0, 1, 4, 1],
                [0, 0, 0, -1, 1, 4],
            ]
        )
        / 120
    )

    assert np.allclose(M.todense(), M_known)


def test_assemble_lumped(discr, ref_sd_3d):
    L = discr.assemble_lumped_matrix(ref_sd_3d)

    L_known = (
        np.array(
            [
                [10, 0, 0, 0, 0, 0],
                [0, 10, 0, 0, 0, 0],
                [0, 0, 10, 0, 0, 0],
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 2],
            ]
        )
        / 60
    )

    assert np.allclose(L.todense(), L_known)


def test_lumped_consistency(discr, ref_sd_3d):
    M_lumped = discr.assemble_lumped_matrix(ref_sd_3d)
    M_full = discr.assemble_mass_matrix(ref_sd_3d)

    one_interp = discr.interpolate(ref_sd_3d, lambda _: np.ones(3))

    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


def test_error_l2(discr, unit_sd_3d):
    def fun(x):
        return np.array([x[0] + 2 * x[1] - x[2], 2 * x[0] - x[1], 6 * x[2]])

    int_sol = discr.interpolate(unit_sd_3d, fun)

    err = discr.error_l2(unit_sd_3d, np.zeros_like(int_sol), fun)
    assert np.isclose(err, 1)

    err = discr.error_l2(unit_sd_3d, int_sol, fun)
    assert np.isclose(err, 0)
