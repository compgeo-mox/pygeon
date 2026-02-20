"""Module contains specific tests for the Nedelec 0 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.Nedelec0:
    return pg.Nedelec0("test")


def test_ndof(discr, ref_sd):
    assert discr.ndof(ref_sd) == ref_sd.dim * (ref_sd.dim + 1) // 2


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = np.array([[1]])
        case 2:
            M_known = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 2, 1],
                        [0, 1, 2],
                    ]
                )
                / 6
            )
        case 3:
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


def test_assemble_lumped(discr, ref_sd):
    L = discr.assemble_lumped_matrix(ref_sd)
    match ref_sd.dim:
        case 1:
            L_known = np.array([[1]])
        case 2:
            L_known = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 3, 0],
                        [0, 0, 3],
                    ]
                )
                / 6
            )
        case 3:
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


def test_lumped_consistency(discr, ref_sd):
    M_lumped = discr.assemble_lumped_matrix(ref_sd)
    M_full = discr.assemble_mass_matrix(ref_sd)

    one_interp = discr.interpolate(ref_sd, lambda _: np.ones(3))

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
