"""Module contains specific tests for the vector Lagrangean L1 discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecLagrange1("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.dim * unit_sd.num_nodes


def test_div_matrix(discr, ref_sd):
    D = discr.assemble_div_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            D_known = np.array([[1.0, -1.0]])
        case 2:
            D_known = np.array([[-0.5, 0.5, 0.0, -0.5, 0.0, 0.5]])
        case 3:
            D_known = np.array([[-1, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1]]) / 6

    assert np.allclose(D.todense(), D_known)


def test_symgrad_matrix(discr, ref_sd):
    D = discr.assemble_symgrad_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            D_known = np.array([[1.0, -1.0]])
        case 2:
            D_known = (
                np.array(
                    [
                        [-2, 2, 0, 0, 0, 0],
                        [-1, 0, 1, -1, 1, 0],
                        [-1, 0, 1, -1, 1, 0],
                        [0, 0, 0, -2, 0, 2],
                    ]
                )
                / 4
            )
        case 3:
            D_known = (
                np.array(
                    [
                        [-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 1, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0],
                        [-1, 0, 1, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 1, -1, 0, 1, 0],
                        [-1, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 1, -1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 2],
                    ]
                )
                / 12
            )

    assert np.allclose(D.todense(), D_known)


def test_compute_stress(discr, unit_sd):
    u_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.5 + x[1]])
    u = discr.interpolate(unit_sd, u_fun)
    data = {pp.PARAMETERS: {discr.keyword: {"lambda": 1, "mu": 0.5}}}

    sigma = discr.compute_stress(unit_sd, u, data)
    assert np.allclose(sigma, 0)


def test_0d(discr):
    sd = pp.PointGrid([1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    B = discr.assemble_div_matrix(sd).todense()
    S = discr.assemble_symgrad_matrix(sd).todense()

    S_known = np.zeros((1, 1))

    assert np.allclose(B, S_known)
    assert np.allclose(S, S_known)


def test_div_and_symgrad_for_rotations(discr, unit_sd):
    fun = lambda x: np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    fun_interp = discr.interpolate(unit_sd, fun)

    A = discr.assemble_div_div_matrix(unit_sd)
    S = discr.assemble_symgrad_symgrad_matrix(unit_sd)

    assert np.allclose(A @ fun_interp, 0)
    assert np.allclose(S @ fun_interp, 0)


def test_range_disc(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)
