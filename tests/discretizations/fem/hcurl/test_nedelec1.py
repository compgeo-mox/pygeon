"""Module contains specific tests for the Nedelec 1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.Nedelec1:
    return pg.Nedelec1("test")


def test_ndof(discr, unit_sd_3d):
    assert discr.ndof(unit_sd_3d) == 2 * unit_sd_3d.num_ridges


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [2, -1],
                        [-1, 2],
                    ]
                )
                / 6
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [2, -2, -1, 0, 0, 1],
                        [-2, 4, 2, -1, -1, -1],
                        [-1, 2, 4, -2, -1, -1],
                        [0, -1, -2, 2, 1, 0],
                        [0, -1, -1, 1, 2, 0],
                        [1, -1, -1, 0, 0, 2],
                    ]
                )
                / 24
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [2, 0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 0],
                        [0, 2, 0, 1, 0, 0, -1, -1, -1, 0, 0, 1],
                        [0, 0, 2, 0, 1, 1, -1, -1, -1, 0, 0, 0],
                        [0, 1, 0, 2, 0, 0, -2, -1, -1, 0, 0, 1],
                        [0, 0, 1, 0, 2, 1, -2, -1, -1, 0, 0, 0],
                        [0, 0, 1, 0, 1, 2, -1, -2, -1, 0, 0, 0],
                        [-1, -1, -1, -2, -2, -1, 6, 3, 3, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -2, 3, 6, 3, -2, -1, -1],
                        [-1, -1, -1, -1, -1, -1, 3, 3, 6, -1, -2, -2],
                        [1, 0, 0, 0, 0, 0, -1, -2, -1, 2, 1, 0],
                        [1, 0, 0, 0, 0, 0, -1, -1, -2, 1, 2, 0],
                        [0, 1, 0, 1, 0, 0, -1, -1, -2, 0, 0, 2],
                    ]
                )
                / 120
            )

    assert np.allclose(M.todense(), M_known)


def test_assemble_lumped(discr, ref_sd):
    L = discr.assemble_lumped_matrix(ref_sd)
    match ref_sd.dim:
        case 1:
            L_known = np.array([[1, 0], [0, 1]]) / 2
        case 2:
            L_known = (
                np.array(
                    [
                        [1, -1, 0, 0, 0, 0],
                        [-1, 2, 0, 0, 0, 0],
                        [0, 0, 2, -1, 0, 0],
                        [0, 0, -1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )
                / 6
            )
        case 3:
            L_known = (
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                        [0, 0, 0, -1, -1, 0, 3, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, 0, 3, 0, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -1, -1],
                        [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
                    ]
                )
                / 24
            )

    assert np.allclose(L.todense(), L_known)


def test_lumped_consistency(discr, unit_sd_3d):
    M_lumped = discr.assemble_lumped_matrix(unit_sd_3d)
    M_full = discr.assemble_mass_matrix(unit_sd_3d)

    one_interp = discr.interpolate(unit_sd_3d, lambda _: np.ones(3))

    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


def test_interp_eval_linears(discr, unit_sd_3d):
    def q_linear(x):
        return x

    interp_q = discr.interpolate(unit_sd_3d, q_linear)
    eval_q = discr.eval_at_cell_centers(unit_sd_3d) @ interp_q
    eval_q = np.reshape(eval_q, (3, -1))

    known_q = np.array([q_linear(x) for x in unit_sd_3d.cell_centers.T]).T
    assert np.allclose(eval_q, known_q)
