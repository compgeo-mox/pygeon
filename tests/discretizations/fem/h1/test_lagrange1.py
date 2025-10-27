"""Module contains a unit test for the Lagrangean P1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.Lagrange1("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_nodes


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [2, 1],
                        [1, 2],
                    ]
                )
                / 6
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [2, 1, 1],
                        [1, 2, 1],
                        [1, 1, 2],
                    ]
                )
                / 24
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [2, 1, 1, 1],
                        [1, 2, 1, 1],
                        [1, 1, 2, 1],
                        [1, 1, 1, 2],
                    ]
                )
                / 120
            )

    assert np.allclose(M.todense(), M_known)


def test_assemble_diff_matrix(discr, ref_sd):
    M = discr.assemble_diff_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = np.array(
                [
                    [-1, 1],
                ]
            )
        case 2:
            M_known = np.array(
                [
                    [0, -1, 1],
                    [-1, 0, 1],
                    [-1, 1, 0],
                ]
            )
        case 3:
            M_known = np.array(
                [
                    [-1, 1, 0, 0],
                    [-1, 0, 1, 0],
                    [-1, 0, 0, 1],
                    [0, -1, 1, 0],
                    [0, -1, 0, 1],
                    [0, 0, -1, 1],
                ]
            )

    assert np.allclose(M.todense(), M_known)


def test_assemble_stiff_matrix(discr, ref_sd):
    M = discr.assemble_stiff_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = np.array(
                [
                    [1, -1],
                    [-1, 1],
                ]
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [2, -1, -1],
                        [-1, 1, 0],
                        [-1, 0, 1],
                    ]
                )
                / 2
            )

        case 3:
            M_known = (
                np.array(
                    [
                        [3, -1, -1, -1],
                        [-1, 1, 0, 0],
                        [-1, 0, 1, 0],
                        [-1, 0, 0, 1],
                    ]
                )
                / 6
            )

    assert np.allclose(M.todense(), M_known)


def test_range_discr(discr):
    assert discr.get_range_discr_class(1) is pg.PwConstants
    assert discr.get_range_discr_class(2) is pg.RT0
    assert discr.get_range_discr_class(3) is pg.Nedelec0


def test_assemble_lumped_matrix(discr, ref_sd):
    import scipy.special

    L = discr.assemble_lumped_matrix(ref_sd)
    L_known = np.eye(ref_sd.dim + 1) / scipy.special.factorial(ref_sd.dim + 1)

    assert np.allclose(L.todense(), L_known)
