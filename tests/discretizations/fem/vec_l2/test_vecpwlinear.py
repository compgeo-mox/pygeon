"""Module contains specific tests for the vector P1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.VecPwLinears:
    return pg.VecPwLinears("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_cells * unit_sd.dim * (unit_sd.dim + 1)


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [2.0, 1.0],
                        [1.0, 2.0],
                    ]
                )
                / 6
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 2.0],
                    ]
                )
                / 24
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2],
                    ]
                )
                / 120
            )

    assert np.allclose(M.todense(), M_known)


def test_assemble_lumped_matrix(discr, ref_sd):
    from math import factorial

    L = discr.assemble_lumped_matrix(ref_sd)
    L_known = np.eye(discr.ndof(ref_sd)) / factorial(ref_sd.dim + 1)

    assert np.allclose(L.todense(), L_known)


def test_proj_to_pwconstants(discr, unit_sd):
    P0 = pg.VecPwConstants()

    Proj = discr.proj_to_lower_PwPolynomials(unit_sd)
    fun_P1 = discr.interpolate(unit_sd, lambda x: x)
    fun_P0 = P0.interpolate(unit_sd, lambda x: x)

    assert np.allclose(Proj @ fun_P1, fun_P0)
