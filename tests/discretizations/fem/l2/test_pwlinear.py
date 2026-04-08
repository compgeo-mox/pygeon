"""Module contains specific tests for the P1 discretization."""

import numpy as np
import pytest

import pygeon as pg
from tests.helpers import matrix_equals


@pytest.fixture
def discr() -> pg.PwLinears:
    return pg.PwLinears("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == (unit_sd.dim + 1) * unit_sd.num_cells


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
                        [2.0, 1.0, 1.0],
                        [1.0, 2.0, 1.0],
                        [1.0, 1.0, 2.0],
                    ]
                )
                / 24
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [2.0, 1.0, 1.0, 1.0],
                        [1.0, 2.0, 1.0, 1.0],
                        [1.0, 1.0, 2.0, 1.0],
                        [1.0, 1.0, 1.0, 2.0],
                    ]
                )
                / 120
            )

    assert matrix_equals(M.todense(), M_known)


def test_assemble_lumped_matrix(discr, ref_sd):
    from math import factorial

    L = discr.assemble_lumped_matrix(ref_sd)
    L_known = np.eye(discr.ndof(ref_sd)) / factorial(ref_sd.dim + 1)

    assert matrix_equals(L.todense(), L_known)


def test_interpolate(discr, unit_sd):
    interp = discr.interpolate(unit_sd, lambda x: x[0])
    P = discr.eval_at_cell_centers(unit_sd)
    known = unit_sd.cell_centers[0]

    assert np.allclose(P @ interp, known)


def test_interpolate_heaviside(discr, unit_sd_1d):
    def heaviside(x):
        return 0 if x[0] < 0.5 else 1

    true_norm_squared = 0.5
    mass = discr.assemble_mass_matrix(unit_sd_1d)

    # Test to show that nodal interpolation of a discontinuous function leads to errors.
    lagrange1 = pg.Lagrange1()
    interp_nodal = lagrange1.interpolate(unit_sd_1d, heaviside)
    interp_nodal = lagrange1.proj_to_PwPolynomials(unit_sd_1d) @ interp_nodal
    assert not np.isclose(interp_nodal @ mass @ interp_nodal, true_norm_squared)

    # Test to show that interpolation using Gauss points is more accurate in this case.
    interp_gauss = discr.interpolate(unit_sd_1d, heaviside)
    assert np.isclose(interp_gauss @ mass @ interp_gauss, true_norm_squared)


def test_proj_to_lower_PwPolynomials(discr, unit_sd):
    P0 = pg.PwConstants()

    Proj = discr.proj_to_lower_PwPolynomials(unit_sd)
    fun_P1 = discr.interpolate(unit_sd, lambda x: np.sum(x))
    fun_P0 = P0.interpolate(unit_sd, lambda x: np.sum(x))

    assert np.allclose(Proj @ fun_P1, fun_P0)
