import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
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

    assert np.allclose(M.todense(), M_known)


def test_assemble_lumped_matrix(discr, ref_sd):
    from math import factorial

    L = discr.assemble_lumped_matrix(ref_sd)
    L_known = np.eye(discr.ndof(ref_sd)) / factorial(ref_sd.dim + 1)

    assert np.allclose(L.todense(), L_known)


def test_interpolate(discr, unit_sd_2d):
    interp = discr.interpolate(unit_sd_2d, lambda x: x[0])
    P = discr.eval_at_cell_centers(unit_sd_2d)
    known = unit_sd_2d.cell_centers[0]

    assert np.allclose(P @ interp, known)


def test_proj_to_lower_PwPolynomials(discr, unit_sd):
    P0 = pg.PwConstants()

    Proj = discr.proj_to_lower_PwPolynomials(unit_sd)
    fun_P1 = discr.interpolate(unit_sd, lambda x: np.sum(x))
    fun_P0 = P0.interpolate(unit_sd, lambda x: np.sum(x))

    assert np.allclose(Proj @ fun_P1, fun_P0)
