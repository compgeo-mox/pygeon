"""Module contains specific tests for the vector virtual Lagrangean L1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecVLagrange1("test")


def test_ndof(discr, pentagon_sd):
    assert discr.ndof(pentagon_sd) == 10


def test_compliance_lagrange1_triangles(discr, unit_sd_2d):
    lag1 = pg.VecLagrange1(discr.keyword)

    lag_mass = lag1.assemble_mass_matrix(unit_sd_2d)
    vlag_mass = discr.assemble_mass_matrix(unit_sd_2d)

    lag_diff = lag1.assemble_diff_matrix(unit_sd_2d)
    vlag_diff = discr.assemble_diff_matrix(unit_sd_2d)

    assert np.allclose((lag_mass - vlag_mass).data, 0)
    assert np.allclose((lag_diff - vlag_diff).data, 0)


def test_zero_penalization_on_triangles(discr, unit_sd_2d):
    P = discr.assemble_penalisation_matrix(unit_sd_2d)
    assert np.allclose(P.data, 0)


def setup(sd):
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    discr = pg.VecVLagrange1()
    M = discr.assemble_mass_matrix(sd)

    div = discr.assemble_div_matrix(sd)
    symgrad = discr.assemble_symgrad_matrix(sd)

    div_div = discr.assemble_div_div_matrix(sd)
    symgrad_symgrad = discr.assemble_symgrad_symgrad_matrix(sd)

    pen = discr.assemble_penalisation_matrix(sd)
    pen.data[abs(pen.data) < 1e-10] = 0

    diff = discr.assemble_diff_matrix(sd)

    stiff = discr.assemble_stiff_matrix(sd)
    stiff.data[abs(stiff.data) < 1e-10] = 0

    return M, div, symgrad, div_div, symgrad_symgrad, pen, diff, stiff, discr


def test_assemble_mass(discr, ref_square):
    M = discr.assemble_mass_matrix(ref_square)

    M_known = (
        np.array(
            [
                [119, -63, -63, 91, 0, 0, 0, 0],
                [-63, 119, 91, -63, 0, 0, 0, 0],
                [-63, 91, 119, -63, 0, 0, 0, 0],
                [91, -63, -63, 119, 0, 0, 0, 0],
                [0, 0, 0, 0, 119, -63, -63, 91],
                [0, 0, 0, 0, -63, 119, 91, -63],
                [0, 0, 0, 0, -63, 91, 119, -63],
                [0, 0, 0, 0, 91, -63, -63, 119],
            ]
        )
        / 336
    )

    assert np.allclose(M.todense(), M_known)


def test_assemble_div(discr, ref_square):
    div = discr.assemble_div_matrix(ref_square)
    div_known = np.array([[-1, 1, -1, 1, -1, -1, 1, 1]]) / 2

    assert np.allclose(div.todense(), div_known)


def test_assemble_symgrad(discr, ref_square):
    symgrad = discr.assemble_symgrad_matrix(ref_square)
    symgrad_known = (
        np.array(
            [
                [-2, 2, -2, 2, 0, 0, 0, 0],
                [-1, -1, 1, 1, -1, 1, -1, 1],
                [-1, -1, 1, 1, -1, 1, -1, 1],
                [0, 0, 0, 0, -2, -2, 2, 2],
            ]
        )
        / 4
    )

    assert np.allclose(symgrad.todense(), symgrad_known)


def test_assemble_divdiv(discr, ref_square):
    divdiv = discr.assemble_div_div_matrix(ref_square)
    divdiv_known = (
        np.array(
            [
                [1, -1, 1, -1, 1, 1, -1, -1],
                [-1, 1, -1, 1, -1, -1, 1, 1],
                [1, -1, 1, -1, 1, 1, -1, -1],
                [-1, 1, -1, 1, -1, -1, 1, 1],
                [1, -1, 1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, 1, -1, -1],
                [-1, 1, -1, 1, -1, -1, 1, 1],
                [-1, 1, -1, 1, -1, -1, 1, 1],
            ]
        )
        / 4
    )

    assert np.allclose(divdiv.todense(), divdiv_known)


def test_assemble_symgradsymgrad(discr, ref_square):
    symgradsymgrad = discr.assemble_symgrad_symgrad_matrix(ref_square)
    symgradsymgrad_known = (
        np.array(
            [
                [3, -1, 1, -3, 1, -1, 1, -1],
                [-1, 3, -3, 1, 1, -1, 1, -1],
                [1, -3, 3, -1, -1, 1, -1, 1],
                [-3, 1, -1, 3, -1, 1, -1, 1],
                [1, 1, -1, -1, 3, 1, -1, -3],
                [-1, -1, 1, 1, 1, 3, -3, -1],
                [1, 1, -1, -1, -1, -3, 3, 1],
                [-1, -1, 1, 1, -3, -1, 1, 3],
            ]
        )
        / 4
    )

    assert np.allclose(symgradsymgrad.todense(), symgradsymgrad_known)


def test_interp_and_eval(discr, ref_octagon):
    func = lambda x: x
    interp_func = discr.interpolate(ref_octagon, func)
    eval_interp = discr.eval_at_cell_centers(ref_octagon) @ interp_func

    known_vals = np.vstack([func(c)[:2] for c in ref_octagon.cell_centers.T]).T.ravel()

    assert np.allclose(eval_interp, known_vals)


def test_3d_failure(discr, unit_sd_3d):
    with pytest.raises(ValueError):
        discr.assemble_symgrad_matrix(unit_sd_3d)

    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(3)
