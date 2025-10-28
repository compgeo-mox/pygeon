import pytest

import numpy as np
import pygeon as pg


@pytest.fixture(
    params=[
        pg.PwConstants,
        pg.PwLinears,
        pg.PwQuadratics,
        pg.VecPwConstants,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
        pg.MatPwConstants,
        pg.MatPwLinears,
        pg.MatPwQuadratics,
    ]
)
def discr(request):
    return request.param("test")


def test_assemble_diff_matrix(discr, unit_sd_2d):
    B = discr.assemble_diff_matrix(unit_sd_2d)
    assert B.nnz == 0


def test_assemble_stiff_matrix(discr, unit_sd_2d):
    stiff = discr.assemble_stiff_matrix(unit_sd_2d)
    assert stiff.nnz == 0


def test_get_range_discr_class(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)


def test_assemble_nat_bc(discr, unit_sd_2d):
    b_faces = np.array([0, 1, 3])  # Example boundary faces
    func = lambda x: np.sin(x[0])  # Example function

    b = discr.assemble_nat_bc(unit_sd_2d, func, b_faces)

    assert np.allclose(b, 0.0)


def test_proj_to_higherPwPolynomials(discr, unit_sd):
    if discr.poly_order > 1:
        with pytest.raises(NotImplementedError):
            discr.proj_to_higher_PwPolynomials(unit_sd)
        return

    proj = discr.proj_to_higher_PwPolynomials(unit_sd)
    mass = discr.assemble_mass_matrix(unit_sd)

    discr_higher = pg.get_PwPolynomials(discr.poly_order + 1, discr.tensor_order)
    mass_higher = discr_higher("test").assemble_mass_matrix(unit_sd)

    diff = proj.T @ mass_higher @ proj - mass

    assert np.allclose(diff.data, 0.0)


def test_interpolate_and_evaluate(discr: pg.Discretization, unit_sd: pg.Grid):
    match discr.tensor_order:
        case 0:
            func = lambda x: x[0] ** discr.poly_order
        case 1:
            func = lambda x: x[: unit_sd.dim] ** discr.poly_order
        case 2:
            func = lambda x: np.tile(x[: unit_sd.dim], (unit_sd.dim, 1))

    known_vals = np.vstack([func(x).ravel() for x in unit_sd.cell_centers.T]).T

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals.ravel())


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    match discr.tensor_order:
        case 0:
            one = lambda _: 1
        case 1:
            one = lambda _: np.ones(unit_sd.dim)
        case 2:
            one = lambda _: np.ones((unit_sd.dim, unit_sd.dim))

    one_interp = discr.interpolate(unit_sd, one)
    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)
