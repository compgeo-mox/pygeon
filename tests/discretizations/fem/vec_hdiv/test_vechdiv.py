"""Module contains general tests for all vector H(div) discretizations."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.VecRT0,
        pg.VecBDM1,
        pg.VecRT1,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


@pytest.fixture
def data():
    params = {pg.LAME_MU: 0.5, pg.LAME_LAMBDA: 0.5, pg.LAME_MU_COSSERAT: 0.25}
    return pp.initialize_data({}, "test", params)


@pytest.fixture
def constant_fun():
    return lambda _: np.array([[1, 2, 0], [4, 3, 0], [0, 1, 1]])


def test_range(discr, ref_sd):
    if ref_sd.dim == 1:
        return
    known_range = pg.get_PwPolynomials(discr.poly_order - 1, 1)
    assert discr.get_range_discr_class(ref_sd.dim) is known_range


def test_assemble_elasticity_matrices(discr, unit_sd, data, constant_fun):
    if unit_sd.dim == 1:
        return
    M = discr.assemble_mass_matrix_elasticity(unit_sd, data)
    u = discr.interpolate(unit_sd, constant_fun)

    known = 26 if unit_sd.dim == 2 else 27
    assert np.isclose(u.T @ M @ u, known)

    if isinstance(discr, pg.VecRT0):
        return

    L = discr.assemble_lumped_matrix_elasticity(unit_sd, data)
    assert np.isclose(u.T @ L @ u, known)


def test_assemble_deviator_matrix(discr, unit_sd, data, constant_fun):
    if unit_sd.dim == 1:
        return
    M = discr.assemble_deviator_matrix(unit_sd, data)
    u = discr.interpolate(unit_sd, constant_fun)

    known = 22 if unit_sd.dim == 2 else 71 / 3
    assert np.isclose(u.T @ M @ u, known)


def test_assemble_cosserat_matrices(discr, unit_sd, data, constant_fun):
    if unit_sd.dim == 1:
        return
    M = discr.assemble_mass_matrix_cosserat(unit_sd, data)
    u = discr.interpolate(unit_sd, constant_fun)

    known = 28 if unit_sd.dim == 2 else 29.5
    assert np.isclose(u.T @ M @ u, known)

    if isinstance(discr, pg.VecRT0):
        return

    L = discr.assemble_lumped_matrix_cosserat(unit_sd, data)
    assert np.isclose(u.T @ L @ u, known)


def test_assemble_trace_matrix(discr, unit_sd, constant_fun):
    B = discr.assemble_trace_matrix(unit_sd)
    u = discr.interpolate(unit_sd, constant_fun)

    trace = B @ u
    trace_list = [0, 1, 4, 5]
    known_trace = trace_list[unit_sd.dim]

    assert np.allclose(trace, known_trace)


def test_assemble_asym_matrix(discr, unit_sd, constant_fun):
    if unit_sd.dim == 1:
        return

    u = discr.interpolate(unit_sd, constant_fun)
    asym = discr.assemble_asym_matrix(unit_sd)

    p1 = pg.get_PwPolynomials(discr.poly_order, unit_sd.dim - 2)("R")
    cell_asym_u = p1.eval_at_cell_centers(unit_sd) @ (asym @ u)

    if unit_sd.dim == 2:
        assert np.allclose(cell_asym_u, 2)
    else:
        cell_asym_u = cell_asym_u.reshape((3, -1))
        assert np.allclose(cell_asym_u[0], 1)
        assert np.allclose(cell_asym_u[1], 0)
        assert np.allclose(cell_asym_u[2], 2)


def test_trace_with_proj(discr, unit_sd):
    if unit_sd.dim == 1:
        return

    P1 = pg.get_PwPolynomials(discr.poly_order, discr.tensor_order)(discr.keyword)
    trace = P1.assemble_trace_matrix(unit_sd)

    trace_bdm = discr.assemble_trace_matrix(unit_sd)
    proj = discr.proj_to_PwPolynomials(unit_sd)

    check = trace_bdm - trace @ proj
    assert np.allclose(check.data, 0)


def test_asym_with_proj(discr, unit_sd):
    if unit_sd.dim == 1:
        return

    P1 = pg.get_PwPolynomials(discr.poly_order, discr.tensor_order)(discr.keyword)
    asym = P1.assemble_asym_matrix(unit_sd)

    asym_bdm = discr.assemble_asym_matrix(unit_sd)
    proj = discr.proj_to_PwPolynomials(unit_sd)

    check = asym_bdm - asym @ proj
    assert np.allclose(check.data, 0)


def test_interp_eval_linears(discr, unit_sd):
    def linear(x):
        return np.array([x, 2 * x, -x])

    interp = discr.interpolate(unit_sd, linear)
    eval = discr.eval_at_cell_centers(unit_sd) @ interp
    eval = np.reshape(eval, (unit_sd.dim * 3, unit_sd.num_cells))

    known = np.array(
        [linear(x)[: unit_sd.dim, :].ravel() for x in unit_sd.cell_centers.T]
    ).T
    assert np.allclose(eval, known)


def test_linear_asym(discr, unit_sd):
    if unit_sd.dim == 1:
        return

    asym = discr.assemble_asym_matrix(unit_sd)

    func = lambda x: np.array(
        [
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
        ]
    )
    if unit_sd.dim == 3:
        func_asym = lambda x: np.array(
            [
                x[1] - x[2],
                x[2] - x[0],
                x[0] - x[1],
            ]
        )
    else:
        func_asym = lambda x: x[0] - x[1]

    func_interp = discr.interpolate(unit_sd, func)

    asym_space = pg.get_PwPolynomials(discr.poly_order, unit_sd.dim - 2)(discr.keyword)
    asym_interp = asym_space.interpolate(unit_sd, func_asym)

    assert np.allclose(asym @ func_interp, asym_interp)


def test_cosserat_1d(discr, unit_sd_1d):
    with pytest.raises(ValueError):
        discr.assemble_mass_matrix_cosserat(unit_sd_1d, None)

    with pytest.raises(ValueError):
        discr.assemble_lumped_matrix_cosserat(unit_sd_1d, None)
