import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecBDM1("test")


@pytest.fixture
def data():
    return {pp.PARAMETERS: {"test": {"mu": 0.5, "lambda": 0.5, "mu_c": 0.25}}}


@pytest.fixture
def fun():
    return lambda _: np.array([[1, 2, 0], [4, 3, 0], [0, 1, 1]])


def test_asym_1d(discr, unit_sd_1d):
    with pytest.raises(ValueError):
        discr.assemble_asym_matrix(unit_sd_1d)


def test_trace_2d(discr, unit_sd_2d):
    B = discr.assemble_trace_matrix(unit_sd_2d)

    fun = lambda x: np.array([[x[0] + x[1], x[0], 0], [x[1], -x[0] - x[1], 0]])
    u = discr.interpolate(unit_sd_2d, fun)

    trace = B @ u

    assert np.allclose(trace, 0)


def test_ndof_2d(discr, unit_sd_2d):
    assert discr.ndof(unit_sd_2d) == 180


def test_ndof_3d(discr, unit_sd_3d):
    assert discr.ndof(unit_sd_3d) == 2178


def test_assemble_mass_matrices(discr, unit_sd, data, fun):
    if unit_sd.dim == 1:
        return
    M = discr.assemble_mass_matrix(unit_sd, data)
    L = discr.assemble_lumped_matrix(unit_sd, data)
    u = discr.interpolate(unit_sd, fun)

    known = 26 if unit_sd.dim == 2 else 27

    assert np.isclose(u.T @ M @ u, known)
    assert np.isclose(u.T @ L @ u, known)


def test_assemble_cosserat_matrices(discr, unit_sd, data, fun):
    if unit_sd.dim == 1:
        return
    M = discr.assemble_mass_matrix_cosserat(unit_sd, data)
    L = discr.assemble_lumped_matrix_cosserat(unit_sd, data)
    u = discr.interpolate(unit_sd, fun)

    known = 28 if unit_sd.dim == 2 else 29.5

    assert np.isclose(u.T @ M @ u, known)
    assert np.isclose(u.T @ L @ u, known)


def test_range(discr):
    assert discr.get_range_discr_class(2) is pg.VecPwConstants


def test_trace(discr, unit_sd, fun):
    B = discr.assemble_trace_matrix(unit_sd)
    u = discr.interpolate(unit_sd, fun)

    trace = B @ u
    known_trace = [0, 1, 4, 5]
    known = known_trace[unit_sd.dim]

    assert np.allclose(trace, known)


def test_assemble_asym_matrix(discr, unit_sd, fun):
    if unit_sd.dim == 1:
        return

    u = discr.interpolate(unit_sd, fun)

    asym = discr.assemble_asym_matrix(unit_sd, False)

    p1 = pg.PwLinears("p1") if unit_sd.dim == 2 else pg.VecPwLinears("p1")
    cell_asym_u = p1.eval_at_cell_centers(unit_sd) @ (asym @ u)

    if unit_sd.dim == 2:
        assert np.allclose(cell_asym_u, 2)
    else:
        cell_asym_u = cell_asym_u.reshape((3, -1))
        assert np.allclose(cell_asym_u[0], 1)
        assert np.allclose(cell_asym_u[1], 0)
        assert np.allclose(cell_asym_u[2], 2)


def test_proj_to_and_from_rt0(discr, unit_sd):
    def linear(x):
        return np.array([x, 2 * x, 3 * x])

    interp = discr.interpolate(unit_sd, linear)
    interp_to_rt0 = discr.proj_to_RT0(unit_sd) @ interp
    interp_from_rt0 = discr.proj_from_RT0(unit_sd) @ interp_to_rt0

    assert np.allclose(interp, interp_from_rt0)


def test_trace_with_proj(discr, unit_sd):
    if unit_sd.dim == 1:
        return

    P1 = pg.MatPwLinears()
    trace = P1.assemble_trace_matrix(unit_sd)

    trace_bdm = discr.assemble_trace_matrix(unit_sd)
    proj = discr.proj_to_PwPolynomials(unit_sd)

    check = trace_bdm - trace @ proj
    assert np.allclose(check.data, 0)


def test_asym_with_proj(discr, unit_sd):
    if unit_sd.dim == 1:
        return

    P1 = pg.MatPwLinears()
    asym = P1.assemble_asym_matrix(unit_sd)

    asym_bdm = discr.assemble_asym_matrix(unit_sd)
    proj = discr.proj_to_PwPolynomials(unit_sd)

    check = asym_bdm - asym @ proj
    assert np.allclose(check.data, 0)
