"""Module contains specific tests for the vector Lagrangean L2 discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecLagrange2("test")


def test_ndof(discr, unit_sd):
    scalar_discr = pg.Lagrange2(discr.keyword)
    assert discr.ndof(unit_sd) == unit_sd.dim * scalar_discr.ndof(unit_sd)


def test_base_class(discr):
    assert isinstance(discr, pg.VecLagrange)
    assert isinstance(pg.VecLagrange1(), pg.VecLagrange)


def test_div_and_symgrad_of_quadratics(discr, unit_sd):
    # The divergence and the symmetric gradient of a quadratic field are linear, so
    # they are represented exactly in the piecewise linears
    fun = lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1], x[2] ** 2 - x[0]])
    div = lambda x: 2 * x[0] + (unit_sd.dim > 1) * x[0] + (unit_sd.dim > 2) * 2 * x[2]

    def symgrad(x):
        grad = np.array(
            [
                [2 * x[0], np.ones_like(x[0]), np.zeros_like(x[0])],
                [x[1], x[0], np.zeros_like(x[0])],
                [-np.ones_like(x[0]), np.zeros_like(x[0]), 2 * x[2]],
            ]
        )
        return 0.5 * (grad + grad.transpose(1, 0, *range(2, grad.ndim)))

    fun_interp = discr.interpolate(unit_sd, fun)

    p1 = pg.PwLinears(discr.keyword)
    mat_p1 = pg.MatPwLinears(discr.keyword)

    D = discr.assemble_div_matrix(unit_sd)
    S = discr.assemble_symgrad_matrix(unit_sd)

    assert np.allclose(D @ fun_interp, p1.interpolate(unit_sd, div))
    assert np.allclose(S @ fun_interp, mat_p1.interpolate(unit_sd, symgrad))


def test_compute_stress(discr, unit_sd):
    u_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.5 + x[1]])
    u = discr.interpolate(unit_sd, u_fun)
    data = pp.initialize_data({}, discr.keyword, {pg.LAME_LAMBDA: 1, pg.LAME_MU: 0.5})

    sigma = discr.compute_stress(unit_sd, u, data)
    assert np.allclose(sigma, 0)


def test_div_and_symgrad_for_rotations(discr, unit_sd):
    fun = lambda x: np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    fun_interp = discr.interpolate(unit_sd, fun)

    A = discr.assemble_div_matrix(unit_sd)
    S = discr.assemble_symgrad_matrix(unit_sd)

    assert np.allclose(A @ fun_interp, 0)
    assert np.allclose(S @ fun_interp, 0)


def test_point_grid(discr, ref_sd_0d):
    with pytest.raises(ValueError):
        discr.assemble_symgrad_matrix(ref_sd_0d)


def test_range_disc(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)
