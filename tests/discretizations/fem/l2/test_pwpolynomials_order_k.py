"""Module contains tests of the generic methods of pg.PwPolynomials for any order,
through classes of the orders above 2 declared here."""

import numpy as np
import pytest
import sympy as sp

import pygeon as pg


class PwCubics(pg.PwPolynomials):
    poly_order = 3


class PwQuartics(pg.PwPolynomials):
    poly_order = 4


class PwQuintics(pg.PwPolynomials):
    poly_order = 5


classes = {
    1: pg.PwLinears,
    2: pg.PwQuadratics,
    3: PwCubics,
    4: PwQuartics,
    5: PwQuintics,
}


def polynomial(degree):
    """A polynomial of the given total degree in the coordinates."""
    x, y, z = pg.exact.coordinates()
    a = degree // 2
    return x**a * (1 + y) ** (degree - a) + z


@pytest.fixture(params=[2, 3, 4])
def order(request):
    return request.param


def test_ndof(unit_sd, order):
    discr = classes[order]()
    n_local = np.prod(np.arange(order + 1, order + unit_sd.dim + 1)) // np.prod(
        np.arange(1, unit_sd.dim + 1)
    )
    assert discr.ndof(unit_sd) == n_local * unit_sd.num_cells


def test_mass_is_exact(unit_sd, order):
    # The integral of a polynomial of the order of the space, with the mass matrix
    discr = classes[order]()
    fun = polynomial(order)
    fun_h = discr.interpolate(unit_sd, pg.exact.to_callable(fun))
    one_h = discr.interpolate(unit_sd, lambda x: np.ones_like(x[0]))

    integral = one_h @ discr.assemble_mass_matrix(unit_sd) @ fun_h

    # The exact integral on the unit cube of dimension dim, with the other
    # coordinates equal to zero
    coords = pg.exact.coordinates()
    fun = fun.subs({c: 0 for c in coords[unit_sd.dim :]})
    known = sp.integrate(fun, *[(c, 0, 1) for c in coords[: unit_sd.dim]])
    assert np.isclose(integral, float(known))


def test_broken_grad_is_exact(tilted_sd, order):
    # The gradient is in the ambient coordinates, and its components on the grid are
    # the ones in the plane, R grad f, with R the rotation of the grid
    sd = tilted_sd
    discr, low = classes[order](), classes[order - 1]()
    fun = polynomial(order)
    fun_h = discr.interpolate(sd, pg.exact.to_callable(fun))
    grad_h = discr.assemble_broken_grad_matrix(sd) @ fun_h

    grad = sp.Matrix(sd.rotation_matrix) * pg.exact.gradient(fun)
    known = [low.interpolate(sd, pg.exact.to_callable(grad_a)) for grad_a in grad]
    assert np.allclose(grad_h, np.concatenate(known))


def test_projections(tilted_sd, order):
    sd = tilted_sd
    discr, high = classes[order](), classes[order + 1]()
    fun = pg.exact.to_callable(polynomial(order))
    fun_h = discr.interpolate(sd, fun)

    # The inclusion is exact, and the L2 projection back is the identity
    up = discr.proj_to_higher_PwPolynomials(sd)
    down = high.proj_to_lower_PwPolynomials(sd)
    assert np.allclose(up @ fun_h, high.interpolate(sd, fun))
    assert np.allclose(down @ up @ fun_h, fun_h)

    # The inclusion preserves the mass
    mass, mass_high = (
        discr.assemble_mass_matrix(sd),
        high.assemble_mass_matrix(sd),
    )
    assert np.allclose((up.T @ mass_high @ up - mass).data, 0)


def test_eval_at_cell_centers(tilted_sd, order):
    sd = tilted_sd
    discr = classes[order]()
    fun = pg.exact.to_callable(polynomial(order))
    fun_h = discr.interpolate(sd, fun)

    values = discr.eval_at_cell_centers(sd) @ fun_h
    assert np.allclose(values, fun(sd.cell_centers))


def test_no_lumped_mass(unit_sd):
    with pytest.raises(NotImplementedError):
        PwCubics().assemble_lumped_matrix(unit_sd)
