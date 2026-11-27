"""Module contains the consistency tests of the discretizations."""

import numpy as np
import pytest
import sympy as sp

import pygeon as pg


@pytest.mark.parametrize(
    "discr_class",
    [
        pg.Lagrange1,
        pg.Lagrange2,
        pg.PwLinears,
        pg.PwQuadratics,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
        pg.RT0,
        pg.BDM1,
        pg.VecLagrange1,
        pg.VecLagrange2,
    ],
)
def test_broken_directional_derivative(discr_class, tilted_sd):
    # With u in the space and beta constant, both in the ambient coordinates,
    # beta . grad u on the grid is (P beta) . grad u, with P the projection on the
    # plane of the grid, for each component of u in the plane: a polynomial of one
    # order less, interpolated exactly by the piecewise polynomials of that order
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    discr = discr_class("test")
    k = discr.poly_order

    x, y, z = pg.exact.coordinates()
    if discr_class is pg.RT0:
        u = [0.3 + 1.5 * x, -0.7 + 1.5 * y, 1.1 + 1.5 * z]
    else:
        u = [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    beta = np.array([1.0, 2.0, 3.0])
    f = [sum(b * g for b, g in zip(P @ beta, pg.exact.gradient(u_i))) for u_i in u]
    if discr.tensor_order == pg.SCALAR:
        u_ex, f_ex = u[0], f[0]
    else:
        u_ex, f_ex = sp.Matrix(u), sp.Matrix(f)

    low = pg.get_PwPolynomials(k - 1, discr.tensor_order)("test")
    u_h = discr.interpolate(sd, pg.exact.to_callable(u_ex))
    f_h = low.interpolate(sd, pg.exact.to_callable(f_ex))
    vector_field = np.tile(beta.reshape((-1, 1)), sd.num_cells)

    dir_der = discr.assemble_broken_directional_derivative(sd, vector_field)
    assert np.allclose(dir_der @ u_h, f_h)
