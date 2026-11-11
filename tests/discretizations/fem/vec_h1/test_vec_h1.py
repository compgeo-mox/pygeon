"""Module contains general tests for all vector H1 discretizations."""

import numpy as np
import porepy as pp
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture(params=[pg.VecLagrange1, pg.VecLagrange2])
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_assemble_adv_matrix_is_exact(discr, tilted_sd):
    # With u a vector polynomial of the order of the space and a constant field, both
    # in the ambient coordinates, beta . grad u on the grid is w = grad(u) P beta, with
    # P the projection on the plane of the grid: a polynomial of order k - 1, which
    # the space interpolates exactly, so that A u = M w
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    x, y, z = pg.exact.coordinates()
    k = discr.poly_order
    u = sp.Matrix(
        [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    )
    beta = np.array([1.0, 2.0, 3.0])
    w = pg.exact.vector_gradient(u) @ sp.Matrix(P @ beta)

    u_h = discr.interpolate(sd, pg.exact.to_callable(u))
    w_h = discr.interpolate(sd, pg.exact.to_callable(w))
    field = np.tile(beta.reshape((-1, 1)), sd.num_cells)
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: field})

    A = discr.assemble_adv_matrix(sd, data)
    M = discr.assemble_mass_matrix(sd)
    assert np.allclose(A @ u_h, M @ w_h)
