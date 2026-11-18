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


def test_assemble_nat_bc_is_physical(discr, tilted_sd):
    # With g and c constant, both in the ambient coordinates, (v, g) on the boundary
    # for v the interpolant of c is (P c) . g times the measure of the boundary, with
    # P the projection on the plane of the grid
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    b_faces = sd.tags["domain_boundary_faces"]

    g = np.array([1.0, 2.0, 3.0])
    c = np.array([0.5, -1.0, 2.0])

    nat_bc = discr.assemble_nat_bc(sd, lambda _: g, b_faces)
    c_h = discr.interpolate(sd, lambda _: c)
    assert np.isclose(nat_bc @ c_h, (P @ c) @ g * sd.face_areas[b_faces].sum())


def test_compute_stress_is_physical(discr, tilted_sd):
    # With u = a + A x in the ambient coordinates, only the derivatives along the
    # plane of the grid act, so that the stress in the ambient coordinates is
    # 2 mu P sym(A) P + lambda Tr(P A) P, with P the projection on the plane
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    a = np.array([0.3, -0.7, 1.1])
    A = np.array([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [3.0, 1.0, 1.0]])
    mu, lambda_ = 0.7, 1.3

    u = discr.interpolate(sd, lambda x: a.reshape((-1, 1)) + A @ x)
    data = pp.initialize_data({}, "test", {pg.LAME_MU: mu, pg.LAME_LAMBDA: lambda_})
    sigma = discr.compute_stress(sd, u, data)

    known = mu * P @ (A + A.T) @ P + lambda_ * np.trace(P @ A) * P
    assert np.allclose(sigma, known.reshape((3, 3, 1)))
