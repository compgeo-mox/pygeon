"""Module contains general tests for Vector L2 discretizations."""

import numpy as np
import porepy as pp
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture(
    params=[
        pg.VecPwConstants,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.VecPwPolynomials:
    return request.param("test")


def test_assemble_local_dofs(discr, ref_sd):
    # Coverage test
    dofs = discr.local_dofs_of_cell(ref_sd, 0)
    known_dofs = ref_sd.num_cells * np.arange(discr.ndof_per_element(ref_sd.dim))

    assert np.all(dofs == known_dofs)


def test_dot_product(discr, tilted_sd):
    # With u a vector polynomial of the order of the space and w constant, both in the
    # ambient coordinates, w . u on the grid is w . (P u), with P the projection on
    # the plane of the grid: a polynomial of the same order, interpolated exactly
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    k = discr.poly_order

    x, y, z = pg.exact.coordinates()
    u = sp.Matrix(
        [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    )
    w = np.array([1.0, 2.0, 3.0])
    f = (sp.Matrix(w).T @ sp.Matrix(P) @ u)[0]

    scalar_discr = discr.base_discr
    u_h = discr.interpolate(sd, pg.exact.to_callable(u))
    f_h = scalar_discr.interpolate(sd, pg.exact.to_callable(f))
    vector_field = np.tile(w.reshape((-1, 1)), sd.num_cells)

    dot = discr.assemble_dot_product(sd, vector_field)
    assert np.allclose(dot @ u_h, f_h)


@pytest.mark.parametrize("method", ["assemble_mass_matrix", "assemble_lumped_matrix"])
def test_tensor_weighted_inner_product(discr, method, tilted_sd):
    # With u and v constant and K a constant tensor, not symmetric, all in the ambient
    # coordinates, (K u, v) on the grid is |Omega| (P v) . K (P u), with P the
    # projection on the plane of the grid
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    u = np.array([1.0, -2.0, 0.5])
    v = np.array([0.3, 1.5, -1.0])
    K = np.array([[2.0, 1.0, 0.0], [-0.5, 3.0, 0.7], [0.2, -1.0, 1.5]])

    # PorePy builds only symmetric tensors, so the values are set directly
    sot = pp.SecondOrderTensor(np.ones(sd.num_cells))
    sot.values = np.tile(K.reshape((3, 3, 1)), sd.num_cells)
    data = {pp.PARAMETERS: {"test": {pg.SECOND_ORDER_TENSOR: sot}}}

    u_h = discr.interpolate(sd, lambda _: u)
    v_h = discr.interpolate(sd, lambda _: v)
    M = getattr(discr, method)(sd, data)

    known = sd.cell_volumes.sum() * (P @ v) @ K @ (P @ u)
    assert np.isclose(v_h @ M @ u_h, known)
