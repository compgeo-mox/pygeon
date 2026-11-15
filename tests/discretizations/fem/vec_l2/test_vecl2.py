"""Module contains general tests for Vector L2 discretizations."""

import numpy as np
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
