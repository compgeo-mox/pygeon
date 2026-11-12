"""Module contains general tests for matrix L2 discretizations."""

import numpy as np
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture(params=[pg.MatPwConstants, pg.MatPwLinears, pg.MatPwQuadratics])
def mat_discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_assemble_mult_matrix_requires_direction(mat_discr, ref_sd):
    with pytest.raises(ValueError):
        mat_discr.assemble_mult_matrix(ref_sd, np.empty(0))


def test_assemble_upper_convected_distortion(mat_discr, ref_sd):
    dim = ref_sd.dim

    match dim:
        case 1:
            grad_block = np.array([[2.0]])
        case 2:
            grad_block = np.array([[1.0, 2.0], [-1.0, 3.0]])
        case 3:
            grad_block = np.array([[1.0, 2.0, -1.0], [0.0, 3.0, 4.0], [5.0, -2.0, 1.0]])
        case _:
            raise ValueError("The dimension must be 1, 2 or 3.")

    grad_full = np.zeros((pg.AMBIENT_DIM, pg.AMBIENT_DIM))
    grad_full[:dim, :dim] = grad_block

    def grad_func(_):
        return grad_full

    def mat_func(x):
        mat = np.zeros((pg.AMBIENT_DIM, pg.AMBIENT_DIM, x.shape[-1]))
        mat[0, 0] = 1 + 2 * x[0]

        if dim >= 2:
            mat[0, 1] = x[0] - x[1]
            mat[1, 0] = 3 * x[1]
            mat[1, 1] = 2 - x[0]

        if dim == 3:
            mat[0, 2] = x[2]
            mat[1, 2] = x[0] + x[2]
            mat[2, 0] = x[1] - x[2]
            mat[2, 1] = x[0] + 2 * x[2]
            mat[2, 2] = 1 + x[2]

        return mat

    def distortion_func(x):
        mat = mat_func(x)[:dim, :dim]  # (d, d, n)

        mat_shift = np.moveaxis(mat, -1, 0)
        distortion = -(grad_block @ mat_shift + mat_shift @ grad_block.T)
        distortion = np.moveaxis(distortion, 0, -1)

        val = np.zeros((pg.AMBIENT_DIM, pg.AMBIENT_DIM, x.shape[-1]))
        val[:dim, :dim] = distortion
        return val

    grad_v = pg.MatPwConstants(mat_discr.keyword).interpolate(ref_sd, grad_func)
    mat_dof = mat_discr.interpolate(ref_sd, mat_func)

    distortion = mat_discr.assemble_upper_convected_distortion(ref_sd, grad_v)
    known = mat_discr.interpolate(ref_sd, distortion_func)

    assert np.allclose(distortion @ mat_dof, known)


def test_dot_product(mat_discr, tilted_sd):
    # With tau a matrix polynomial of the order of the space and w constant, both in
    # the ambient coordinates, tau w on the grid is (P tau P) w, with P the projection
    # on the plane of the grid: a vector polynomial of the same order, interpolated
    # exactly
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    k = mat_discr.poly_order

    x, y, z = pg.exact.coordinates()
    rows = [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    tau = sp.Matrix([[(i + 1) * r + j for j in range(3)] for i, r in enumerate(rows)])
    w = np.array([1.0, 2.0, 3.0])
    f = tau @ sp.Matrix(P @ w)

    vec_discr = mat_discr.base_discr
    tau_h = mat_discr.interpolate(sd, pg.exact.to_callable(tau))
    f_h = vec_discr.interpolate(sd, pg.exact.to_callable(f))
    vector_field = np.tile(w.reshape((-1, 1)), sd.num_cells)

    dot = mat_discr.assemble_dot_product(sd, vector_field)
    assert np.allclose(dot @ tau_h, f_h)
