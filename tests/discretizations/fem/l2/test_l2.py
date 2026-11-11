"""Module contains general tests for all L2 discretizations."""

import numpy as np
import pytest
import sympy as sp

import pygeon as pg


@pytest.fixture(
    params=[
        pg.PwConstants,
        pg.PwLinears,
        pg.PwQuadratics,
        pg.VecPwConstants,
        pg.VecPwLinears,
        pg.VecPwQuadratics,
        pg.MatPwConstants,
        pg.MatPwLinears,
        pg.MatPwQuadratics,
        pg.SymMatPwConstants,
        pg.SymMatPwLinears,
        pg.SymMatPwQuadratics,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_assemble_diff_matrix(discr, unit_sd_2d):
    B = discr.assemble_diff_matrix(unit_sd_2d)
    assert B.nnz == 0


def test_assemble_stiff_matrix(discr, unit_sd_2d):
    stiff = discr.assemble_stiff_matrix(unit_sd_2d)
    assert stiff.nnz == 0


def test_get_range_discr_class(discr):
    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(2)


def test_assemble_nat_bc(discr, unit_sd_2d):
    b_faces = np.array([0, 1, 3])  # Example boundary faces
    func = lambda x: np.sin(x[0])  # Example function

    b = discr.assemble_nat_bc(unit_sd_2d, func, b_faces)

    assert np.allclose(b, 0.0)


def test_proj_to_higherPwPolynomials(discr, unit_sd):
    if discr.poly_order > 1:
        with pytest.raises(NotImplementedError):
            discr.proj_to_higher_PwPolynomials(unit_sd)
        return

    proj = discr.proj_to_higher_PwPolynomials(unit_sd)
    mass = discr.assemble_mass_matrix(unit_sd)

    discr_higher = pg.get_PwPolynomials(discr.poly_order + 1, discr.tensor_order)
    mass_higher = discr_higher("test").assemble_mass_matrix(unit_sd)

    diff = proj.T @ mass_higher @ proj - mass

    assert np.allclose(diff.data, 0.0)


def test_interpolate_and_evaluate(discr: pg.Discretization, unit_sd: pg.Grid):
    match discr.tensor_order:
        case 0:
            func = lambda x: x[0] ** discr.poly_order
        case 1:

            def func(x):
                ans = x.copy()
                ans[unit_sd.dim :] = 0
                return ans
        case 2:

            def func(x):
                d = unit_sd.dim
                ans = np.tile(x, [3, 1, 1][: x.ndim + 1])
                ans[d:] = 0
                ans[:, d:] = 0

                # Make it symmetric for the SymMat case
                if "Sym" in type(discr).__name__:
                    ans += np.transpose(ans, [1, 0, 2][: x.ndim + 1])

                return ans

    known_vals = np.vstack([func(x).ravel() for x in unit_sd.cell_centers.T]).T

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals.ravel())


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    match discr.tensor_order:
        case 0:
            one = lambda _: 1
        case 1:
            one = lambda _: np.ones(pg.AMBIENT_DIM)
        case 2:
            one = lambda _: np.ones((pg.AMBIENT_DIM, pg.AMBIENT_DIM))

    one_interp = discr.interpolate(unit_sd, one)
    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


@pytest.mark.parametrize(
    "vec_class", [pg.VecPwConstants, pg.VecPwLinears, pg.VecPwQuadratics]
)
def test_dot_product(vec_class, tilted_sd):
    # With u a vector polynomial of the order of the space and w constant, both in the
    # ambient coordinates, w . u on the grid is w . (P u), with P the projection on
    # the plane of the grid: a polynomial of the same order, interpolated exactly
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    vec_discr = vec_class("test")
    k = vec_discr.poly_order

    x, y, z = pg.exact.coordinates()
    u = sp.Matrix(
        [(1 + x + 2 * y + z) ** k, (2 - x + y - z) ** k, (x - y + 2 * z) ** k]
    )
    w = np.array([1.0, 2.0, 3.0])
    f = (sp.Matrix(w).T @ sp.Matrix(P) @ u)[0]

    scalar_discr = vec_discr.base_discr
    u_h = vec_discr.interpolate(sd, pg.exact.to_callable(u))
    f_h = scalar_discr.interpolate(sd, pg.exact.to_callable(f))
    field = np.tile(w.reshape((-1, 1)), sd.num_cells)

    dot = vec_discr.assemble_dot_product(sd, field)
    assert np.allclose(dot @ u_h, f_h)


@pytest.mark.parametrize(
    "mat_class", [pg.MatPwConstants, pg.MatPwLinears, pg.MatPwQuadratics]
)
def test_matrix_dot_product(mat_class, tilted_sd):
    # With tau a matrix polynomial of the order of the space and w constant, both in
    # the ambient coordinates, tau w on the grid is (P tau P) w, with P the projection
    # on the plane of the grid: a vector polynomial of the same order, interpolated
    # exactly
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix
    mat_discr = mat_class("test")
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
