"""Module contains general tests for all H1 discretizations."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg
from tests.helpers import constant_tensor


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_interpolate_and_evaluate(discr, unit_sd):
    func = lambda x: x[0] ** discr.poly_order
    known_vals = func(unit_sd.cell_centers)

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals)


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    one_interp = discr.interpolate(unit_sd, lambda _: 1)

    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


def test_stiffness_consistency(discr, unit_sd):
    """Compare the implemented stiffness matrix
    to the one obtained by mapping to the range discretization"""

    Stiff_1 = discr.assemble_stiff_matrix(unit_sd)
    Stiff_2 = discr.assemble_grad_grad_matrix(unit_sd)

    diff = Stiff_1 - Stiff_2
    assert np.allclose(diff.data, 0)


def test_differential_consistency(discr, unit_sd):
    pi_grad = discr.assemble_diff_matrix(unit_sd)
    range_dis = discr.get_range_discr_class(unit_sd.dim)()
    pi_grad = range_dis.proj_to_PwPolynomials(unit_sd) @ pi_grad

    grad_pi = discr.assemble_broken_grad_matrix(unit_sd)

    if isinstance(discr, pg.Lagrange1) and unit_sd.dim > 1:
        proj = pg.VecPwConstants().proj_to_higher_PwPolynomials(unit_sd)
        grad_pi = proj @ grad_pi

    if unit_sd.dim == 2:
        # In 2D, the differential is a rotated gradient. So the matrices don't
        # coincide, but their squares do.
        diff = pi_grad.T @ pi_grad - grad_pi.T @ grad_pi
    else:
        diff = pi_grad - grad_pi

    assert np.allclose(diff.data, 0)


def test_stiffness_of_linear(discr, ref_sd):
    stiff = discr.assemble_stiff_matrix(ref_sd)

    sol_func = lambda x: x[0]
    interp = discr.interpolate(ref_sd, sol_func)

    assert np.isclose(interp @ stiff @ interp, ref_sd.cell_volumes[0])


def test_point_grid(discr, ref_sd_0d):
    """Tests with a point grid"""

    assert discr.ndof(ref_sd_0d) == 0
    assert discr.assemble_diff_matrix(ref_sd_0d).nnz == 0
    assert discr.eval_at_cell_centers(ref_sd_0d).nnz == 0
    assert discr.interpolate(ref_sd_0d, lambda x: x).size == 0

    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(ref_sd_0d.dim)

    with pytest.raises(ValueError):
        sd_copy = ref_sd_0d.copy()
        sd_copy.dim = -1
        discr.assemble_diff_matrix(sd_copy)


def test_postprocess_darcy_flux(discr, tilted_sd):
    # The tensor and the slope are given in the ambient coordinates: on the grid only
    # their tangential parts act, through the projection P on the plane of the grid
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    K = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    perm = constant_tensor(sd, K)
    data = pp.initialize_data({}, "test", {pg.SECOND_ORDER_TENSOR: perm})

    # The flux of a linear function is constant, equal to -P K P grad p
    slope = np.array([1.0, 2.0, 3.0])
    p = discr.interpolate(sd, lambda x: slope @ x)
    q = discr.postprocess_darcy_flux(p, sd, data)

    q_space = pg.get_PwPolynomials(discr.poly_order - 1, pg.VECTOR)("test")

    assert np.allclose(q, q_space.interpolate(sd, lambda _: -P @ K @ P @ slope))


def test_assemble_adv_matrix_is_exact(discr, tilted_sd):
    # With u a polynomial of the order of the space and a constant field, both in the
    # ambient coordinates, beta . grad u on the grid is w = (P beta) . grad u, with P
    # the projection on the plane of the grid: a polynomial of order k - 1, which the
    # space interpolates exactly, so that A u = M w
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    x, y, z = pg.exact.coordinates()
    u = (1 + x + 2 * y + z) ** discr.poly_order
    beta = np.array([1.0, 2.0, 3.0])
    grad_u = pg.exact.gradient(u)
    w = sum(b * g for b, g in zip(P @ beta, grad_u))

    u_h = discr.interpolate(sd, pg.exact.to_callable(u))
    w_h = discr.interpolate(sd, pg.exact.to_callable(w))
    field = np.tile(beta.reshape((-1, 1)), sd.num_cells)
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: field})

    A = discr.assemble_adv_matrix(sd, data)
    M = discr.assemble_mass_matrix(sd)
    assert np.allclose(A @ u_h, M @ w_h)
