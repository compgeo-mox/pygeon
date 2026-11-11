"""Module contains general tests for all H(div) discretizations."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.RT0,
        pg.BDM1,
        pg.RT1,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_interp_eval_constants(discr, unit_sd):
    f = lambda _: np.array([2, 3, -1])
    f_known = np.vstack([f(x) for x in unit_sd.cell_centers.T]).T
    f_known[unit_sd.dim :, :] = 0
    f_known = f_known.ravel()

    P = discr.eval_at_cell_centers(unit_sd)
    f_interp = P @ discr.interpolate(unit_sd, f)

    assert np.allclose(f_interp, f_known)


def test_interp_eval_linears(discr, unit_sd):
    if isinstance(discr, pg.RT0):
        pytest.skip("RT0 does not contain all the linear functions")

    def q_linear(x):
        return x

    interp_q = discr.interpolate(unit_sd, q_linear)
    eval_q = discr.eval_at_cell_centers(unit_sd) @ interp_q
    eval_q = np.reshape(eval_q, (pg.AMBIENT_DIM, -1))

    known_q = np.array([q_linear(x) for x in unit_sd.cell_centers.T]).T
    assert np.allclose(eval_q, known_q)


def test_norm_of_linear_function(discr, unit_sd):
    if isinstance(discr, pg.RT0):
        pytest.skip("RT0 does not contain all the linear functions")

    def q_linear(x):
        return x

    interp = discr.interpolate(unit_sd, q_linear)
    M = discr.assemble_mass_matrix(unit_sd)

    computed_norm = interp @ M @ interp

    assert np.isclose(computed_norm, unit_sd.dim / 3)


def test_point_grid(discr, ref_sd_0d):
    """Tests with a point grid"""

    assert discr.ndof(ref_sd_0d) == 0
    assert discr.assemble_mass_matrix(ref_sd_0d).nnz == 0
    assert discr.assemble_lumped_matrix(ref_sd_0d).nnz == 0
    assert discr.eval_at_cell_centers(ref_sd_0d).shape == (pg.AMBIENT_DIM, 0)


def test_broken_div(discr, unit_sd):
    div = discr.assemble_diff_matrix(unit_sd)

    proj = discr.proj_to_PwPolynomials(unit_sd)
    pwp = pg.get_PwPolynomials(discr.poly_order, pg.VECTOR)()
    broken_div = pwp.assemble_broken_div_matrix(unit_sd)

    assert np.allclose((div - broken_div @ proj).data, 0)


def test_assemble_adv_matrix_is_exact(discr, tilted_sd):
    # With u = a + b x, in all the spaces, and the tensor and the field constant, all
    # in the ambient coordinates, K^{-1} beta . u on the grid is
    # f = (P K^{-1} P beta) . u, with P the projection on the plane of the grid: a
    # linear function, which the range of the divergence tests exactly, so that
    # A u = M f
    sd = tilted_sd
    if isinstance(discr, pg.RT1) and sd.dim < 3:
        # TODO: RT1 evaluates its basis and interpolates the cell dofs with the
        # ambient components, so that it fails on grids out of the coordinate planes
        pytest.xfail("RT1 is not yet supported on tilted grids")
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    inv_K = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    beta = np.array([1.0, 2.0, 3.0])
    a, b = np.array([0.5, -1.0, 2.0]), 1.5
    w = P @ inv_K @ P @ beta

    perm = pg.constant_tensor(sd.num_cells, inv_K)
    field = np.tile(beta.reshape((-1, 1)), sd.num_cells)
    data = pp.initialize_data(
        {}, "test", {pg.SECOND_ORDER_TENSOR: perm, pg.VECTOR_FIELD: field}
    )

    range_discr = discr.get_range_discr_class(sd.dim)("test")
    u_h = discr.interpolate(sd, lambda x: a.reshape((-1, 1)) + b * x)
    f_h = range_discr.interpolate(sd, lambda x: w @ (a.reshape((-1, 1)) + b * x))

    A = discr.assemble_adv_matrix(sd, data)
    M = range_discr.assemble_mass_matrix(sd)
    assert np.allclose(A @ u_h, M @ f_h)
