"""Module contains specific tests for the RT0 discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg
from tests.helpers import matrix_equals


@pytest.fixture
def discr() -> pg.RT0:
    return pg.RT0("test")


@pytest.fixture
def vector_field() -> np.ndarray:
    return np.array([[1], [1], [1]])


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_faces


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [2, 1],
                        [1, 2],
                    ]
                )
                / 6
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, 2, 1],
                        [0, 1, 2],
                    ]
                )
                / 6
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [6, -1, 1, -1],
                        [-1, 16, 4, -4],
                        [1, 4, 16, 4],
                        [-1, -4, 4, 16],
                    ]
                )
                / 30
            )

    assert matrix_equals(M.todense(), M_known)


def test_assemble_adv_matrix(discr, ref_sd, vector_field):
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: vector_field})
    M = discr.assemble_adv_matrix(ref_sd, data)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [1, 1],
                    ]
                )
                / 2
            )
        case 2:
            M_known = (
                np.array(
                    [
                        [2, 1, -1],
                    ]
                )
                / 3
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [3, 1, -1, 1],
                    ]
                )
                / 2
            )

    assert matrix_equals(M.todense(), M_known)


def test_assemble_adv_matrix_is_exact(discr, tilted_sd):
    # With u = a + b x in RT0, and the tensor and the field constant, all in the
    # ambient coordinates, D^{-1} beta . u on the grid is f = (P D^{-1} P beta) . u,
    # with P the projection on the plane of the grid: a linear function, whose mean on
    # each cell is tested by the piecewise constants, so that A u = M f
    sd = tilted_sd
    P = sd.rotation_matrix.T @ sd.rotation_matrix

    D_inv = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    beta = np.array([1.0, 2.0, 3.0])
    a, b = np.array([0.5, -1.0, 2.0]), 1.5
    w = P @ D_inv @ P @ beta

    ones = np.ones(sd.num_cells)
    perm = pp.SecondOrderTensor(
        kxx=D_inv[0, 0] * ones,
        kyy=D_inv[1, 1] * ones,
        kzz=D_inv[2, 2] * ones,
        kxy=D_inv[0, 1] * ones,
        kxz=D_inv[0, 2] * ones,
        kyz=D_inv[1, 2] * ones,
    )
    field = np.tile(beta.reshape((-1, 1)), sd.num_cells)
    data = pp.initialize_data(
        {}, "test", {pg.SECOND_ORDER_TENSOR: perm, pg.VECTOR_FIELD: field}
    )

    p0 = pg.PwConstants("test")
    u_h = discr.interpolate(sd, lambda x: a.reshape((-1, 1)) + b * x)
    f_h = p0.interpolate(sd, lambda x: w @ (a.reshape((-1, 1)) + b * x))

    A = discr.assemble_adv_matrix(sd, data)
    M = p0.assemble_mass_matrix(sd)
    assert np.allclose(A @ u_h, M @ f_h)


def test_assemble_adv_matrix_default(discr, ref_sd):
    M = discr.assemble_adv_matrix(ref_sd)

    assert np.allclose(M.todense(), 0)


def test_mass_matrix_vs_pp(discr, unit_sd):
    M = discr.assemble_mass_matrix(unit_sd)

    discr_pp = pp.RT0(discr.keyword)

    perm = pg.get_cell_data(
        unit_sd, {}, discr.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
    )
    data = pp.initialize_data({}, discr.keyword, {pg.SECOND_ORDER_TENSOR: perm})

    discr_pp.discretize(unit_sd, data)

    M_pp = data[pp.DISCRETIZATION_MATRICES][discr.keyword][
        discr_pp.mass_matrix_key
    ].tocsc()

    assert np.allclose((M - M_pp).data, 0)


def test_eval_at_cc_vs_pp(discr, unit_sd):
    P = discr.eval_at_cell_centers(unit_sd)

    discr_pp = pp.RT0(discr.keyword)

    perm = pg.get_cell_data(
        unit_sd, {}, discr.keyword, pg.SECOND_ORDER_TENSOR, pg.MATRIX
    )
    data = pp.initialize_data({}, discr.keyword, {pg.SECOND_ORDER_TENSOR: perm})

    discr_pp.discretize(unit_sd, data)
    P_pp = data[pp.DISCRETIZATION_MATRICES][discr_pp.keyword][discr_pp.vector_proj_key]

    # Translate from porepy to pygeon ordering
    indices = np.reshape(np.arange(3 * unit_sd.num_cells), (3, -1), order="F").ravel()

    assert np.allclose((P_pp.tolil()[indices] - P).data, 0)


def test_range_discr_class(discr):
    assert discr.get_range_discr_class(2) is pg.PwConstants


def test_error_l2(discr, unit_sd):
    def fun(pt):
        return np.array([pt[0] ** 2 + 2 * pt[1], 2 * pt[0] + pt[1], 0 * pt[0]])

    int_sol = discr.interpolate(unit_sd, fun)

    # Test that the relative error is 1 for a zero distribution
    err = discr.error_l2(unit_sd, np.zeros_like(int_sol), fun)
    assert np.isclose(err, 1)

    # Test that the error is 0 if num_sol is the interpolant
    err = discr.error_l2(unit_sd, int_sol, fun, poly_order=None)
    assert np.isclose(err, 0)

    # Test that the error is nonzero with lowest-order integration
    err = discr.error_l2(unit_sd, int_sol, fun, poly_order=0)
    assert not np.isclose(err, 0)


def test_nat_bc_of_constant_function(discr, unit_sd):
    # A constant datum is broadcast to the boundary faces as its vectorized version
    b_faces = unit_sd.tags["domain_boundary_faces"]
    constant = discr.assemble_nat_bc(unit_sd, lambda _: 2.0, b_faces)
    vectorized = discr.assemble_nat_bc(
        unit_sd, lambda x: 2.0 * np.ones(x.shape[1]), b_faces
    )

    assert np.allclose(constant, vectorized)
    assert np.allclose(np.abs(constant), 2.0 * b_faces)
