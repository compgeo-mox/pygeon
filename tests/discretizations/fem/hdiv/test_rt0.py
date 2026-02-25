"""Module contains specific tests for the RT0 discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


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

    assert np.allclose(M.todense(), M_known)


def test_assemble_adv_matrix(discr, ref_sd, vector_field):
    data = pp.initialize_data({}, "test", {pg.VECTOR_FIELD: vector_field})
    M = discr.assemble_adv_matrix(ref_sd, data)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [-1, -1],
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

    assert np.allclose(M.todense(), M_known)


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
        return np.array([pt[0] ** 2 + 2 * pt[1], 2 * pt[0] + pt[1], 0])

    int_sol = discr.interpolate(unit_sd, fun)

    # Test that the relative error is 1 for a zero distribution
    err = discr.error_l2(unit_sd, np.zeros_like(int_sol), fun)
    assert np.isclose(err, 1)

    # Test that the error is 0 if num_sol is the interpolant
    err = discr.error_l2(unit_sd, int_sol, fun)
    assert np.isclose(err, 0)

    # Test that the error is nonzero with lowest-order integration
    err = discr.error_l2(unit_sd, int_sol, fun, poly_order=0)
    assert not np.isclose(err, 0)
