import pytest

import numpy as np
import porepy as pp

import pygeon as pg


@pytest.fixture
def discr():
    return pg.RT0("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_faces


def test_asssemble_mass_matrix(discr, ref_sd):
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


def test_mass_matrix_vs_pp(discr, unit_sd):
    M = discr.assemble_mass_matrix(unit_sd)

    discr_pp = pp.RT0("flow")
    data = pg.RT0.create_unitary_data(discr_pp.keyword, unit_sd)
    discr_pp.discretize(unit_sd, data)

    M_pp = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_pp.mass_matrix_key].tocsc()

    assert np.allclose((M - M_pp).data, 0)


def test_range_discr_class(discr):
    assert discr.get_range_discr_class(2) is pg.PwConstants


def test_error_l2(discr, unit_sd):
    def fun(pt):
        return np.array([pt[0] + 2 * pt[1], 2 * pt[0] + pt[1], 0])

    int_sol = discr.interpolate(unit_sd, fun)

    err = discr.error_l2(unit_sd, np.zeros_like(int_sol), fun)
    assert np.isclose(err, 1)

    err = discr.error_l2(unit_sd, int_sol, fun, etype="standard")
    assert np.isclose(err, 0)

    err = discr.error_l2(unit_sd, int_sol, fun, etype="specific")
    assert not np.isclose(err, 0)
