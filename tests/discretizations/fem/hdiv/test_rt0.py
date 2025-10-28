import pytest

import numpy as np
import porepy as pp

import pygeon as pg


@pytest.fixture
def discr():
    return pg.RT0("test")


def test_0d(discr):
    sd = pp.PointGrid(np.zeros(3))

    M = discr.assemble_mass_matrix(sd)
    assert M.shape == (0, 0)


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
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 1.0, 2.0],
                    ]
                )
                / 6
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [6.0, -1.0, 1.0, -1.0],
                        [-1.0, 16.0, 4.0, -4.0],
                        [1.0, 4.0, 16.0, 4.0],
                        [-1.0, -4.0, 4.0, 16.0],
                    ]
                )
                / 30
            )

    assert np.allclose(M.todense(), M_known)


def test_mass_matrix(discr, unit_sd):
    M = discr.assemble_mass_matrix(unit_sd)

    discr_pp = pp.RT0("flow")
    data = pg.RT0.create_unitary_data(discr_pp.keyword, unit_sd)
    discr_pp.discretize(unit_sd, data)

    M_pp = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_pp.mass_matrix_key].tocsc()

    assert np.allclose((M - M_pp).data, 0)


def test_range_discr_class(discr):
    assert discr.get_range_discr_class(2) is pg.PwConstants
