"""Module contains specific tests for the BDM1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.BDM1:
    return pg.BDM1("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.dim * unit_sd.num_faces


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
                        [2, 1, 0, 0, 1, 2],
                        [1, 2, 0, 0, 1, 1],
                        [0, 0, 2, -1, 1, 1],
                        [0, 0, -1, 2, -2, -1],
                        [1, 1, 1, -2, 4, 2],
                        [2, 1, 1, -1, 2, 4],
                    ]
                )
                / 24
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [2, 1, 0, 0, 0, 1, 2, -2, 0, 1, 0, 0],
                        [1, 2, 0, 0, 0, 1, 1, -1, 0, 1, 0, 0],
                        [0, 0, 2, 0, -1, 1, 1, 0, 0, 0, 1, 1],
                        [0, 0, 0, 2, 0, 0, 0, 1, 1, -1, 1, 1],
                        [0, 0, -1, 0, 2, -2, -1, 0, 0, 0, -1, -2],
                        [1, 1, 1, 0, -2, 4, 2, -1, 0, 1, 1, 2],
                        [2, 1, 1, 0, -1, 2, 4, -2, 0, 1, 1, 1],
                        [-2, -1, 0, 1, 0, -1, -2, 4, 1, -2, 1, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1, 2, -2, 2, 1],
                        [1, 1, 0, -1, 0, 1, 1, -2, -2, 4, -2, -1],
                        [0, 0, 1, 1, -1, 1, 1, 1, 2, -2, 4, 2],
                        [0, 0, 1, 1, -2, 2, 1, 1, 1, -1, 2, 4],
                    ]
                )
                / 30
            )

    assert np.allclose(M.todense(), M_known)


def test_range_discr_class(discr):
    assert discr.get_range_discr_class(2) is pg.PwConstants


def test_mass_vs_RT0(discr, unit_sd):
    rt0 = pg.RT0("test")
    M_rt0 = rt0.assemble_mass_matrix(unit_sd)

    M_bdm1 = discr.assemble_mass_matrix(unit_sd)
    P = discr.proj_from_RT0(unit_sd)

    difference = M_rt0 - P.T @ M_bdm1 @ P

    assert np.allclose(difference.data, 0)
