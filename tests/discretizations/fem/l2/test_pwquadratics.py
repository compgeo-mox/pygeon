"""Module contains specific tests for the P2 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.PwQuadratics:
    return pg.PwQuadratics("test")


def test_ndof(discr, unit_sd):
    factor = [0, 3, 6, 10]
    assert discr.ndof(unit_sd) == unit_sd.num_cells * factor[unit_sd.dim]


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [4, -1, 2],
                        [-1, 4, 2],
                        [2, 2, 16],
                    ]
                )
                / 30
            )

        case 2:
            M_known = (
                np.array(
                    [
                        [6, -1, -1, 0, 0, -4],
                        [-1, 6, -1, 0, -4, 0],
                        [-1, -1, 6, -4, 0, 0],
                        [0, 0, -4, 32, 16, 16],
                        [0, -4, 0, 16, 32, 16],
                        [-4, 0, 0, 16, 16, 32],
                    ]
                )
                / 360
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [3.0, 0.5, 0.5, 0.5, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0],
                        [0.5, 3.0, 0.5, 0.5, -2.0, -3.0, -3.0, -2.0, -2.0, -3.0],
                        [0.5, 0.5, 3.0, 0.5, -3.0, -2.0, -3.0, -2.0, -3.0, -2.0],
                        [0.5, 0.5, 0.5, 3.0, -3.0, -3.0, -2.0, -3.0, -2.0, -2.0],
                        [-2.0, -2.0, -3.0, -3.0, 16.0, 8.0, 8.0, 8.0, 8.0, 4.0],
                        [-2.0, -3.0, -2.0, -3.0, 8.0, 16.0, 8.0, 8.0, 4.0, 8.0],
                        [-2.0, -3.0, -3.0, -2.0, 8.0, 8.0, 16.0, 4.0, 8.0, 8.0],
                        [-3.0, -2.0, -2.0, -3.0, 8.0, 8.0, 4.0, 16.0, 8.0, 8.0],
                        [-3.0, -2.0, -3.0, -2.0, 8.0, 4.0, 8.0, 8.0, 16.0, 8.0],
                        [-3.0, -3.0, -2.0, -2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 16.0],
                    ]
                )
                / 1260
            )

    assert np.allclose(M.todense(), M_known)


def test_source(discr, unit_sd_2d):
    func = lambda _: 2
    source = discr.source_term(unit_sd_2d, func)

    assert np.isclose(source.sum(), 2)
