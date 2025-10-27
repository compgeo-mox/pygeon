import numpy as np
import pytest
import scipy.sparse as sps

import pygeon as pg


@pytest.fixture
def discr():
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
                        [4.0, -1.0, 2.0],
                        [-1.0, 4.0, 2.0],
                        [2.0, 2.0, 16.0],
                    ]
                )
                / 30
            )

        case 2:
            M_known = (
                np.array(
                    [
                        [6.0, -1.0, -1.0, 0.0, 0.0, -4.0],
                        [-1.0, 6.0, -1.0, 0.0, -4.0, 0.0],
                        [-1.0, -1.0, 6.0, -4.0, 0.0, 0.0],
                        [0.0, 0.0, -4.0, 32.0, 16.0, 16.0],
                        [0.0, -4.0, 0.0, 16.0, 32.0, 16.0],
                        [-4.0, 0.0, 0.0, 16.0, 16.0, 32.0],
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

    source_known = np.zeros(discr.ndof(unit_sd_2d))
    source_known[(unit_sd_2d.dim + 1) * unit_sd_2d.num_cells :] = 1 / 12

    assert np.allclose(source, source_known)
