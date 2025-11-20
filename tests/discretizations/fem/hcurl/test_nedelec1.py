"""Module contains specific tests for the Nedelec 1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.Nedelec1:
    return pg.Nedelec1("test")


def test_ndof(discr, unit_sd_3d):
    assert discr.ndof(unit_sd_3d) == 2 * unit_sd_3d.num_ridges


def test_interp_eval_linears(discr, unit_sd_3d):
    def q_linear(x):
        return x

    interp_q = discr.interpolate(unit_sd_3d, q_linear)
    eval_q = discr.eval_at_cell_centers(unit_sd_3d) @ interp_q
    eval_q = np.reshape(eval_q, (3, -1))

    known_q = np.array([q_linear(x) for x in unit_sd_3d.cell_centers.T]).T
    assert np.allclose(eval_q, known_q)


def test_mass_matrix(discr, ref_sd_3d):
    with pytest.raises(NotImplementedError):
        discr.assemble_mass_matrix(ref_sd_3d)


def test_assemble_lumped(discr, ref_sd_3d):
    L = discr.assemble_lumped_matrix(ref_sd_3d)

    L_known = (
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, -1, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 3, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
            ]
        )
        / 24
    )

    assert np.allclose(L.todense(), L_known)
