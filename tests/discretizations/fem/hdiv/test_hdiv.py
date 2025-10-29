import numpy as np
import pytest

import pygeon as pg

"""
Module contains general tests for all H(div) discretizations.
"""


@pytest.fixture(
    params=[
        pg.RT0,
        pg.BDM1,
        pg.RT1,
    ]
)
def discr(request):
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
        return

    def q_linear(x):
        return x

    interp_q = discr.interpolate(unit_sd, q_linear)
    eval_q = discr.eval_at_cell_centers(unit_sd) @ interp_q
    eval_q = np.reshape(eval_q, (3, -1))

    known_q = np.array([q_linear(x) for x in unit_sd.cell_centers.T]).T
    assert np.allclose(eval_q, known_q)


def test_norm_of_linear_function(discr, unit_sd):
    if isinstance(discr, pg.RT0):
        return

    def q_linear(x):
        return x

    interp = discr.interpolate(unit_sd, q_linear)
    M = discr.assemble_mass_matrix(unit_sd)

    computed_norm = interp @ M @ interp

    assert np.isclose(computed_norm, unit_sd.dim / 3)
