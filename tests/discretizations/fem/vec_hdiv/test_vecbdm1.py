"""Module contains specific tests for the vector BDM1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecBDM1("test")


def test_ndof(discr, ref_sd):
    known = [0, 2, 12, 36]
    assert discr.ndof(ref_sd) == known[ref_sd.dim]


def test_asym_1d(discr, unit_sd_1d):
    with pytest.raises(ValueError):
        discr.assemble_asym_matrix(unit_sd_1d)


def test_trace_2d(discr, unit_sd_2d):
    B = discr.assemble_trace_matrix(unit_sd_2d)

    fun = lambda x: np.array([[x[0] + x[1], x[0], 0], [x[1], -x[0] - x[1], 0]])
    u = discr.interpolate(unit_sd_2d, fun)

    trace = B @ u

    assert np.allclose(trace, 0)


def test_proj_to_and_from_rt0(discr, unit_sd):
    # The function f = x has the property that x_i dot n = x_j dot n for each pair of
    # face nodes x_i and x_j. Therefore its interpolation on RT0 is the same as in BDM1.
    def linear(x):
        return np.array([x, 2 * x, 3 * x])

    interp = discr.interpolate(unit_sd, linear)
    interp_to_rt0 = discr.proj_to_RT0(unit_sd) @ interp
    interp_from_rt0 = discr.proj_from_RT0(unit_sd) @ interp_to_rt0

    assert np.allclose(interp, interp_from_rt0)
