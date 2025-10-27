import pytest

import numpy as np

import pygeon as pg

"""
Module contains tests to validate the consistency between H(div) discretizations.
"""


def test_mass(unit_sd):
    rt0 = pg.RT0("test")
    M_rt0 = rt0.assemble_mass_matrix(unit_sd)

    bdm1 = pg.BDM1("test")
    M_bdm1 = bdm1.assemble_mass_matrix(unit_sd)
    P = bdm1.proj_from_RT0(unit_sd)

    difference = M_rt0 - P.T @ M_bdm1 @ P

    assert np.allclose(difference.data, 0)


def test_interp_eval_constants(unit_sd):
    f = lambda _: np.array([2, 3, -1])

    rt0 = pg.RT0()
    P = rt0.eval_at_cell_centers(unit_sd)
    f_rt0 = P @ rt0.interpolate(unit_sd, f)

    bdm1 = pg.BDM1()
    P = bdm1.eval_at_cell_centers(unit_sd)
    f_bdm1 = P @ bdm1.interpolate(unit_sd, f)

    assert np.allclose(f_rt0, f_bdm1)
