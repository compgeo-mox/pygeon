import numpy as np

import pygeon as pg


def test_eval_lagrange1(unit_sd):
    mdg = pg.as_mdg(unit_sd)

    discr = pg.Lagrange1("test")
    P = pg.eval_at_cell_centers(mdg, discr)

    assert np.allclose(P.data, 1 / (unit_sd.dim + 1))


def test_eval_at_cc_pwconstants(mdg):
    discr = pg.PwConstants("test")
    P = pg.eval_at_cell_centers(mdg, discr)

    known = np.hstack([1 / sd.cell_volumes for sd in mdg.subdomains()])

    assert np.allclose(P.data, known)
