"""Module contains tests to validate the cache of the projections."""

import gc
import weakref

import numpy as np

import pygeon as pg


def grid() -> pg.Grid:
    sd = pg.unit_grid(2, 0.5, as_mdg=False)
    sd.compute_geometry()
    return sd


def test_grid_released():
    sd = grid()
    pg.RT1().proj_to_PwPolynomials(sd)
    ref = weakref.ref(sd)

    del sd
    gc.collect()
    assert ref() is None


def test_shared_by_instances():
    sd = grid()
    assert pg.RT0("a").proj_to_PwPolynomials(sd) is pg.RT0("b").proj_to_PwPolynomials(
        sd
    )


def test_subclasses_apart():
    # both inherit VecDiscretization.proj_to_PwPolynomials
    sd = grid()
    P_bdm1 = pg.VecBDM1().proj_to_PwPolynomials(sd)
    P_rt1 = pg.VecRT1().proj_to_PwPolynomials(sd)

    assert P_bdm1.shape[1] == pg.VecBDM1().ndof(sd)
    assert P_rt1.shape[1] == pg.VecRT1().ndof(sd)


def test_recomputed_with_geometry():
    sd = grid()
    discr = pg.RT0()
    P = discr.proj_to_PwPolynomials(sd)
    assert discr.proj_to_PwPolynomials(sd) is P

    # move the nodes and recompute the geometry from scratch
    sd.nodes *= 2
    sd.history = [h for h in sd.history if h != "Compute geometry"]
    sd.compute_geometry()

    P_new = discr.proj_to_PwPolynomials(sd)
    assert P_new is not P
    assert not np.allclose(P_new.toarray(), P.toarray())
