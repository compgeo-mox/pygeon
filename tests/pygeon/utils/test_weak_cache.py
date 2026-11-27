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
    assert pg.RT1("a").proj_to_PwPolynomials(sd) is pg.RT1("b").proj_to_PwPolynomials(
        sd
    )


def test_recomputed_with_geometry():
    sd = grid()
    discr = pg.RT1()
    P = discr.proj_to_PwPolynomials(sd)
    assert discr.proj_to_PwPolynomials(sd) is P

    # move the nodes and recompute the geometry from scratch
    sd.nodes *= 2
    sd.history = [h for h in sd.history if h != "Compute geometry"]
    sd.compute_geometry()

    P_new = discr.proj_to_PwPolynomials(sd)
    assert P_new is not P
    assert not np.allclose(P_new.toarray(), P.toarray())
