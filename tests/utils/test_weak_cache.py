"""Module contains tests to validate the cache of the projections."""

import gc
import weakref

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
