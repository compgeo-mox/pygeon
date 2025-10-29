"""Module contains Nedelec unit tests."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Nedelec0,
        pg.Nedelec1,
    ]
)
def discr(request):
    return request.param("test")


def test_assemble_nat_bc(discr, ref_sd_3d):
    with pytest.raises(NotImplementedError):
        discr.assemble_nat_bc(ref_sd_3d, lambda _: np.zeros(1), np.zeros(1))


def test_range_disc(discr):
    assert discr.get_range_discr_class(3) is pg.RT0


def test_interp_eval_constants(discr, unit_sd_3d):
    f = lambda _: np.array([2, 3, -1])
    f_known = np.vstack([f(x) for x in unit_sd_3d.cell_centers.T]).T
    f_known[unit_sd_3d.dim :, :] = 0
    f_known = f_known.ravel()

    P = discr.eval_at_cell_centers(unit_sd_3d)
    f_interp = P @ discr.interpolate(unit_sd_3d, f)

    assert np.allclose(f_interp, f_known)
