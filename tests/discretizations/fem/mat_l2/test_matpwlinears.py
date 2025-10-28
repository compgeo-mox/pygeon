import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.MatPwLinears("test")


def test_ndof(discr, ref_sd):
    assert discr.ndof(ref_sd) == ref_sd.dim**2 * (ref_sd.dim + 1)


def test_trace_2d(discr, unit_sd_2d):
    trace = discr.assemble_trace_matrix(unit_sd_2d)

    func = lambda x: np.array([[x[0], x[1]], [x[0], x[1]]])
    func_interp = discr.interpolate(unit_sd_2d, func)

    func_trace = lambda x: x[0] + x[1]
    trace_interp = pg.PwLinears().interpolate(unit_sd_2d, func_trace)

    assert np.allclose(trace @ func_interp, trace_interp)


def test_asym_2d(discr, unit_sd_2d):
    asym = discr.assemble_asym_matrix(unit_sd_2d)

    func = lambda x: np.array([[x[0], x[1]], [x[0], x[1]]])
    func_interp = discr.interpolate(unit_sd_2d, func)

    func_asym = lambda x: x[0] - x[1]
    asym_interp = pg.PwLinears().interpolate(unit_sd_2d, func_asym)

    assert np.allclose(asym @ func_interp, asym_interp)


def test_trace_3d(discr, unit_sd_3d):
    trace = discr.assemble_trace_matrix(unit_sd_3d)

    func = lambda x: np.array(
        [
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
        ]
    )
    func_trace = lambda x: x[0] + x[1] + x[2]

    func_interp = discr.interpolate(unit_sd_3d, func)
    trace_interp = pg.PwLinears().interpolate(unit_sd_3d, func_trace)

    assert np.allclose(trace @ func_interp, trace_interp)


def test_asym_3d(discr, unit_sd_3d):
    asym = discr.assemble_asym_matrix(unit_sd_3d)

    func = lambda x: np.array(
        [
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
            [x[0], x[1], x[2]],
        ]
    )
    func_asym = lambda x: np.array(
        [
            x[1] - x[2],
            x[2] - x[0],
            x[0] - x[1],
        ]
    )

    func_interp = discr.interpolate(unit_sd_3d, func)
    asym_interp = pg.VecPwLinears().interpolate(unit_sd_3d, func_asym)

    assert np.allclose(asym @ func_interp, asym_interp)
