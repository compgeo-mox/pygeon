import pytest

import numpy as np

import pygeon as pg


@pytest.fixture
def discr():
    return pg.MatPwQuadratics("test")


def test_ndof(discr, ref_sd):
    num_points = [0, 3, 6, 10]
    discr = pg.MatPwQuadratics()
    assert discr.ndof(ref_sd) == num_points[ref_sd.dim] * ref_sd.dim**2


def test_trace_2d(discr, unit_sd_2d):
    trace = discr.assemble_trace_matrix(unit_sd_2d)

    func = lambda x: np.array([[x[0], x[1]], [x[1], x[0] * x[1]]])
    func_trace = lambda x: x[0] + x[0] * x[1]

    func_interp = discr.interpolate(unit_sd_2d, func)
    trace_interp = pg.PwQuadratics().interpolate(unit_sd_2d, func_trace)

    assert np.allclose(trace @ func_interp, trace_interp)


def test_asym_2d(discr, unit_sd_2d):
    asym = discr.assemble_asym_matrix(unit_sd_2d)

    func = lambda x: np.array([[x[0], x[1]], [x[0] * x[1], x[1]]])
    func_asym = lambda x: x[0] * x[1] - x[1]

    func_interp = discr.interpolate(unit_sd_2d, func)
    asym_interp = pg.PwQuadratics().interpolate(unit_sd_2d, func_asym)

    assert np.allclose(asym @ func_interp, asym_interp)


def test_trace_3d(discr, unit_sd_3d):
    trace = discr.assemble_trace_matrix(unit_sd_3d)

    func = lambda x: np.array(
        [
            [x[2] * x[0], x[1], x[2]],
            [x[0] * x[1], x[1], x[0]],
            [x[0] * x[2], x[1], x[0]],
        ]
    )
    func_trace = lambda x: x[2] * x[0] + x[1] + x[0]

    func_interp = discr.interpolate(unit_sd_3d, func)
    trace_interp = pg.PwQuadratics().interpolate(unit_sd_3d, func_trace)

    assert np.allclose(trace @ func_interp, trace_interp)


def test_asym_3d(discr, unit_sd_3d):
    asym = discr.assemble_asym_matrix(unit_sd_3d)

    func = lambda x: np.array(
        [
            [x[2] * x[0], x[1], x[2]],
            [x[0] * x[1], x[1], x[0]],
            [x[0] * x[2], x[1], x[0]],
        ]
    )
    func_asym = lambda x: np.array(
        [
            x[1] - x[0],
            x[2] - x[0] * x[2],
            x[0] * x[1] - x[1],
        ]
    )

    func_interp = discr.interpolate(unit_sd_3d, func)
    asym_interp = pg.VecPwQuadratics().interpolate(unit_sd_3d, func_asym)

    assert np.allclose(asym @ func_interp, asym_interp)
