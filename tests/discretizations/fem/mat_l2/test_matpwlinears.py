"""Module contains specific tests for the matrix P1 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.MatPwLinears:
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


def test_assemble_mult_matrix(discr, ref_sd):
    mult_mat = np.ones((ref_sd.num_nodes, ref_sd.dim, ref_sd.dim))
    vec = np.ones(mult_mat.size)

    mult = discr.assemble_mult_matrix(ref_sd, mult_mat.ravel(), right_mult=True)
    assert np.allclose(mult @ vec, ref_sd.dim * vec)

    mult = discr.assemble_mult_matrix(ref_sd, mult_mat.ravel(), right_mult=False)
    assert np.allclose(mult @ vec, ref_sd.dim * vec)


def test_assemble_corotational_correction(discr, ref_sd):
    if ref_sd.dim < 2:
        return

    if ref_sd.dim == 2:
        rot = np.ones(ref_sd.num_cells)
    else:
        rot = np.ones((ref_sd.num_cells, ref_sd.dim)).ravel()

    vec = np.ones(ref_sd.dim * ref_sd.dim * ref_sd.num_nodes)
    corr = discr.assemble_corotational_correction(ref_sd, rot)

    assert np.allclose(np.sum(corr @ vec), 0)
