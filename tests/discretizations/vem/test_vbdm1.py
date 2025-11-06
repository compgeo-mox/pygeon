import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VBDM1("test")


def test_ndof(discr, pentagon_sd):
    assert discr.ndof(pentagon_sd) == 10


def test_mass_matrix(discr, ref_square):
    M = discr.assemble_mass_matrix(ref_square)

    M_known = (
        np.array(
            [
                [17, -9, -9, 13, 0, 0, 0, 0],
                [-9, 17, 13, -9, 0, 0, 0, 0],
                [-9, 13, 17, -9, 0, 0, 0, 0],
                [13, -9, -9, 17, 0, 0, 0, 0],
                [0, 0, 0, 0, 17, -9, -9, 13],
                [0, 0, 0, 0, -9, 17, 13, -9],
                [0, 0, 0, 0, -9, 13, 17, -9],
                [0, 0, 0, 0, 13, -9, -9, 17],
            ]
        )
        / 48
    )

    assert np.allclose(M.todense(), M_known)


def test_proj_to_VRT0(discr, ref_square):
    P = discr.proj_to_VRT0(ref_square)

    P_known = (
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
        / 2
    )

    assert np.allclose(P.todense(), P_known)


def test_proj_from_RT0(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.proj_from_RT0(ref_square)


def test_diff_matrix(discr, ref_square):
    D = discr.assemble_diff_matrix(ref_square)

    D_known = np.array([[-1, -1, 1, 1, -1, -1, 1, 1]]) / 2
    assert np.allclose(D.todense(), D_known)


def test_eval_at_cc(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.eval_at_cell_centers(ref_square)


def test_interpolate(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.interpolate(ref_square, None)


def test_lumped(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.assemble_lumped_matrix(ref_square)


def test_assemble_nat_bc(discr, pentagon_sd):
    fun = lambda x: x[0] + x[1]
    faces = pentagon_sd.tags["domain_boundary_faces"]

    vals = discr.assemble_nat_bc(pentagon_sd, fun, faces.nonzero()[0])
    vals_from_bool = discr.assemble_nat_bc(pentagon_sd, fun, faces)

    vals_known = np.array([6, 12, 22, 26, 31, 32, 30, 27, 16, 8]) / 12

    assert np.allclose(vals, vals_known)
    assert np.allclose(vals_from_bool, vals_known)
