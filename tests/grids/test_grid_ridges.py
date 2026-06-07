"""Module contains tests to validate the computation of ridges."""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def test_grid_0d(ref_sd_0d):
    # no ridges or peaks are defined in 0D, we should obtain an empty map with
    # correct size

    assert ref_sd_0d.num_ridges == 0
    assert ref_sd_0d.num_peaks == 0

    assert ref_sd_0d.ridge_peaks.shape == (0, 0)
    assert ref_sd_0d.face_ridges.shape == (0, 0)


def test_grid_1d(unit_sd_1d):
    # no ridges or peaks are defined in 1D, we should obtain an empty map with
    # correct size

    assert unit_sd_1d.num_ridges == 0
    assert unit_sd_1d.num_peaks == 0

    assert unit_sd_1d.ridge_peaks.shape == (0, 0)
    assert unit_sd_1d.face_ridges.shape == (0, 5)


def test_grid_2d_cart():
    N = 2
    sd = pp.CartGrid([N] * 2, [1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == (N + 1) ** 2
    assert sd.num_peaks == 0

    assert sd.ridge_peaks.shape == (0, (N + 1) ** 2)


def test_grid_2d_tris():
    N = 2
    sd = pp.StructuredTriangleGrid([N] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == (N + 1) ** 2
    assert sd.num_peaks == 0

    assert sd.ridge_peaks.shape == (0, (N + 1) ** 2)


def test_grid_3d_cart():
    N = 2
    sd = pp.CartGrid([N] * 3, [1] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == 3 * N * (N + 1) ** 2
    assert sd.num_peaks == (N + 1) ** 3


def test_grid_3d_tet():
    N = 1
    sd = pp.StructuredTetrahedralGrid([N] * 3)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    assert sd.num_ridges == 7 * N**3 + 9 * N**2 + 3 * N
    assert sd.num_peaks == (N + 1) ** 3


def test_mdg_2d(mdg_embedded_frac_2d):
    def known_face_ridges():
        data = np.array([1, -1])
        indices = np.array([10, 11])
        indptr = np.array([0, 0, 2, 2])

        return sps.csc_array((data, indices, indptr), (16, 3))

    mg = mdg_embedded_frac_2d.interfaces()[0]

    assert mg.ridge_peaks.shape == (0, 0)
    assert (mg.face_ridges - known_face_ridges()).nnz == 0


def test_mdg_3d(mdg):
    for mg in mdg.interfaces():
        sd_down = mg.sd_pair[1]

        # Check that the lower-dimensional faces occur twice, except at tips.
        if mg.dim >= 1:
            sums = np.sum(np.abs(mg.face_ridges), axis=0)
            assert np.allclose(sums[sd_down.tags["tip_faces"]], 0)
            assert np.allclose(sums[~sd_down.tags["tip_faces"]], 2)

        # Check that the lower-dimensional ridges occur twice, except at tips.
        if mg.dim >= 2:
            sums = np.sum(np.abs(mg.ridge_peaks), axis=0)
            assert np.allclose(sums[sd_down.tags["tip_nodes"]], 0)
            assert np.allclose(sums[~sd_down.tags["tip_nodes"]], 2)


def test_mdg_3d_itsc(_mdg_dict):
    mdg = _mdg_dict["fracs_3D"]

    for mg in mdg.interfaces(dim=1):
        assert mg.ridge_peaks.shape == (0, 0)
        assert mg.face_ridges.shape == (20, 2)

    for mg in mdg.interfaces(dim=2):
        assert mg.ridge_peaks.shape == (112, 20)
        assert mg.face_ridges.shape == (392, 32)
