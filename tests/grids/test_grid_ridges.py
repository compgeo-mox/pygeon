"""Module contains tests to validate the computation of ridges."""

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def test_grid_0d(ref_sd_0d):
    # no ridges or peaks are defined in 0d, we should obtain an empty map with
    # correct size

    assert ref_sd_0d.num_ridges == 0
    assert ref_sd_0d.num_peaks == 0

    assert ref_sd_0d.ridge_peaks.shape == (0, 0)
    assert ref_sd_0d.face_ridges.shape == (0, 0)


def test_grid_1d(unit_sd_1d):
    # no ridges or peaks are defined in 1d, we should obtain an empty map with
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


def test_mdg_2d():
    def setup_problem():
        p = np.array([[0.0, 1.0], [0.5, 0.5]])

        fracs = [pp.LineFracture(p)]

        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        domain = pp.Domain(bounding_box=bbox)
        network = pp.create_fracture_network(fracs, domain)
        mesh_kwargs = {"mesh_size_frac": 1, "mesh_size_min": 1}

        return network.mesh(mesh_kwargs)

    def known_face_ridges():
        data = np.array([-1, 1, 1, -1, 1, -1])
        indices = np.array([0, 1, 10, 11, 2, 3])
        indptr = np.array([0, 2, 4, 6])

        return sps.csc_array((data, indices, indptr), (16, 3))

    mdg = setup_problem()
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    mg = mdg.interfaces()[0]

    assert mg.ridge_peaks.shape == (0, 0)
    assert (mg.face_ridges - known_face_ridges()).nnz == 0


def test_mdg_3d():
    def setup_mdg():
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        )

        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bounding_box=bbox)
        network = pp.create_fracture_network([f_1], domain=domain)
        mesh_args = {"mesh_size_frac": 1, "mesh_size_min": 1}

        return network.mesh(mesh_args)

    def known_face_ridges():
        data = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        indices = np.array([0, 6, 1, 7, 5, 11, 12, 17, 16, 21, 22, 27, 26, 31, 35, 39])
        indptr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])

        return sps.csc_array((data, indices, indptr), (98, 8))

    def known_ridge_peaks():
        data = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 26, 27])
        indptr = np.array([0, 2, 4, 6, 8, 10])

        return sps.csc_array((data, indices, indptr), (28, 5))

    mdg = setup_mdg()
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    mg = mdg.interfaces()[0]

    assert (mg.ridge_peaks - known_ridge_peaks()).nnz == 0
    assert (mg.face_ridges - known_face_ridges()).nnz == 0


def test_mdg_3d_itsc():
    def setup_mdg():
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        )
        f_2 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
        )

        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(bounding_box=bbox)
        network = pp.create_fracture_network([f_1, f_2], domain=domain)
        mesh_args = {"mesh_size_frac": 1, "mesh_size_min": 1}

        return network.mesh(mesh_args)

    def known_face_ridges_mg():
        return np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, -1],
                [0, 1],
                [1, 0],
                [-1, 0],
                [0, 0],
                [0, 0],
            ]
        )

    mdg = setup_mdg()
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    for mg in mdg.interfaces():
        if mg.dim == 1:
            assert mg.ridge_peaks.shape == (0, 0)
            assert np.all(mg.face_ridges.todense() == known_face_ridges_mg())
