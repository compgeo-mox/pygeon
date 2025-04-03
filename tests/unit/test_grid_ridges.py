import unittest

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

"""
Module contains a unit tests to validate the computation of ridges (co-dimension 2 from 
a cell).
"""


class GridRidgesTest(unittest.TestCase):
    def test_grid_0d(self):
        # no ridges or peaks are defined in 0d, we should obtain an empty map with
        # correct size
        sd = pp.PointGrid([0, 0, 0])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, 0)
        self.assertEqual(sd.num_peaks, 0)

        self.assertEqual(sd.ridge_peaks.shape, (0, 0))
        self.assertEqual(sd.face_ridges.shape, (0, 0))

    def test_grid_1d(self):
        # no ridges or peaks are defined in 1d, we should obtain an empty map with
        # correct size
        N = 3
        sd = pp.CartGrid(N)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, 0)
        self.assertEqual(sd.num_peaks, 0)

        self.assertEqual(sd.ridge_peaks.shape, (0, 0))
        self.assertEqual(sd.face_ridges.shape, (0, N + 1))

    def test_grid_2d_cart(self):
        N = 2
        sd = pp.CartGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, (N + 1) ** 2)
        self.assertEqual(sd.num_peaks, 0)

        self.assertEqual(sd.ridge_peaks.shape, (0, (N + 1) ** 2))
        self.assertEqual((sd.face_ridges - self.known_fr_2d_cart()).nnz, 0)

    def test_grid_2d_tris(self):
        N = 2
        sd = pp.StructuredTriangleGrid([N] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, (N + 1) ** 2)
        self.assertEqual(sd.num_peaks, 0)

        self.assertEqual(sd.ridge_peaks.shape, (0, (N + 1) ** 2))
        self.assertEqual((sd.face_ridges - self.known_fr_2d_tris()).nnz, 0)

    def test_grid_3d_cart(self):
        N = 2
        sd = pp.CartGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, 3 * N * (N + 1) ** 2)
        self.assertEqual(sd.num_peaks, (N + 1) ** 3)

        self.assertEqual((sd.ridge_peaks - self.known_rp_3d_cart()).nnz, 0)
        self.assertEqual((sd.face_ridges - self.known_fr_3d_cart()).nnz, 0)

    def test_grid_3d_tet(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        # do the checks
        self.assertEqual(sd.num_ridges, 7 * N**3 + 9 * N**2 + 3 * N)
        self.assertEqual(sd.num_peaks, (N + 1) ** 3)

        self.assertEqual((sd.ridge_peaks - self.known_rp_3d_tet()).nnz, 0)
        self.assertEqual((sd.face_ridges - self.known_fr_3d_tet()).nnz, 0)

    def test_mdg_2d(self):
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

        self.assertEqual(mg.ridge_peaks.shape, (0, 0))
        self.assertEqual((mg.face_ridges - known_face_ridges()).nnz, 0)

    def test_mdg_3d(self):
        def setup_problem():
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
            indices = np.array(
                [0, 6, 1, 7, 5, 11, 12, 17, 16, 21, 22, 27, 26, 31, 35, 39]
            )
            indptr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])

            return sps.csc_array((data, indices, indptr), (98, 8))

        def known_ridge_peaks():
            data = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
            indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 26, 27])
            indptr = np.array([0, 2, 4, 6, 8, 10])

            return sps.csc_array((data, indices, indptr), (28, 5))

        mdg = setup_problem()
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        mg = mdg.interfaces()[0]

        self.assertEqual((mg.ridge_peaks - known_ridge_peaks()).nnz, 0)
        self.assertEqual((mg.face_ridges - known_face_ridges()).nnz, 0)

    def test_mdg_3d_itsc(self):
        def setup_problem():
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

        mdg = setup_problem()
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for mg in mdg.interfaces():
            if mg.dim == 1:
                self.assertEqual(mg.ridge_peaks.shape, (0, 0))
                self.assertTrue(
                    np.all(mg.face_ridges.todense() == known_face_ridges_mg())
                )

    def known_fr_2d_cart(self):
        data = np.array(
            [
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
            ]
        )

        indices = np.array(
            [0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8]
        )
        indptr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])

        return sps.csc_array((data, indices, indptr))

    def known_fr_2d_tris(self):
        data = np.array(
            [
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
            ]
        )
        indices = np.array(
            [
                0,
                1,
                0,
                3,
                0,
                4,
                1,
                2,
                1,
                4,
                1,
                5,
                2,
                5,
                3,
                4,
                3,
                6,
                3,
                7,
                4,
                5,
                4,
                7,
                4,
                8,
                5,
                8,
                6,
                7,
                7,
                8,
            ]
        )
        indptr = np.array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        )

        return sps.csc_array((data, indices, indptr))

    def known_rp_3d_cart(self):
        data = np.array(
            [
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
            ]
        )
        indices = np.array(
            [
                0,
                1,
                0,
                3,
                0,
                9,
                1,
                2,
                1,
                4,
                1,
                10,
                2,
                5,
                2,
                11,
                3,
                4,
                3,
                6,
                3,
                12,
                4,
                5,
                4,
                7,
                4,
                13,
                5,
                8,
                5,
                14,
                6,
                7,
                6,
                15,
                7,
                8,
                7,
                16,
                8,
                17,
                9,
                10,
                9,
                12,
                9,
                18,
                10,
                11,
                10,
                13,
                10,
                19,
                11,
                14,
                11,
                20,
                12,
                13,
                12,
                15,
                12,
                21,
                13,
                14,
                13,
                16,
                13,
                22,
                14,
                17,
                14,
                23,
                15,
                16,
                15,
                24,
                16,
                17,
                16,
                25,
                17,
                26,
                18,
                19,
                18,
                21,
                19,
                20,
                19,
                22,
                20,
                23,
                21,
                22,
                21,
                24,
                22,
                23,
                22,
                25,
                23,
                26,
                24,
                25,
                25,
                26,
            ]
        )
        indptr = np.array(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                42,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
                60,
                62,
                64,
                66,
                68,
                70,
                72,
                74,
                76,
                78,
                80,
                82,
                84,
                86,
                88,
                90,
                92,
                94,
                96,
                98,
                100,
                102,
                104,
                106,
                108,
            ]
        )

        return sps.csc_array((data, indices, indptr))

    def known_fr_3d_cart(self):
        data = np.array(
            [
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
            ]
        )
        indices = np.array(
            [
                1,
                10,
                22,
                2,
                4,
                13,
                25,
                5,
                6,
                15,
                27,
                7,
                9,
                17,
                30,
                10,
                12,
                19,
                33,
                13,
                14,
                20,
                35,
                15,
                22,
                31,
                43,
                23,
                25,
                34,
                45,
                26,
                27,
                36,
                46,
                28,
                30,
                38,
                48,
                31,
                33,
                40,
                50,
                34,
                35,
                41,
                51,
                36,
                2,
                21,
                5,
                0,
                5,
                24,
                7,
                3,
                10,
                29,
                13,
                8,
                13,
                32,
                15,
                11,
                17,
                37,
                19,
                16,
                19,
                39,
                20,
                18,
                23,
                42,
                26,
                21,
                26,
                44,
                28,
                24,
                31,
                47,
                34,
                29,
                34,
                49,
                36,
                32,
                38,
                52,
                40,
                37,
                40,
                53,
                41,
                39,
                0,
                4,
                8,
                1,
                3,
                6,
                11,
                4,
                8,
                12,
                16,
                9,
                11,
                14,
                18,
                12,
                21,
                25,
                29,
                22,
                24,
                27,
                32,
                25,
                29,
                33,
                37,
                30,
                32,
                35,
                39,
                33,
                42,
                45,
                47,
                43,
                44,
                46,
                49,
                45,
                47,
                50,
                52,
                48,
                49,
                51,
                53,
                50,
            ]
        )
        indptr = np.array(
            [
                0,
                4,
                8,
                12,
                16,
                20,
                24,
                28,
                32,
                36,
                40,
                44,
                48,
                52,
                56,
                60,
                64,
                68,
                72,
                76,
                80,
                84,
                88,
                92,
                96,
                100,
                104,
                108,
                112,
                116,
                120,
                124,
                128,
                132,
                136,
                140,
                144,
            ]
        )
        return sps.csc_array((data, indices, indptr))

    def known_rp_3d_tet(self):
        data = np.array(
            [
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
            ]
        )
        indices = np.array(
            [
                0,
                1,
                0,
                2,
                0,
                4,
                1,
                2,
                1,
                3,
                1,
                4,
                1,
                5,
                1,
                6,
                2,
                3,
                2,
                4,
                2,
                6,
                3,
                5,
                3,
                6,
                3,
                7,
                4,
                5,
                4,
                6,
                5,
                6,
                5,
                7,
                6,
                7,
            ]
        )
        indptr = np.array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
        )
        return sps.csc_array((data, indices, indptr))

    def known_fr_3d_tet(self):
        data = np.array(
            [
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
            ]
        )
        indices = np.array(
            [
                0,
                3,
                1,
                0,
                5,
                2,
                1,
                9,
                2,
                3,
                8,
                4,
                3,
                9,
                5,
                3,
                10,
                7,
                4,
                11,
                6,
                4,
                12,
                7,
                5,
                14,
                6,
                5,
                15,
                7,
                6,
                16,
                7,
                8,
                12,
                10,
                9,
                15,
                10,
                11,
                16,
                12,
                11,
                17,
                13,
                12,
                18,
                13,
                14,
                16,
                15,
                16,
                18,
                17,
            ]
        )
        indptr = np.array(
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]
        )
        return sps.csc_array((data, indices, indptr))


if __name__ == "__main__":
    unittest.main()
