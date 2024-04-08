""" Module contains a dummy unit test that always passes.
"""

import unittest
import numpy as np

import porepy as pp
import pygeon as pg
import scipy.sparse as sps


class SubVolumeTest(unittest.TestCase):
    def test_quads(self):
        sd = pp.CartGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

        sub_volumes, sub_simplices = sd.compute_subvolumes(return_subsimplices=True)

        # fmt: off
        sub_simplices_known_data = np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        )

        sub_simplices_known_indices = np.array(
        [ 0,  1, 20, 24,  1,  2, 21, 25,  2,  3, 22, 26,  3,  4, 23, 27,  5,
         6, 24, 28,  6,  7, 25, 29,  7,  8, 26, 30,  8,  9, 27, 31, 10, 11,
        28, 32, 11, 12, 29, 33, 12, 13, 30, 34, 13, 14, 31, 35, 15, 16, 32,
        36, 16, 17, 33, 37, 17, 18, 34, 38, 18, 19, 35, 39]
        )

        sub_simplices_known_indptr = np.array(
        [ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        )
        # fmt: on

        self.assertTrue(np.allclose(sub_simplices.data, sub_simplices_known_data))
        self.assertTrue(np.allclose(sub_simplices.indices, sub_simplices_known_indices))
        self.assertTrue(np.allclose(sub_simplices.indptr, sub_simplices_known_indptr))

    def test_tris(self):
        sd = pp.StructuredTriangleGrid([4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

        sub_volumes, sub_simplices = sd.compute_subvolumes(return_subsimplices=True)

        # fmt: off
        sub_simplices_known_data = np.array(
        [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
        0.16666667]
        )

        sub_simplices_known_indices = np.array(
        [ 0,  2,  4,  1,  2, 13,  3,  5,  7,  4,  5, 16,  6,  8, 10,  7,  8,
         19,  9, 11, 12, 10, 11, 22, 13, 15, 17, 14, 15, 26, 16, 18, 20, 17,
         18, 29, 19, 21, 23, 20, 21, 32, 22, 24, 25, 23, 24, 35, 26, 28, 30,
         27, 28, 39, 29, 31, 33, 30, 31, 42, 32, 34, 36, 33, 34, 45, 35, 37,
         38, 36, 37, 48, 39, 41, 43, 40, 41, 52, 42, 44, 46, 43, 44, 53, 45,
         47, 49, 46, 47, 54, 48, 50, 51, 49, 50, 55]
        )

        sub_simplices_known_indptr = np.array(
        [ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
        51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96]
        )
        # fmt: on

        sub_simplices.sum_duplicates()
        self.assertTrue(np.allclose(sub_simplices.data, sub_simplices_known_data))
        self.assertTrue(np.allclose(sub_simplices.indices, sub_simplices_known_indices))
        self.assertTrue(np.allclose(sub_simplices.indptr, sub_simplices_known_indptr))

    def test_oct(self):
        sd = pg.OctagonGrid([5, 5])
        sd.compute_geometry()
        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

        sub_volumes, sub_simplices = sd.compute_subvolumes(return_subsimplices=True)

        # fmt: off
        sub_simplices_known_data = np.array(
        [0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00414214, 0.00414214, 0.00414214, 0.00414214, 0.00414214,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00171573,
        0.00171573, 0.00171573, 0.00171573, 0.00171573, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00114382, 0.00114382, 0.00114382,
        0.00114382, 0.00114382, 0.00057191, 0.00057191, 0.00057191,
        0.00057191, 0.00057191, 0.00057191, 0.00057191, 0.00057191,
        0.00057191, 0.00057191, 0.00057191, 0.00057191]
        )

        sub_simplices_known_indices = np.array(
        [  0,   5,  30,  31,  60,  85, 110, 135,   1,   6,  31,  32,  61,
         86, 111, 136,   2,   7,  32,  33,  62,  87, 112, 137,   3,   8,
         33,  34,  63,  88, 113, 138,   4,   9,  34,  35,  64,  89, 114,
        139,   5,  10,  36,  37,  65,  90, 115, 140,   6,  11,  37,  38,
         66,  91, 116, 141,   7,  12,  38,  39,  67,  92, 117, 142,   8,
         13,  39,  40,  68,  93, 118, 143,   9,  14,  40,  41,  69,  94,
        119, 144,  10,  15,  42,  43,  70,  95, 120, 145,  11,  16,  43,
         44,  71,  96, 121, 146,  12,  17,  44,  45,  72,  97, 122, 147,
         13,  18,  45,  46,  73,  98, 123, 148,  14,  19,  46,  47,  74,
         99, 124, 149,  15,  20,  48,  49,  75, 100, 125, 150,  16,  21,
         49,  50,  76, 101, 126, 151,  17,  22,  50,  51,  77, 102, 127,
        152,  18,  23,  51,  52,  78, 103, 128, 153,  19,  24,  52,  53,
         79, 104, 129, 154,  20,  25,  54,  55,  80, 105, 130, 155,  21,
         26,  55,  56,  81, 106, 131, 156,  22,  27,  56,  57,  82, 107,
        132, 157,  23,  28,  57,  58,  83, 108, 133, 158,  24,  29,  58,
         59,  84, 109, 134, 159,  66,  90, 111, 135,  67,  91, 112, 136,
         68,  92, 113, 137,  69,  93, 114, 138,  71,  95, 116, 140,  72,
         96, 117, 141,  73,  97, 118, 142,  74,  98, 119, 143,  76, 100,
        121, 145,  77, 101, 122, 146,  78, 102, 123, 147,  79, 103, 124,
        148,  81, 105, 126, 150,  82, 106, 127, 151,  83, 107, 128, 152,
         84, 108, 129, 153,  61,  85, 160,  62,  86, 161,  63,  87, 162,
         64,  88, 163, 131, 155, 164, 132, 156, 165, 133, 157, 166, 134,
        158, 167,  65, 110, 168,  70, 115, 169,  75, 120, 170,  80, 125,
        171,  94, 139, 172,  99, 144, 173, 104, 149, 174, 109, 154, 175,
         60, 176, 177,  89, 178, 179, 130, 180, 181, 159, 182, 183]
        )

        sub_simplices_known_indptr = np.array(
        [  0,   8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
        104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
        204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252,
        256, 260, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294,
        297, 300, 303, 306, 309, 312, 315, 318, 321, 324]
        )
        # fmt: on

        sub_simplices.sum_duplicates()
        self.assertTrue(np.allclose(sub_simplices.data, sub_simplices_known_data))
        self.assertTrue(np.allclose(sub_simplices.indices, sub_simplices_known_indices))
        self.assertTrue(np.allclose(sub_simplices.indptr, sub_simplices_known_indptr))

    def test_tets(self):
        sd = pp.StructuredTetrahedralGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_hexes(self):
        sd = pp.CartGrid([4, 4, 4])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_hitchhiker_pentagon(self):
        nodes = np.array([[0, 3, 3, 3.0 / 2, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        indptr = np.arange(0, 11, 2)
        indices = np.roll(np.repeat(np.arange(5), 2), -1)
        face_nodes = sps.csc_matrix((np.ones(10), indices, indptr))
        cell_faces = sps.csc_matrix(np.ones((5, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_concave_quad(self):
        nodes = np.array([[0, 0.5, 1, 0.5], [0, 0.5, 0, 1], np.zeros(4)])
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        face_nodes = sps.csc_matrix((np.ones(8), indices, np.arange(0, 9, 2)))
        cell_faces = sps.csc_matrix(np.ones((4, 1)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()
        self.assertTrue(np.allclose(sd.cell_volumes, np.sum(sub_volumes, 0)))

    def test_multiple_concave_quads(self):
        nodes = np.array(
            [[0, 0.5, 1, 0.5, 0.5, 0.5], [0, 0.5, 0, 1, -0.5, -1], np.zeros(6)]
        )
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 5, 5, 2, 2, 4, 4, 0])
        face_nodes = sps.csc_matrix((np.ones(16), indices, np.arange(0, 17, 2)))
        cell_faces_j = np.repeat(np.arange(3), 4)
        cell_faces_i = np.array([0, 1, 2, 3, 7, 6, 1, 0, 4, 5, 6, 7])
        cell_faces_v = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
        cell_faces = sps.csc_matrix((cell_faces_v, (cell_faces_i, cell_faces_j)))

        sd = pp.Grid(2, nodes, face_nodes, cell_faces, "concave")
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        sub_volumes = sd.compute_subvolumes()

        known_sub_volumes = np.array(
            [
                [6.25e-02, 1.25e-01, 6.25e-02],
                [0, 1.25e-01, 0],
                [6.25e-02, 1.25e-01, 6.25e-02],
                [1.25e-01, 0, 0],
                [0, 1.25e-01, 0],
                [0, 0, 1.25e-01],
            ]
        )
        self.assertTrue(np.allclose(sub_volumes.todense(), known_sub_volumes))


if __name__ == "__main__":
    unittest.main()
