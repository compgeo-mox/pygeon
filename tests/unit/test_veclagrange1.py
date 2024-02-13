""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VecLagrange1Test(unittest.TestCase):
    def test_mass_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")

        M = vec_p1.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.16666667, 0.04166667, 0.04166667, 0.08333333, 0.04166667,
        0.08333333, 0.04166667, 0.04166667, 0.08333333, 0.04166667,
        0.08333333, 0.04166667, 0.04166667, 0.16666667, 0.16666667,
        0.04166667, 0.04166667, 0.08333333, 0.04166667, 0.08333333,
        0.04166667, 0.04166667, 0.08333333, 0.04166667, 0.08333333,
        0.04166667, 0.04166667, 0.16666667]
        )

        M_known_indices = np.array(
        [0, 1, 2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 7, 4,
        6, 7, 4, 5, 6, 7]
        )

        M_known_indptr = np.array(
        [ 0,  4,  7, 10, 14, 18, 21, 24, 28]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_mass_3d(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")

        M = vec_p1.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.01666667, 0.00833333, 0.00833333, 0.00833333, 0.00833333,
        0.08333333, 0.025     , 0.01666667, 0.025     , 0.01666667,
        0.03333333, 0.00833333, 0.025     , 0.05      , 0.00833333,
        0.01666667, 0.01666667, 0.01666667, 0.00833333, 0.05      ,
        0.01666667, 0.025     , 0.00833333, 0.00833333, 0.025     ,
        0.01666667, 0.05      , 0.00833333, 0.01666667, 0.01666667,
        0.01666667, 0.00833333, 0.05      , 0.025     , 0.00833333,
        0.03333333, 0.01666667, 0.025     , 0.01666667, 0.025     ,
        0.08333333, 0.00833333, 0.00833333, 0.00833333, 0.00833333,
        0.01666667, 0.01666667, 0.00833333, 0.00833333, 0.00833333,
        0.00833333, 0.08333333, 0.025     , 0.01666667, 0.025     ,
        0.01666667, 0.03333333, 0.00833333, 0.025     , 0.05      ,
        0.00833333, 0.01666667, 0.01666667, 0.01666667, 0.00833333,
        0.05      , 0.01666667, 0.025     , 0.00833333, 0.00833333,
        0.025     , 0.01666667, 0.05      , 0.00833333, 0.01666667,
        0.01666667, 0.01666667, 0.00833333, 0.05      , 0.025     ,
        0.00833333, 0.03333333, 0.01666667, 0.025     , 0.01666667,
        0.025     , 0.08333333, 0.00833333, 0.00833333, 0.00833333,
        0.00833333, 0.01666667, 0.01666667, 0.00833333, 0.00833333,
        0.00833333, 0.00833333, 0.08333333, 0.025     , 0.01666667,
        0.025     , 0.01666667, 0.03333333, 0.00833333, 0.025     ,
        0.05      , 0.00833333, 0.01666667, 0.01666667, 0.01666667,
        0.00833333, 0.05      , 0.01666667, 0.025     , 0.00833333,
        0.00833333, 0.025     , 0.01666667, 0.05      , 0.00833333,
        0.01666667, 0.01666667, 0.01666667, 0.00833333, 0.05      ,
        0.025     , 0.00833333, 0.03333333, 0.01666667, 0.025     ,
        0.01666667, 0.025     , 0.08333333, 0.00833333, 0.00833333,
        0.00833333, 0.00833333, 0.01666667]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  4,  0,  1,  2,  3,  4,  5,  6,  0,  1,  2,  3,  4,  6,
         1,  2,  3,  5,  6,  7,  0,  1,  2,  4,  5,  6,  1,  3,  4,  5,  6,
         7,  1,  2,  3,  4,  5,  6,  7,  3,  5,  6,  7,  8,  9, 10, 12,  8,
         9, 10, 11, 12, 13, 14,  8,  9, 10, 11, 12, 14,  9, 10, 11, 13, 14,
        15,  8,  9, 10, 12, 13, 14,  9, 11, 12, 13, 14, 15,  9, 10, 11, 12,
        13, 14, 15, 11, 13, 14, 15, 16, 17, 18, 20, 16, 17, 18, 19, 20, 21,
        22, 16, 17, 18, 19, 20, 22, 17, 18, 19, 21, 22, 23, 16, 17, 18, 20,
        21, 22, 17, 19, 20, 21, 22, 23, 17, 18, 19, 20, 21, 22, 23, 19, 21,
        22, 23]
        )

        M_known_indptr = np.array(
        [  0,   4,  11,  17,  23,  29,  35,  42,  46,  50,  57,  63,  69,
        75,  81,  88,  92,  96, 103, 109, 115, 121, 127, 134, 138]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_div_0d(self):
        sd = pp.PointGrid([1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_div_matrix(sd).todense()

        B_known = np.array([[0]])
        self.assertTrue(np.allclose(B, B_known))

    def test_div_2d(self):
        """
        Test the div operator in 2D using VecLagrange1.

        This method tests the computation of the divergence matrix, interpolation of a function,
        and the assembly of the divergence-divergence matrix using VecLagrange1.

        Returns:
            None
        """
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_div_matrix(sd).todense()

        B_known = 0.5 * np.array(
            [[-1, 1, 0, 0, 0, -1, 0, 1], [0, 0, -1, 1, -1, 0, 1, 0]]
        )
        self.assertTrue(np.allclose(B, B_known))

        fun = lambda x: np.array([-x[1], x[0]])
        fun_interp = vec_p1.interpolate(sd, fun)

        self.assertTrue(np.allclose(B @ fun_interp, 0))

        A = vec_p1.assemble_div_div_matrix(sd).todense()

        self.assertTrue(np.allclose(A.T, A))
        self.assertTrue(np.allclose(A @ np.ones(8), 0))

        A_known = 0.5 * np.array(
            [
                [1, -1, 0, 0, 0, 1, 0, -1],
                [-1, 1, 0, 0, 0, -1, 0, 1],
                [0, 0, 1, -1, 1, 0, -1, 0],
                [0, 0, -1, 1, -1, 0, 1, 0],
                [0, 0, 1, -1, 1, 0, -1, 0],
                [1, -1, 0, 0, 0, 1, 0, -1],
                [0, 0, -1, 1, -1, 0, 1, 0],
                [-1, 1, 0, 0, 0, -1, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(A, A_known))

    def test_symgrad_0d(self):
        sd = pp.PointGrid([1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_symgrad_matrix(sd).todense()

        B_known = sps.csc_matrix((1, 1)).todense()
        self.assertTrue(np.allclose(B, B_known))

    def test_symgrad_2d(self):
        """
        Test the symgrad_2d method of VecLagrange1 class.

        This method tests the computation of the symmetric gradient matrix,
        interpolation, and assembly of the symmetric gradient-symmetric gradient matrix.

        Returns:
            None
        """
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_symgrad_matrix(sd)

        B_known = 0.25 * np.array(
            [
                [-2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 2, 0, 0, 0, 0],
                [0, -1, 0, 1, -1, 1, 0, 0],
                [-1, 0, 1, 0, 0, 0, -1, 1],
                [0, -1, 0, 1, -1, 1, 0, 0],
                [-1, 0, 1, 0, 0, 0, -1, 1],
                [0, 0, 0, 0, 0, -2, 0, 2],
                [0, 0, 0, 0, -2, 0, 2, 0],
            ]
        )

        self.assertTrue(np.allclose(B.todense(), B_known))

        fun = lambda x: np.array([-x[1], x[0]])
        fun_interp = vec_p1.interpolate(sd, fun)

        self.assertTrue(np.allclose(B @ fun_interp, 0))

        A = vec_p1.assemble_symgrad_symgrad_matrix(sd).todense()

        self.assertTrue(np.allclose(A.T, A))
        self.assertTrue(np.allclose(A @ np.ones(8), 0))

        A_known = 0.25 * np.array(
            [
                [3, -2, -1, 0, 0, 0, 1, -1],
                [-2, 3, 0, -1, 1, -1, 0, 0],
                [-1, 0, 3, -2, 0, 0, -1, 1],
                [0, -1, -2, 3, -1, 1, 0, 0],
                [0, 1, 0, -1, 3, -1, -2, 0],
                [0, -1, 0, 1, -1, 3, 0, -2],
                [1, 0, -1, 0, -2, 0, 3, -1],
                [-1, 0, 1, 0, 0, -2, -1, 3],
            ]
        )
        self.assertTrue(np.allclose(A, A_known))

        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_div_3d(self):
        """
        Test the div operator in 3D using VecLagrange1.

        This test verifies the correctness of the div operator by assembling the
        divergence matrix and comparing it with the known indices, indptr, and data.
        It also checks if the divergence of the interpolated functions is zero.
        Finally, it verifies the symmetry and the property of the assembled
        symgrad_symgrad matrix.

        Returns:
            None
        """
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_div_matrix(sd)
        B.eliminate_zeros()

        # fmt: off
        B_indices_known = np.array(
            [0, 0, 1, 4, 1, 3, 3, 4, 1, 2, 2, 4, 1, 4, 5, 5, 0, 3, 4, 0, 3, 4,
             1, 2, 5, 1, 2, 5, 0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5], dtype=int)

        B_indptr_known = np.array(
            [ 0,  1,  4,  6,  8, 10, 12, 15, 16, 17, 19, 20, 22, 24, 25, 27, 28,
             29, 31, 33, 34, 35, 37, 39, 40], dtype=int)

        B_data_known = np.array(
            [-1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1,
             -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
              1,  1,  1,  1,  1,  1]) / 6
        # fmt: on
        B_known = sps.csc_matrix((B_data_known, B_indices_known, B_indptr_known))
        self.assertTrue(np.allclose(sps.find(B), sps.find(B_known)))

        fun_x = lambda x: np.array([0, -x[2], x[1]])
        fun_y = lambda x: np.array([-x[2], 0, x[0]])
        fun_z = lambda x: np.array([-x[1], x[0], 0])
        for fun in [fun_x, fun_y, fun_z]:
            fun_interp = vec_p1.interpolate(sd, fun)
            self.assertTrue(np.allclose(B @ fun_interp, 0))

        A = vec_p1.assemble_symgrad_symgrad_matrix(sd).todense()

        self.assertTrue(np.allclose(A.T, A))
        self.assertTrue(np.allclose(A @ np.ones(24), 0))

        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_diff_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_diff_matrix(sd)
        B.eliminate_zeros()

        # fmt: off
        B_known_data = np.array(
        [-0.5 , -0.25, -0.25, -0.5 ,  0.5 , -0.25, -0.25,  0.5 , -0.5 ,
         0.25,  0.25, -0.5 ,  0.5 ,  0.25,  0.25,  0.5 , -0.25, -0.25,
        -0.5 , -0.5 ,  0.25,  0.25, -0.5 , -0.5 , -0.25, -0.25,  0.5 ,
         0.5 ,  0.25,  0.25,  0.5 ,  0.5 ]
        )

        B_known_indices = np.array(
        [0, 3, 5, 8, 0, 2, 4, 8, 1, 3, 5, 9, 1, 2, 4, 9, 2, 4, 7, 9, 2, 4,
        6, 8, 3, 5, 7, 9, 3, 5, 6, 8]
        )

        B_known_indptr = np.array(
        [ 0,  4,  8, 12, 16, 20, 24, 28, 32]
        )
        # fmt: on

        self.assertTrue(np.allclose(B.data, B_known_data))
        self.assertTrue(np.allclose(B.indptr, B_known_indptr))
        self.assertTrue(np.allclose(B.indices, B_known_indices))

        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_diff_3d(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_diff_matrix(sd)
        B.eliminate_zeros()

        # fmt: off
        B_known_data = np.array(
        [-0.16666667, -0.08333333, -0.08333333, -0.08333333, -0.08333333,
        -0.16666667,  0.16666667,  0.16666667, -0.16666667, -0.08333333,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333, -0.08333333,
        -0.08333333, -0.08333333,  0.16666667,  0.16666667, -0.16666667,
        -0.16666667, -0.16666667,  0.08333333, -0.08333333, -0.08333333,
         0.08333333, -0.08333333, -0.08333333, -0.16666667, -0.16666667,
         0.16666667,  0.16666667,  0.08333333,  0.08333333, -0.08333333,
         0.08333333,  0.08333333, -0.08333333,  0.16666667,  0.16666667,
        -0.16666667, -0.16666667, -0.08333333, -0.08333333,  0.08333333,
        -0.08333333, -0.08333333,  0.08333333, -0.16666667, -0.16666667,
         0.16666667,  0.16666667, -0.08333333,  0.08333333,  0.08333333,
        -0.08333333,  0.08333333,  0.08333333,  0.16666667,  0.16666667,
         0.16666667, -0.16666667, -0.16666667,  0.08333333,  0.08333333,
         0.08333333,  0.08333333,  0.08333333,  0.08333333,  0.08333333,
         0.08333333,  0.16666667, -0.16666667, -0.16666667,  0.16666667,
         0.08333333,  0.08333333,  0.08333333,  0.08333333,  0.16666667,
        -0.08333333, -0.08333333, -0.16666667, -0.08333333, -0.08333333,
        -0.16666667,  0.08333333,  0.08333333, -0.08333333,  0.08333333,
         0.08333333, -0.08333333, -0.16666667, -0.16666667, -0.08333333,
        -0.08333333, -0.08333333, -0.08333333, -0.16666667, -0.16666667,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333,  0.16666667,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333,  0.16666667,
         0.08333333,  0.08333333,  0.08333333,  0.08333333,  0.16666667,
         0.16666667, -0.08333333, -0.08333333,  0.16666667,  0.16666667,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333, -0.16666667,
        -0.16666667,  0.08333333,  0.08333333, -0.16666667, -0.16666667,
         0.08333333,  0.08333333,  0.08333333,  0.08333333, -0.16666667,
         0.08333333,  0.08333333,  0.08333333,  0.08333333, -0.16666667,
         0.08333333, -0.08333333, -0.08333333,  0.08333333, -0.08333333,
        -0.08333333,  0.16666667,  0.16666667,  0.08333333,  0.08333333,
         0.08333333,  0.08333333,  0.16666667,  0.16666667,  0.08333333,
         0.08333333,  0.16666667,  0.08333333,  0.08333333,  0.16666667,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333, -0.16666667,
        -0.16666667,  0.08333333,  0.08333333, -0.08333333, -0.08333333,
        -0.08333333,  0.08333333,  0.08333333, -0.08333333, -0.08333333,
        -0.08333333, -0.16666667, -0.16666667, -0.16666667, -0.16666667,
        -0.08333333, -0.08333333,  0.08333333, -0.08333333, -0.08333333,
         0.08333333, -0.16666667, -0.16666667, -0.16666667, -0.16666667,
         0.08333333,  0.08333333,  0.08333333,  0.08333333,  0.08333333,
         0.08333333,  0.08333333,  0.08333333, -0.16666667, -0.16666667,
        -0.08333333, -0.08333333, -0.08333333, -0.08333333, -0.08333333,
        -0.08333333, -0.08333333, -0.08333333,  0.16666667,  0.16666667,
         0.08333333,  0.08333333, -0.08333333,  0.08333333,  0.08333333,
        -0.08333333,  0.16666667,  0.16666667,  0.16666667,  0.16666667,
         0.08333333, -0.08333333, -0.08333333,  0.08333333,  0.08333333,
         0.08333333, -0.08333333, -0.08333333,  0.08333333,  0.08333333,
         0.16666667,  0.16666667,  0.16666667,  0.16666667,  0.08333333,
         0.08333333,  0.08333333,  0.08333333,  0.16666667,  0.16666667]
        )

        B_known_indices = np.array(
        [ 0,  6, 12, 18, 36, 54,  0,  1,  4,  9, 10, 14, 16, 21, 22, 38, 40,
        54, 55, 58,  1,  3,  6, 13, 15, 18, 37, 39, 55, 57,  3,  4,  9, 10,
        17, 21, 22, 41, 57, 58,  1,  2,  7,  8, 12, 19, 20, 36, 55, 56,  2,
         4, 11, 14, 16, 23, 38, 40, 56, 58,  1,  4,  5,  7,  8, 13, 15, 19,
        20, 37, 39, 55, 58, 59,  5, 11, 17, 23, 41, 59,  6, 18, 24, 30, 42,
        54,  6,  7, 10, 18, 19, 22, 27, 28, 32, 34, 44, 46, 57, 58,  7,  9,
        19, 21, 24, 31, 33, 43, 45, 54,  9, 10, 21, 22, 27, 28, 35, 47, 57,
        58,  7,  8, 19, 20, 25, 26, 30, 42, 55, 56,  8, 10, 20, 22, 29, 32,
        34, 44, 46, 59,  7, 10, 11, 19, 22, 23, 25, 26, 31, 33, 43, 45, 55,
        56, 11, 23, 29, 35, 47, 59, 12, 30, 36, 42, 48, 54, 12, 13, 16, 33,
        34, 36, 37, 40, 45, 46, 50, 52, 56, 58, 13, 15, 30, 37, 39, 42, 49,
        51, 55, 57, 15, 16, 33, 34, 39, 40, 45, 46, 53, 59, 13, 14, 31, 32,
        37, 38, 43, 44, 48, 54, 14, 16, 35, 38, 40, 47, 50, 52, 56, 58, 13,
        16, 17, 31, 32, 37, 40, 41, 43, 44, 49, 51, 55, 57, 17, 35, 41, 47,
        53, 59]
        )

        B_known_indptr = np.array(
        [  0,   6,  20,  30,  40,  50,  60,  74,  80,  86, 100, 110, 120,
        130, 140, 154, 160, 166, 180, 190, 200, 210, 220, 234, 240]
        )
        # fmt: on

        self.assertTrue(np.allclose(B.data, B_known_data))
        self.assertTrue(np.allclose(B.indptr, B_known_indptr))
        self.assertTrue(np.allclose(B.indices, B_known_indices))

        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_eval_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        P = vec_p1.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333]
        )

        P_known_indices = np.array(
        [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3]
        )

        P_known_indptr = np.array(
        [ 0,  2,  3,  4,  6,  8,  9, 10, 12]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertRaises(NotImplementedError, vec_p1.get_range_discr_class, 2)
        self.assertEqual(vec_p1.ndof(sd), sd.dim * sd.num_nodes)

    def test_eval_3d(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        P = vec_p1.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        )

        P_known_indices = np.array(
        [ 0,  0,  1,  2,  3,  4,  0,  1,  3,  3,  4,  5,  0,  1,  2,  2,  4,
         5,  1,  2,  3,  4,  5,  5,  6,  6,  7,  8,  9, 10,  6,  7,  9,  9,
        10, 11,  6,  7,  8,  8, 10, 11,  7,  8,  9, 10, 11, 11, 12, 12, 13,
        14, 15, 16, 12, 13, 15, 15, 16, 17, 12, 13, 14, 14, 16, 17, 13, 14,
        15, 16, 17, 17]
        )

        P_known_indptr = np.array(
        [ 0,  1,  6,  9, 12, 15, 18, 23, 24, 25, 30, 33, 36, 39, 42, 47, 48,
        49, 54, 57, 60, 63, 66, 71, 72]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertRaises(NotImplementedError, vec_p1.get_range_discr_class, 3)


if __name__ == "__main__":
    unittest.main()
