""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VecLagrange1Test(unittest.TestCase):
    def test_div_2d(self):
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

    def test_symgrad_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("vlagrange1")
        B = vec_p1.assemble_symgrad_matrix(sd)

        B_known = 0.25 * np.matrix(
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

    def test_div_3d(self):
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


if __name__ == "__main__":
    unittest.main()
