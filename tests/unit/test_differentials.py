"""Module contains a dummy unit test that always passes."""

import unittest

import numpy as np
import scipy.sparse as sps
import porepy as pp

import pygeon as pg


class DifferentialsUnitTest(unittest.TestCase):
    def test_0d(self):
        sd = pp.PointGrid([0, 0, 0])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        self.assertTrue(np.allclose(div.shape, (1, 0)))
        self.assertTrue(np.allclose(curl.shape, (0, 0)))
        self.assertTrue(np.allclose(grad.shape, (0, 0)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 5)
        self.assertTrue(np.allclose(ext_der.shape, (0, 0)))

    def test_1d(self):
        N, dim = 3, 1
        sd = pp.CartGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        # fmt: off
        known_div = np.array([
        [-1.,  1.,  0.,  0.],
        [ 0., -1.,  1.,  0.],
        [ 0.,  0., -1.,  1.]])

        known_curl = np.zeros((4, 0))

        known_grad = np.zeros((0, 0))
        # fmt: on

        self.assertTrue(np.sum(curl @ grad) == 0)
        self.assertTrue(np.sum(div @ curl) == 0)

        self.assertTrue(np.allclose(div.todense(), known_div))
        self.assertTrue(np.allclose(curl.todense(), known_curl))
        self.assertTrue(np.allclose(grad.todense(), known_grad))

        class Dummy:
            pass

        self.assertRaises(
            TypeError,
            pg.div,
            Dummy(),
        )

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

    def test_2d_simplicial(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        # fmt: off
        known_div = np.array([
        [ 1,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1,  1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0, -1,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0, -1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0, -1,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0, -1]])

        known_curl = np.array([
        [-1,  1,  0,  0,  0,  0,  0,  0,  0],
        [-1,  0,  0,  1,  0,  0,  0,  0,  0],
        [-1,  0,  0,  0,  1,  0,  0,  0,  0],
        [ 0, -1,  1,  0,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  1,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0, -1,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0, -1,  1,  0,  0,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  1,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0, -1,  1,  0,  0,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  1,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  0,  1],
        [ 0,  0,  0,  0,  0, -1,  0,  0,  1],
        [ 0,  0,  0,  0,  0,  0, -1,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0, -1,  1]])

        known_grad = np.zeros((9, 0))
        # fmt: on

        self.assertTrue(np.sum(curl @ grad) == 0)
        self.assertTrue(np.sum(div @ curl) == 0)

        self.assertTrue(np.allclose(div.todense(), known_div))
        self.assertTrue(np.allclose(curl.todense(), known_curl))
        self.assertTrue(np.allclose(grad.todense(), known_grad))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

    def test_2d_cartesian(self):
        N, dim = 2, 2
        sd = pp.CartGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        # fmt: off
        known_div = np.array([
        [-1.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  1.,  0.,  0.,  0., -1.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0., -1.,  0.,  1.]])

        known_curl = np.array([
        [-1,  0,  0,  1,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0, -1,  0,  0,  1,  0,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0, -1,  0,  0,  1],
        [ 1, -1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1, -1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1, -1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1, -1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  1, -1]])

        known_grad = np.zeros((9, 0))
        # fmt: on

        self.assertTrue(np.sum(curl @ grad) == 0)
        self.assertTrue(np.sum(div @ curl) == 0)

        self.assertTrue(np.allclose(div.todense(), known_div))
        self.assertTrue(np.allclose(curl.todense(), known_curl))
        self.assertTrue(np.allclose(grad.todense(), known_grad))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

    def test_3d_simplicial(self):
        N, dim = 2, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        known_div, known_curl, known_grad = self._3d_single_simplicial_grid()

        self.assertTrue(np.sum(curl @ grad) == 0)
        self.assertTrue(np.sum(div @ curl) == 0)

        self.assertTrue(np.allclose(div.todense(), known_div))
        self.assertTrue(np.allclose(curl.todense(), known_curl))
        self.assertTrue(np.allclose(grad.todense(), known_grad))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

    def test_3d_cartesian(self):
        N, dim = 2, 3
        sd = pp.CartGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        div = pg.div(sd)
        curl = pg.curl(sd)
        grad = pg.grad(sd)

        known_div, known_curl, known_grad = self._3d_single_cartesian_grid()

        self.assertTrue(np.sum(curl @ grad) == 0)
        self.assertTrue(np.sum(div @ curl) == 0)

        self.assertTrue(np.allclose(div.todense(), known_div))
        self.assertTrue(np.allclose(curl.todense(), known_curl))
        self.assertTrue(np.allclose(grad.todense(), known_grad))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 0)
        self.assertTrue(np.allclose(ext_der.shape, (0, sd.num_cells)))

        ext_der = pg.numerics.differentials.exterior_derivative(sd, 4)
        self.assertTrue(np.allclose(ext_der.shape, (sd.num_peaks, 0)))

    def _3d_single_simplicial_grid(self):
        # fmt: off
        div_data = np.array(
            [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1])

        div_indptr = np.array(
            [  0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
              52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96, 100,
             104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
             156, 160, 164, 168, 172, 176, 180, 184, 188, 192])

        div_indices = np.array(
            [  0,   1,   2,   6,   6,   7,  11,  24,  10,  11,  12,  56,   5,
               7,   9,  22,   8,   9,  12,  32,  32,  33,  34,  61,   3,   4,
               8,  14,  14,  15,  19,  33,  18,  19,  20,  59,  13,  15,  17,
              26,  16,  17,  20,  42,  42,  43,  44,  69,  21,  22,  23,  28,
              28,  29,  35,  48,  34,  35,  36,  77,  27,  29,  31,  47,  30,
              31,  36,  50,  50,  51,  52,  83,  25,  26,  30,  38,  38,  39,
              45,  51,  44,  45,  46,  81,  37,  39,  41,  49,  40,  41,  46,
              53,  53,  54,  55,  93,  56,  57,  58,  62,  62,  63,  67,  80,
              66,  67,  68, 112,  61,  63,  65,  78,  64,  65,  68,  88,  88,
              89,  90, 114,  59,  60,  64,  70,  70,  71,  75,  89,  74,  75,
              76, 113,  69,  71,  73,  82,  72,  73,  76,  98,  98,  99, 100,
             115,  77,  78,  79,  84,  84,  85,  91, 104,  90,  91,  92, 116,
              83,  85,  87, 103,  86,  87,  92, 106, 106, 107, 108, 118,  81,
              82,  86,  94,  94,  95, 101, 107, 100, 101, 102, 117,  93,  95,
              97, 105,  96,  97, 102, 109, 109, 110, 111, 119])
        # fmt: on

        div = sps.csr_array((div_data, div_indices, div_indptr)).todense()

        # fmt: off
        curl_data = np.array(
            [ 1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,
             -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,
              1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,
              1,  1, -1])

        curl_indptr = np.array(
            [  0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33,  36,
              39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
              78,  81,  84,  87,  90,  93,  96,  99, 102, 105, 108, 111, 114,
             117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153,
             156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192,
             195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231,
             234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270,
             273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309,
             312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348,
             351, 354, 357, 360])

        curl_indices = np.array(
            [ 0,  4,  1,  0,  6,  2,  1, 16,  2,  3,  9,  5,  3, 11,  7,  4, 14,
              5,  4, 16,  6,  4, 17,  8,  5, 21,  7,  5, 22,  8,  6, 41,  7,  6,
             42,  8,  7, 45,  8,  9, 18, 10,  9, 21, 11,  9, 23, 13, 10, 27, 12,
             10, 28, 13, 11, 44, 12, 11, 46, 13, 12, 50, 13, 14, 19, 15, 14, 22,
             17, 15, 32, 17, 16, 42, 17, 18, 25, 20, 18, 28, 23, 19, 31, 20, 19,
             32, 22, 19, 33, 24, 20, 35, 23, 20, 36, 24, 21, 45, 22, 21, 46, 23,
             22, 55, 23, 22, 56, 24, 23, 60, 24, 25, 34, 26, 25, 35, 28, 25, 37,
             30, 26, 38, 29, 26, 39, 30, 27, 50, 28, 27, 51, 29, 28, 59, 29, 28,
             61, 30, 29, 66, 30, 31, 36, 33, 32, 56, 33, 34, 39, 37, 35, 60, 36,
             35, 61, 37, 36, 72, 37, 38, 66, 39, 38, 67, 40, 39, 75, 40, 41, 45,
             42, 41, 47, 43, 42, 57, 43, 44, 50, 46, 44, 52, 48, 45, 55, 46, 45,
             57, 47, 45, 58, 49, 46, 62, 48, 46, 63, 49, 47, 82, 48, 47, 83, 49,
             48, 85, 49, 50, 59, 51, 50, 62, 52, 50, 64, 54, 51, 68, 53, 51, 69,
             54, 52, 84, 53, 52, 86, 54, 53, 87, 54, 55, 60, 56, 55, 63, 58, 56,
             73, 58, 57, 83, 58, 59, 66, 61, 59, 69, 64, 60, 72, 61, 60, 73, 63,
             60, 74, 65, 61, 76, 64, 61, 77, 65, 62, 85, 63, 62, 86, 64, 63, 89,
             64, 63, 90, 65, 64, 92, 65, 66, 75, 67, 66, 76, 69, 66, 78, 71, 67,
             79, 70, 67, 80, 71, 68, 87, 69, 68, 88, 70, 69, 91, 70, 69, 93, 71,
             70, 94, 71, 72, 77, 74, 73, 90, 74, 75, 80, 78, 76, 92, 77, 76, 93,
             78, 77, 96, 78, 79, 94, 80, 79, 95, 81, 80, 97, 81, 82, 85, 83, 84,
             87, 86, 85, 89, 86, 87, 91, 88, 89, 92, 90, 91, 94, 93, 92, 96, 93,
             94, 97, 95])

        # fmt: on

        curl = sps.csr_array((curl_data, curl_indices, curl_indptr)).todense()

        # fmt: off
        grad_data = np.array(
            [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1])

        grad_indptr = np.array(
            [  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
              26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
              52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
              78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102,
             104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128,
             130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
             156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
             182, 184, 186, 188, 190, 192, 194, 196])

        grad_indices = np.array(
            [ 0,  1,  0,  3,  0,  9,  1,  2,  1,  3,  1,  4,  1,  9,  1, 10,  1,
             12,  2,  4,  2,  5,  2, 10,  2, 11,  2, 13,  3,  4,  3,  6,  3,  9,
              3, 12,  4,  5,  4,  6,  4,  7,  4, 10,  4, 12,  4, 13,  4, 15,  5,
              7,  5,  8,  5, 11,  5, 13,  5, 14,  5, 16,  6,  7,  6, 12,  6, 15,
              7,  8,  7, 13,  7, 15,  7, 16,  8, 14,  8, 16,  8, 17,  9, 10,  9,
             12,  9, 18, 10, 11, 10, 12, 10, 13, 10, 18, 10, 19, 10, 21, 11, 13,
             11, 14, 11, 19, 11, 20, 11, 22, 12, 13, 12, 15, 12, 18, 12, 21, 13,
             14, 13, 15, 13, 16, 13, 19, 13, 21, 13, 22, 13, 24, 14, 16, 14, 17,
             14, 20, 14, 22, 14, 23, 14, 25, 15, 16, 15, 21, 15, 24, 16, 17, 16,
             22, 16, 24, 16, 25, 17, 23, 17, 25, 17, 26, 18, 19, 18, 21, 19, 20,
             19, 21, 19, 22, 20, 22, 20, 23, 21, 22, 21, 24, 22, 23, 22, 24, 22,
             25, 23, 25, 23, 26, 24, 25, 25, 26])
        # fmt: on

        grad = sps.csr_array((grad_data, grad_indices, grad_indptr)).todense()

        return div, curl, grad

    def _3d_single_cartesian_grid(self):
        # fmt: off
        div_data = np.array(
            [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1])

        div_indptr = np.array([ 0,  6, 12, 18, 24, 30, 36, 42, 48])

        div_indices = np.array(
            [ 0,  1, 12, 14, 24, 28,  1,  2, 13, 15, 25, 29,  3,  4, 14, 16, 26,
             30,  4,  5, 15, 17, 27, 31,  6,  7, 18, 20, 28, 32,  7,  8, 19, 21,
             29, 33,  9, 10, 20, 22, 30, 34, 10, 11, 21, 23, 31, 35])
        # fmt: on

        div = sps.csr_array((div_data, div_indices, div_indptr)).todense()

        # fmt: off
        curl_data = np.array(
            [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
              1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,
             -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1,
             -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
              1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
              1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,
             -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1,
             -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
              1,  1, -1, -1,  1,  1, -1, -1])

        curl_indptr = np.array(
            [  0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
              52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96, 100,
             104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144])

        curl_indices = np.array(
            [ 1, 10, 22,  2,  4, 13, 25,  5,  6, 15, 27,  7,  9, 17, 30, 10, 12,
             19, 33, 13, 14, 20, 35, 15, 22, 31, 43, 23, 25, 34, 45, 26, 27, 36,
             46, 28, 30, 38, 48, 31, 33, 40, 50, 34, 35, 41, 51, 36,  2, 21,  5,
              0,  5, 24,  7,  3, 10, 29, 13,  8, 13, 32, 15, 11, 17, 37, 19, 16,
             19, 39, 20, 18, 23, 42, 26, 21, 26, 44, 28, 24, 31, 47, 34, 29, 34,
             49, 36, 32, 38, 52, 40, 37, 40, 53, 41, 39,  0,  4,  8,  1,  3,  6,
             11,  4,  8, 12, 16,  9, 11, 14, 18, 12, 21, 25, 29, 22, 24, 27, 32,
             25, 29, 33, 37, 30, 32, 35, 39, 33, 42, 45, 47, 43, 44, 46, 49, 45,
             47, 50, 52, 48, 49, 51, 53, 50])
        # fmt: on

        curl = sps.csr_array((curl_data, curl_indices, curl_indptr)).todense()

        # fmt: off
        grad_data = np.array(
            [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
              1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
             -1,  1, -1,  1, -1,  1])

        grad_indptr = np.array(
            [  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
              26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
              52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
              78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102,
             104, 106, 108])

        grad_indices = np.array(
            [ 0,  1,  0,  3,  0,  9,  1,  2,  1,  4,  1, 10,  2,  5,  2, 11,  3,
              4,  3,  6,  3, 12,  4,  5,  4,  7,  4, 13,  5,  8,  5, 14,  6,  7,
              6, 15,  7,  8,  7, 16,  8, 17,  9, 10,  9, 12,  9, 18, 10, 11, 10,
             13, 10, 19, 11, 14, 11, 20, 12, 13, 12, 15, 12, 21, 13, 14, 13, 16,
             13, 22, 14, 17, 14, 23, 15, 16, 15, 24, 16, 17, 16, 25, 17, 26, 18,
             19, 18, 21, 19, 20, 19, 22, 20, 23, 21, 22, 21, 24, 22, 23, 22, 25,
             23, 26, 24, 25, 25, 26])
        # fmt: on

        grad = sps.csr_array((grad_data, grad_indices, grad_indptr)).todense()

        return div, curl, grad


if __name__ == "__main__":
    unittest.main()
