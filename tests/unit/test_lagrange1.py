"""Module contains a unit test for the Lagrangean P1 discretization."""

import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg  # type: ignore[import-untyped]


class Lagrange1Test(unittest.TestCase):
    def test_0d(self):
        sd = pp.PointGrid([1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()
        discr = pg.Lagrange1()

        D = discr.assemble_diff_matrix(sd).todense()
        D_known = sps.csc_array((0, 1)).todense()

        self.assertTrue(np.allclose(D, D_known))

        P = discr.eval_at_cell_centers(sd).todense()
        P_known = sps.csc_array((1, 0)).todense()

        self.assertTrue(np.allclose(P, P_known))

        sd.dim = -1
        self.assertRaises(ValueError, discr.assemble_diff_matrix, sd)
        self.assertRaises(NotImplementedError, discr.get_range_discr_class, sd.dim)

    def test_1d(self):
        dim = 1
        sd = pp.CartGrid(3, dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.Lagrange1()

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.11111111, 0.05555556, 0.05555556, 0.22222222, 0.05555556,
        0.05555556, 0.22222222, 0.05555556, 0.05555556, 0.11111111]
        )

        M_known_indices = np.array(
        [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
        )

        M_known_indptr = np.array(
        [ 0,  2,  5,  8, 10]
        )
        # fmt: on

        M.sum_duplicates()
        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-1.,  1., -1.,  1., -1.,  1.]
        )

        D_known_indices = np.array(
        [0, 0, 1, 1, 2, 2]
        )

        D_known_indptr = np.array(
        [0, 1, 3, 5, 6]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        Ml = discr.assemble_lumped_matrix(sd)

        # fmt: off
        Ml_known_data = np.array(
        [0.16666667, 0.33333333, 0.33333333, 0.16666667]
        )

        Ml_known_indices = np.array(
        [0, 1, 2, 3]
        )

        Ml_known_indptr = np.array(
        [0, 1, 2, 3, 4]
        )
        # fmt: on

        self.assertTrue(np.allclose(Ml.data, Ml_known_data))
        self.assertTrue(np.allclose(Ml.indptr, Ml_known_indptr))
        self.assertTrue(np.allclose(Ml.indices, Ml_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )

        P_known_indices = np.array(
        [0, 0, 1, 1, 2, 2]
        )

        P_known_indptr = np.array(
        [0, 1, 3, 5, 6]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertTrue(discr.get_range_discr_class(dim) is pg.PwConstants)

    def test_2d(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.Lagrange1()

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.04166667, 0.01041667, 0.01041667, 0.02083333, 0.01041667,
        0.0625    , 0.01041667, 0.02083333, 0.02083333, 0.01041667,
        0.02083333, 0.01041667, 0.01041667, 0.0625    , 0.02083333,
        0.01041667, 0.02083333, 0.02083333, 0.02083333, 0.02083333,
        0.125     , 0.02083333, 0.02083333, 0.02083333, 0.02083333,
        0.01041667, 0.02083333, 0.0625    , 0.01041667, 0.01041667,
        0.02083333, 0.01041667, 0.02083333, 0.02083333, 0.01041667,
        0.0625    , 0.01041667, 0.02083333, 0.01041667, 0.01041667,
        0.04166667]
        )

        M_known_indices = np.array(
        [0, 1, 3, 4, 0, 1, 2, 4, 5, 1, 2, 5, 0, 3, 4, 6, 7, 0, 1, 3, 4, 5,
        7, 8, 1, 2, 4, 5, 8, 3, 6, 7, 3, 4, 6, 7, 8, 4, 5, 7, 8]
        )

        M_known_indptr = np.array(
        [ 0,  4,  9, 12, 17, 24, 29, 32, 37, 41]
        )
        # fmt: on

        M.sum_duplicates()
        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-1., -1., -1.,  1., -1., -1., -1.,  1., -1.,  1., -1., -1., -1.,
        1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1.,
        1.,  1., -1.,  1.,  1.,  1.]
        )

        D_known_indices = np.array(
        [ 0,  1,  2,  0,  3,  4,  5,  3,  6,  1,  7,  8,  9,  2,  4,  7, 10,
        11, 12,  5,  6, 10, 13,  8, 14,  9, 11, 14, 15, 12, 13, 15]
        )

        D_known_indptr = np.array(
        [ 0,  3,  7,  9, 13, 19, 23, 25, 29, 32]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        Ml = discr.assemble_lumped_matrix(sd)

        # fmt: off
        Ml_known_data = np.array(
        [0.08333333, 0.125     , 0.04166667, 0.125     , 0.25      ,
        0.125     , 0.04166667, 0.125     , 0.08333333]
        )

        Ml_known_indices = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )

        Ml_known_indptr = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        # fmt: on

        self.assertTrue(np.allclose(Ml.data, Ml_known_data))
        self.assertTrue(np.allclose(Ml.indptr, Ml_known_indptr))
        self.assertTrue(np.allclose(Ml.indices, Ml_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
        0.33333333, 0.33333333, 0.33333333, 0.33333333]
        )

        P_known_indices = np.array(
        [0, 1, 0, 2, 3, 2, 1, 4, 5, 0, 1, 3, 4, 6, 7, 2, 3, 6, 5, 4, 5, 7,
        6, 7]
        )

        P_known_indptr = np.array(
        [ 0,  2,  5,  6,  9, 15, 18, 19, 22, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertTrue(discr.get_range_discr_class(dim) is pg.RT0)

    def test_3d(self):
        dim = 3
        sd = pp.StructuredTetrahedralGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.Lagrange1()

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.00208333, 0.00104167, 0.00104167, 0.00104167, 0.00104167,
        0.0125    , 0.00104167, 0.003125  , 0.003125  , 0.003125  ,
        0.003125  , 0.00416667, 0.00104167, 0.01041667, 0.003125  ,
        0.00208333, 0.003125  , 0.00208333, 0.00416667, 0.00104167,
        0.003125  , 0.00833333, 0.00208333, 0.00104167, 0.00208333,
        0.003125  , 0.003125  , 0.003125  , 0.00208333, 0.025     ,
        0.00208333, 0.003125  , 0.003125  , 0.00416667, 0.00625   ,
        0.00625   , 0.00416667, 0.00208333, 0.00208333, 0.01666667,
        0.003125  , 0.00208333, 0.00208333, 0.00625   , 0.003125  ,
        0.00416667, 0.00104167, 0.003125  , 0.00625   , 0.00104167,
        0.00208333, 0.00208333, 0.003125  , 0.003125  , 0.00104167,
        0.0125    , 0.00104167, 0.00416667, 0.003125  , 0.003125  ,
        0.00208333, 0.00104167, 0.00625   , 0.00208333, 0.003125  ,
        0.00104167, 0.00104167, 0.003125  , 0.00208333, 0.00833333,
        0.00208333, 0.003125  , 0.00104167, 0.003125  , 0.003125  ,
        0.00416667, 0.00208333, 0.025     , 0.00208333, 0.00625   ,
        0.00625   , 0.003125  , 0.003125  , 0.00416667, 0.00208333,
        0.00208333, 0.00208333, 0.01666667, 0.00625   , 0.003125  ,
        0.003125  , 0.00208333, 0.00416667, 0.00416667, 0.003125  ,
        0.00625   , 0.00208333, 0.003125  , 0.00625   , 0.025     ,
        0.00416667, 0.003125  , 0.00208333, 0.003125  , 0.00416667,
        0.00625   , 0.00625   , 0.00416667, 0.00625   , 0.00625   ,
        0.00416667, 0.05      , 0.00416667, 0.00625   , 0.00625   ,
        0.00416667, 0.00625   , 0.00625   , 0.00416667, 0.003125  ,
        0.00208333, 0.003125  , 0.00416667, 0.025     , 0.00625   ,
        0.003125  , 0.00208333, 0.00625   , 0.003125  , 0.00416667,
        0.00416667, 0.00208333, 0.003125  , 0.003125  , 0.00625   ,
        0.01666667, 0.00208333, 0.00208333, 0.00208333, 0.00416667,
        0.003125  , 0.003125  , 0.00625   , 0.00625   , 0.00208333,
        0.025     , 0.00208333, 0.00416667, 0.003125  , 0.003125  ,
        0.00104167, 0.003125  , 0.00208333, 0.00833333, 0.00208333,
        0.003125  , 0.00104167, 0.00104167, 0.003125  , 0.00208333,
        0.00625   , 0.00104167, 0.00208333, 0.003125  , 0.003125  ,
        0.00416667, 0.00104167, 0.0125    , 0.00104167, 0.003125  ,
        0.003125  , 0.00208333, 0.00208333, 0.00104167, 0.00625   ,
        0.003125  , 0.00104167, 0.00416667, 0.003125  , 0.00625   ,
        0.00208333, 0.00208333, 0.003125  , 0.01666667, 0.00208333,
        0.00208333, 0.00416667, 0.00625   , 0.00625   , 0.00416667,
        0.003125  , 0.003125  , 0.00208333, 0.025     , 0.00208333,
        0.003125  , 0.003125  , 0.003125  , 0.00208333, 0.00104167,
        0.00208333, 0.00833333, 0.003125  , 0.00104167, 0.00416667,
        0.00208333, 0.003125  , 0.00208333, 0.003125  , 0.01041667,
        0.00104167, 0.00416667, 0.003125  , 0.003125  , 0.003125  ,
        0.003125  , 0.00104167, 0.0125    , 0.00104167, 0.00104167,
        0.00104167, 0.00104167, 0.00208333]
        )

        M_known_indices = np.array(
        [ 0,  1,  3,  9,  0,  1,  2,  3,  4,  9, 10, 12,  1,  2,  4,  5, 10,
        11, 13,  0,  1,  3,  4,  6,  9, 12,  1,  2,  3,  4,  5,  6,  7, 10,
        12, 13, 15,  2,  4,  5,  7,  8, 11, 13, 14, 16,  3,  4,  6,  7, 12,
        15,  4,  5,  6,  7,  8, 13, 15, 16,  5,  7,  8, 14, 16, 17,  0,  1,
         3,  9, 10, 12, 18,  1,  2,  4,  9, 10, 11, 12, 13, 18, 19, 21,  2,
         5, 10, 11, 13, 14, 19, 20, 22,  1,  3,  4,  6,  9, 10, 12, 13, 15,
        18, 21,  2,  4,  5,  7, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22, 24,
         5,  8, 11, 13, 14, 16, 17, 20, 22, 23, 25,  4,  6,  7, 12, 13, 15,
        16, 21, 24,  5,  7,  8, 13, 14, 15, 16, 17, 22, 24, 25,  8, 14, 16,
        17, 23, 25, 26,  9, 10, 12, 18, 19, 21, 10, 11, 13, 18, 19, 20, 21,
        22, 11, 14, 19, 20, 22, 23, 10, 12, 13, 15, 18, 19, 21, 22, 24, 11,
        13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 14, 17, 20, 22, 23, 25, 26,
        13, 15, 16, 21, 22, 24, 25, 14, 16, 17, 22, 23, 24, 25, 26, 17, 23,
        25, 26]
        )

        M_known_indptr = np.array(
        [  0,   4,  12,  19,  26,  37,  46,  52,  60,  66,  73,  84,  93,
        104, 119, 130, 139, 150, 157, 163, 171, 177, 186, 197, 204, 211,
        219, 223]
        )
        # fmt: on

        M.sum_duplicates()
        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,
         1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,
        -1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
        -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
         1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,
        -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,
        -1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1,  1,
         1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,  1,
         1,  1,  1,  1,  1, -1,  1,  1,  1]
        )

        D_known_indices = np.array(
        [ 0,  1,  2,  0,  3,  4,  5,  6,  7,  8,  3,  9, 10, 11, 12, 13,  1,
         4, 14, 15, 16, 17,  5,  9, 14, 18, 19, 20, 21, 22, 23, 24, 10, 18,
        25, 26, 27, 28, 29, 30, 15, 19, 31, 32, 33, 20, 25, 31, 34, 35, 36,
        37, 26, 34, 38, 39, 40,  2,  6, 16, 41, 42, 43,  7, 11, 21, 41, 44,
        45, 46, 47, 48, 49, 12, 27, 44, 50, 51, 52, 53, 54,  8, 17, 22, 32,
        42, 45, 55, 56, 57, 58, 13, 23, 28, 35, 46, 50, 55, 59, 60, 61, 62,
        63, 64, 65, 29, 38, 51, 59, 66, 67, 68, 69, 70, 71, 24, 33, 36, 56,
        60, 72, 73, 74, 30, 37, 39, 61, 66, 72, 75, 76, 77, 78, 40, 67, 75,
        79, 80, 81, 43, 47, 57, 82, 83, 48, 52, 62, 82, 84, 85, 86, 53, 68,
        84, 87, 88, 49, 58, 63, 73, 83, 85, 89, 90, 54, 64, 69, 76, 86, 87,
        89, 91, 92, 93, 70, 79, 88, 91, 94, 95, 65, 74, 77, 90, 92, 96, 71,
        78, 80, 93, 94, 96, 97, 81, 95, 97]
        )

        D_known_indptr = np.array(
        [  0,   3,  10,  16,  22,  32,  40,  45,  52,  57,  63,  73,  81,
         91, 105, 115, 123, 133, 139, 144, 151, 156, 164, 174, 180, 186,
        193, 196]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        Ml = discr.assemble_lumped_matrix(sd)

        # fmt: off
        Ml_known_data = np.array(
        [0.00520833, 0.03125   , 0.02604167, 0.02083333, 0.0625    ,
        0.04166667, 0.015625  , 0.03125   , 0.015625  , 0.02083333,
        0.0625    , 0.04166667, 0.0625    , 0.125     , 0.0625    ,
        0.04166667, 0.0625    , 0.02083333, 0.015625  , 0.03125   ,
        0.015625  , 0.04166667, 0.0625    , 0.02083333, 0.02604167,
        0.03125   , 0.00520833]
        )

        Ml_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        )

        Ml_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        )
        # fmt: on

        self.assertTrue(np.allclose(Ml.data, Ml_known_data))
        self.assertTrue(np.allclose(Ml.indptr, Ml_known_indptr))
        self.assertTrue(np.allclose(Ml.indices, Ml_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25]
        )

        P_known_indices = np.array(
        [ 0,  0,  1,  2,  3,  4,  6,  6,  7,  8,  9, 10,  0,  1,  3, 12,  3,
         4,  5,  6,  7,  9, 12, 13, 14, 15, 16, 18,  9, 10, 11, 18, 19, 20,
        21, 22, 12, 13, 15, 15, 16, 17, 18, 19, 21, 21, 22, 23,  0,  1,  2,
        24,  2,  4,  5,  6,  7,  8, 24, 25, 26, 27, 28, 30,  8, 10, 11, 30,
        31, 32, 33, 34,  1,  2,  3,  4,  5, 12, 13, 14, 24, 25, 27, 36,  5,
         7,  8,  9, 10, 11, 14, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 33,
        36, 37, 38, 39, 40, 42, 11, 20, 22, 23, 33, 34, 35, 42, 43, 44, 45,
        46, 13, 14, 15, 16, 17, 36, 37, 39, 17, 19, 20, 21, 22, 23, 39, 40,
        41, 42, 43, 45, 23, 45, 46, 47, 24, 25, 26, 26, 28, 29, 30, 31, 32,
        32, 34, 35, 25, 26, 27, 28, 29, 36, 37, 38, 29, 31, 32, 33, 34, 35,
        38, 40, 41, 42, 43, 44, 35, 44, 46, 47, 37, 38, 39, 40, 41, 41, 43,
        44, 45, 46, 47, 47]
        )

        P_known_indptr = np.array(
        [  0,   1,   7,  12,  16,  28,  36,  39,  45,  48,  52,  64,  72,
         84, 108, 120, 128, 140, 144, 147, 153, 156, 164, 176, 180, 185,
        191, 192]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertTrue(discr.get_range_discr_class(dim) is pg.Nedelec0)

    def test_proj_to_pwlinear(self):
        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            l1 = pg.Lagrange1()
            proj_l1 = l1.proj_to_pwLinears(sd)
            mass_l1 = l1.assemble_mass_matrix(sd)

            p1 = pg.PwLinears()
            mass_p1 = p1.assemble_mass_matrix(sd)

            diff = proj_l1.T @ mass_p1 @ proj_l1 - mass_l1

            self.assertTrue(np.allclose(diff.data, 0.0))

    def test_proj_to_pwconstant(self):
        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            l1 = pg.Lagrange1()
            proj_l1 = l1.proj_to_pwConstants(sd)
            mass_l1 = l1.assemble_mass_matrix(sd)

            p0 = pg.PwConstants()
            mass_p0 = p0.assemble_mass_matrix(sd)

            field = np.ones(sd.num_nodes)
            field_p0 = proj_l1 @ field

            diff = field @ mass_l1 @ field - field_p0 @ mass_p0 @ field_p0

            self.assertTrue(np.isclose(diff, 0.0))

    def test_proj_to_lagrange2(self):
        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            l1 = pg.Lagrange1()
            proj_l1 = l1.proj_to_lagrange2(sd)
            mass_l1 = l1.assemble_mass_matrix(sd)

            l2 = pg.Lagrange2()
            mass_l2 = l2.assemble_mass_matrix(sd)

            diff = proj_l1.T @ mass_l2 @ proj_l1 - mass_l1

            self.assertTrue(np.allclose(diff.data, 0.0))


if __name__ == "__main__":
    unittest.main()
