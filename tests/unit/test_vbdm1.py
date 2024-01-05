""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VBDM1Test(unittest.TestCase):
    def test_on_cart_grid(self):
        sd = pp.CartGrid([2] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.VBDM1("flow")

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [ 0.35416667, -0.1875    , -0.1875    ,  0.27083333, -0.        ,
        0.        ,  0.        , -0.        , -0.1875    ,  0.35416667,
        0.27083333, -0.1875    ,  0.        , -0.        , -0.        ,
        0.        , -0.1875    ,  0.27083333,  0.70833333, -0.375     ,
        -0.1875    ,  0.27083333,  0.        , -0.        , -0.        ,
        0.        , -0.        ,  0.        ,  0.        , -0.        ,
        0.27083333, -0.1875    , -0.375     ,  0.70833333,  0.27083333,
        -0.1875    , -0.        ,  0.        ,  0.        , -0.        ,
        0.        , -0.        , -0.        ,  0.        , -0.1875    ,
        0.27083333,  0.35416667, -0.1875    ,  0.        , -0.        ,
        -0.        ,  0.        ,  0.27083333, -0.1875    , -0.1875    ,
        0.35416667, -0.        ,  0.        ,  0.        , -0.        ,
        0.35416667, -0.1875    , -0.1875    ,  0.27083333, -0.        ,
        0.        ,  0.        , -0.        , -0.1875    ,  0.35416667,
        0.27083333, -0.1875    ,  0.        , -0.        , -0.        ,
        0.        , -0.1875    ,  0.27083333,  0.70833333, -0.375     ,
        -0.1875    ,  0.27083333,  0.        , -0.        , -0.        ,
        0.        , -0.        ,  0.        ,  0.        , -0.        ,
        0.27083333, -0.1875    , -0.375     ,  0.70833333,  0.27083333,
        -0.1875    , -0.        ,  0.        ,  0.        , -0.        ,
        0.        , -0.        , -0.        ,  0.        , -0.1875    ,
        0.27083333,  0.35416667, -0.1875    ,  0.        , -0.        ,
        -0.        ,  0.        ,  0.27083333, -0.1875    , -0.1875    ,
        0.35416667, -0.        ,  0.        ,  0.        , -0.        ,
        -0.        ,  0.        ,  0.        , -0.        ,  0.35416667,
        -0.1875    , -0.1875    ,  0.27083333,  0.        , -0.        ,
        -0.        ,  0.        , -0.1875    ,  0.35416667,  0.27083333,
        -0.1875    , -0.        ,  0.        ,  0.        , -0.        ,
        0.35416667, -0.1875    , -0.1875    ,  0.27083333,  0.        ,
        -0.        , -0.        ,  0.        , -0.1875    ,  0.35416667,
        0.27083333, -0.1875    ,  0.        , -0.        , -0.        ,
        0.        , -0.        ,  0.        ,  0.        , -0.        ,
        -0.1875    ,  0.27083333,  0.70833333, -0.375     , -0.1875    ,
        0.27083333, -0.        ,  0.        ,  0.        , -0.        ,
        0.        , -0.        , -0.        ,  0.        ,  0.27083333,
        -0.1875    , -0.375     ,  0.70833333,  0.27083333, -0.1875    ,
        0.        , -0.        , -0.        ,  0.        , -0.        ,
        0.        ,  0.        , -0.        , -0.1875    ,  0.27083333,
        0.70833333, -0.375     , -0.1875    ,  0.27083333, -0.        ,
        0.        ,  0.        , -0.        ,  0.        , -0.        ,
        -0.        ,  0.        ,  0.27083333, -0.1875    , -0.375     ,
        0.70833333,  0.27083333, -0.1875    ,  0.        , -0.        ,
        -0.        ,  0.        , -0.1875    ,  0.27083333,  0.35416667,
        -0.1875    , -0.        ,  0.        ,  0.        , -0.        ,
        0.27083333, -0.1875    , -0.1875    ,  0.35416667,  0.        ,
        -0.        , -0.        ,  0.        , -0.1875    ,  0.27083333,
        0.35416667, -0.1875    , -0.        ,  0.        ,  0.        ,
        -0.        ,  0.27083333, -0.1875    , -0.1875    ,  0.35416667])

        M_known_indptr = np.array(
        [  0,   8,  16,  30,  44,  52,  60,  68,  76,  90, 104, 112, 120,
        128, 136, 144, 152, 166, 180, 194, 208, 216, 224, 232, 240])

        M_known_indices = np.array(
        [ 0,  1,  2,  3, 12, 13, 16, 17,  0,  1,  2,  3, 12, 13, 16, 17,  0,
        1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,
        4,  5, 12, 13, 14, 15, 16, 17, 18, 19,  2,  3,  4,  5, 14, 15, 18,
        19,  2,  3,  4,  5, 14, 15, 18, 19,  6,  7,  8,  9, 16, 17, 20, 21,
        6,  7,  8,  9, 16, 17, 20, 21,  6,  7,  8,  9, 10, 11, 16, 17, 18,
        19, 20, 21, 22, 23,  6,  7,  8,  9, 10, 11, 16, 17, 18, 19, 20, 21,
        22, 23,  8,  9, 10, 11, 18, 19, 22, 23,  8,  9, 10, 11, 18, 19, 22,
        23,  0,  1,  2,  3, 12, 13, 16, 17,  0,  1,  2,  3, 12, 13, 16, 17,
        2,  3,  4,  5, 14, 15, 18, 19,  2,  3,  4,  5, 14, 15, 18, 19,  0,
        1,  2,  3,  6,  7,  8,  9, 12, 13, 16, 17, 20, 21,  0,  1,  2,  3,
        6,  7,  8,  9, 12, 13, 16, 17, 20, 21,  2,  3,  4,  5,  8,  9, 10,
        11, 14, 15, 18, 19, 22, 23,  2,  3,  4,  5,  8,  9, 10, 11, 14, 15,
        18, 19, 22, 23,  6,  7,  8,  9, 16, 17, 20, 21,  6,  7,  8,  9, 16,
        17, 20, 21,  8,  9, 10, 11, 18, 19, 22, 23,  8,  9, 10, 11, 18, 19,
        22, 23])
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        self.assertEqual(discr.ndof(sd), sd.dim * sd.num_faces)

        class Dummy:
            pass

        self.assertRaises(
            ValueError,
            discr.ndof,
            Dummy(),
        )

        P = discr.proj_to_VRT0(sd)

        # fmt: off
        P_known_data = np.array(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )

        P_known_indices = np.array(
        [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
        8,  9,  9, 10, 10, 11, 11]
        )

        P_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        self.assertRaises(
            NotImplementedError,
            discr.proj_from_RT0,
            sd,
        )

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  0.5, -0.5, -0.5, -0.5,
         0.5, -0.5,  0.5,  0.5,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5,
        -0.5,  0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
        )

        D_known_indices = np.array(
        [0, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 2, 3, 3, 0, 0, 1, 1, 2, 0,
        2, 0, 3, 1, 3, 1, 2, 2, 3, 3]
        )

        D_known_indptr = np.array(
        [ 0,  1,  2,  4,  6,  7,  8,  9, 10, 12, 14, 15, 16, 17, 18, 19, 20,
        22, 24, 26, 28, 29, 30, 31, 32]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        self.assertRaises(
            NotImplementedError,
            discr.eval_at_cell_centers,
            sd,
        )

        self.assertRaises(
            NotImplementedError,
            discr.interpolate,
            sd,
            lambda x: x,
        )

        b_faces = sd.tags["domain_boundary_faces"]

        def p_0(x):
            return x[0]

        bc_val_from_bool = -discr.assemble_nat_bc(sd, p_0, b_faces)
        bc_val = -discr.assemble_nat_bc(sd, p_0, b_faces.nonzero()[0])

        self.assertTrue(np.allclose(bc_val, bc_val_from_bool))

        # fmt: off
        bc_val_known = np.array(
        [-0.        , -0.        , -0.        , -0.        , -1.        ,
        -1.        , -0.        , -0.        , -0.        , -0.        ,
        -1.        , -1.        ,  0.33333333,  0.16666667,  0.83333333,
         0.66666667, -0.        , -0.        , -0.        , -0.        ,
        -0.33333333, -0.16666667, -0.83333333, -0.66666667]
        )
        # fmt: on

        self.assertTrue(np.allclose(bc_val, bc_val_known))

        self.assertTrue(discr.get_range_discr_class(sd.dim) is pg.PwConstants)

        self.assertRaises(
            NotImplementedError,
            discr.assemble_lumped_matrix,
            sd,
            {},
        )

if __name__ == "__main__":
    unittest.main()
