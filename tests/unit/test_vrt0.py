import unittest
import numpy as np

import porepy as pp
import pygeon as pg  # type: ignore[import-untyped]


class VRT0Test(unittest.TestCase):
    def test0(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.VRT0()
        self.assertEqual(discr.ndof(sd), sd.num_faces)

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [ 0.61111111, -0.27777778,  0.11111111,  0.61111111, -0.27777778,
         0.11111111, -0.27777778, -0.27777778,  0.88888889, -0.27777778,
        -0.27777778,  0.61111111, -0.27777778,  0.11111111,  0.11111111,
        -0.27777778,  1.22222222, -0.27777778,  0.11111111, -0.27777778,
        -0.27777778,  0.88888889, -0.27777778, -0.27777778,  0.11111111,
        -0.27777778,  0.61111111,  0.11111111, -0.27777778,  1.22222222,
        -0.27777778,  0.11111111,  0.61111111, -0.27777778,  0.11111111,
        -0.27777778, -0.27777778,  0.88888889, -0.27777778, -0.27777778,
         0.11111111, -0.27777778,  1.22222222, -0.27777778,  0.11111111,
         0.11111111, -0.27777778,  1.22222222, -0.27777778,  0.11111111,
        -0.27777778, -0.27777778,  0.88888889, -0.27777778, -0.27777778,
         0.11111111, -0.27777778,  0.61111111,  0.11111111, -0.27777778,
         0.61111111,  0.11111111, -0.27777778,  0.61111111]
        )

        M_known_indices = np.array(
        [ 0,  2,  4,  1,  2,  7,  0,  1,  2,  4,  7,  3,  5,  6,  0,  2,  4,
         5, 10,  3,  4,  5,  6, 10,  3,  5,  6,  1,  2,  7,  9, 11,  8,  9,
        14,  7,  8,  9, 11, 14,  4,  5, 10, 12, 13,  7,  9, 11, 12, 15, 10,
        11, 12, 13, 15, 10, 12, 13,  8,  9, 14, 11, 12, 15]
        )

        M_known_indptr = np.array(
        [ 0,  3,  6, 11, 14, 19, 24, 27, 32, 35, 40, 45, 50, 55, 58, 61, 64]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        Ml = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.372678  , 0.372678  , 0.33333333, 0.372678  , 0.74535599,
        0.33333333, 0.372678  , 0.74535599, 0.372678  , 0.33333333,
        0.74535599, 0.74535599, 0.33333333, 0.372678  , 0.372678  ,
        0.372678  ]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        )
        # fmt: on

        self.assertTrue(np.allclose(Ml.data, M_known_data))
        self.assertTrue(np.allclose(Ml.indptr, M_known_indptr))
        self.assertTrue(np.allclose(Ml.indices, M_known_indices))

        fun = lambda x: x[0] + x[1]
        faces = sd.tags["domain_boundary_faces"]

        vals = discr.assemble_nat_bc(sd, fun, faces.nonzero()[0])
        vals_from_bool = discr.assemble_nat_bc(sd, fun, faces)

        # fmt: off
        vals_known = np.array(
        [ 0.25, -0.25,  0.  ,  0.75,  0.  ,  0.  ,  1.25,  0.  , -0.75,
        0.  ,  0.  ,  0.  ,  0.  ,  1.75, -1.25, -1.75]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))
        self.assertTrue(np.allclose(vals_from_bool, vals_known))

        interp_fun = lambda x: np.array([x[0] + x[1], x[1], x[2] + x[1]])
        interp_vals = discr.interpolate(sd, interp_fun)

        # fmt: off
        interp_vals_known = np.array(
        [ 0.   ,  0.125,  0.125,  0.   ,  0.375,  0.375,  0.625, -0.25 ,
        0.375,  0.125, -0.25 ,  0.625,  0.375,  0.875, -0.5  , -0.5  ]
        )
        # fmt: on

        self.assertTrue(np.allclose(interp_vals, interp_vals_known))

        self.assertTrue(discr.get_range_discr_class(sd.dim) is pg.PwConstants)

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [ 1., -1., -1.,  1.,  1.,  1., -1., -1.,  1.,  1., -1.,  1., -1.,
        -1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1., -1., -1.]
        )

        D_known_indices = np.array(
        [0, 1, 0, 1, 2, 0, 3, 2, 3, 2, 1, 4, 5, 4, 5, 3, 6, 4, 7, 6, 7, 6,
        5, 7]
        )

        D_known_indptr = np.array(
        [ 0,  1,  2,  4,  5,  7,  9, 10, 12, 13, 15, 17, 19, 21, 22, 23, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [-0.66666667, -1.33333333,  0.        ,  1.33333333,  0.66666667,
         0.        ,  0.66666667, -0.66666667,  0.        ,  0.66666667,
        -0.66666667,  0.        , -0.66666667, -1.33333333,  0.        ,
         1.33333333,  0.66666667,  0.        ,  1.33333333,  0.66666667,
         0.        ,  0.66666667, -0.66666667,  0.        ,  0.66666667,
        -0.66666667,  0.        ,  1.33333333,  0.66666667,  0.        ,
        -0.66666667, -1.33333333,  0.        , -0.66666667, -1.33333333,
         0.        ,  1.33333333,  0.66666667,  0.        ,  0.66666667,
        -0.66666667,  0.        ,  0.66666667, -0.66666667,  0.        ,
        -0.66666667, -1.33333333,  0.        , -0.66666667, -1.33333333,
         0.        ,  1.33333333,  0.66666667,  0.        ,  1.33333333,
         0.66666667,  0.        ,  0.66666667, -0.66666667,  0.        ,
         0.66666667, -0.66666667,  0.        ,  1.33333333,  0.66666667,
         0.        , -0.66666667, -1.33333333,  0.        , -0.66666667,
        -1.33333333,  0.        ]
        )

        P_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,
         2,  9, 10, 11,  6,  7,  8,  9, 10, 11,  6,  7,  8,  3,  4,  5, 12,
        13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17,  9, 10, 11, 18, 19, 20,
        12, 13, 14, 21, 22, 23, 18, 19, 20, 21, 22, 23, 18, 19, 20, 15, 16,
        17, 21, 22, 23]
        )

        P_known_indptr = np.array(
        [ 0,  3,  6, 12, 15, 21, 27, 30, 36, 39, 45, 51, 57, 63, 66, 69, 72]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))


if __name__ == "__main__":
    unittest.main()
