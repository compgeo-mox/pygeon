"""Module contains virtual Lagrangean tests."""

import unittest

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VLagrange1Test(unittest.TestCase):
    def test_on_pentagon(self):
        nodes = np.array([[0, 3, 3, 3.0 / 2.0, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        indptr = np.arange(0, 11, 2)
        indices = np.roll(np.repeat(np.arange(5), 2), -1)
        face_nodes = sps.csc_array((np.ones(10), indices, indptr))
        cell_faces = sps.csc_array(np.ones((5, 1)))

        sd = pg.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        sd.compute_geometry()

        discr = pg.VLagrange1()
        diam = sd.cell_diameters()[0]
        loc_nodes = np.arange(5)

        # Test the three matrices from Hitchhikers sec 4.2
        B = discr.assemble_loc_L2proj_rhs(sd, 0, diam, loc_nodes)
        B_known = (
            np.array(
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [-8.0, 4.0, 8.0, 4.0, -8.0],
                    [-6.0, -6.0, 3.0, 6.0, 3.0],
                ]
            )
            / 20
        )
        self.assertTrue(np.allclose(B, B_known))

        D = discr.assemble_loc_dofs_of_monomials(sd, 0, diam, loc_nodes)
        D_known = (
            np.array(
                [
                    [1470.0, -399.0, -532.0],
                    [1470.0, 483.0, -532.0],
                    [1470.0, 483.0, 56.0],
                    [1470.0, 42.0, 644.0],
                    [1470.0, -399.0, 644.0],
                ]
            )
            / 1470
        )
        self.assertTrue(np.allclose(D, D_known))

        G = discr.assemble_loc_L2proj_lhs(sd, 0, diam, loc_nodes)
        G_known = (
            np.array([[1050.0, 30.0, 40.0], [0.0, 441.0, 0.0], [0.0, 0.0, 441.0]])
            / 1050
        )
        self.assertTrue(np.allclose(G, G_known))
        self.assertEqual(discr.ndof(sd), sd.num_nodes)

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.]
        )

        D_known_indices = np.array(
        [0, 4, 0, 1, 1, 2, 2, 3, 3, 4]
        )

        D_known_indptr = np.array(
        [ 0,  2,  4,  6,  8, 10]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2]
        )

        P_known_indices = np.array(
        [0, 0, 0, 0, 0]
        )

        P_known_indptr = np.array(
        [0, 1, 2, 3, 4, 5]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        fun = lambda x: x[0] + x[1]
        vals = discr.interpolate(sd, fun)

        # fmt: off
        vals_known = np.array(
        [0. , 3. , 5. , 5.5, 4. ]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))

        b_nodes = sd.tags["domain_boundary_nodes"]
        vals = discr.assemble_nat_bc(sd, lambda x: np.ones(1), b_nodes)

        # fmt: off
        vals_known = np.array(
        [3.5 , 2.5 , 2.25, 2.  , 2.75]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            sd.dim,
        )

    def test_on_oct_grid(self):
        sd = pg.OctagonGrid([1] * 2)
        sd.compute_geometry()

        discr = pg.VLagrange1()

        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [ 0.54954042, -0.22879652,  0.04757809,  0.10481714, -0.22522209,
       -0.09060922, -0.09060922,  0.04757809,  0.00357443, -0.22879652,
        0.54954042,  0.10481714,  0.04757809, -0.09060922, -0.22522209,
        0.04757809, -0.09060922,  0.00357443,  0.04757809,  0.10481714,
        0.54954042, -0.22879652, -0.09060922,  0.04757809, -0.22522209,
       -0.09060922,  0.00357443,  0.10481714,  0.04757809, -0.22879652,
        0.54954042,  0.04757809, -0.09060922, -0.09060922, -0.22522209,
        0.00357443, -0.22522209, -0.09060922, -0.09060922,  0.04757809,
        0.54954042,  0.04757809, -0.22879652,  0.10481714,  0.00357443,
       -0.09060922, -0.22522209,  0.04757809, -0.09060922,  0.04757809,
        0.54954042,  0.10481714, -0.22879652,  0.00357443, -0.09060922,
        0.04757809, -0.22522209, -0.09060922, -0.22879652,  0.10481714,
        0.54954042,  0.04757809,  0.00357443,  0.04757809, -0.09060922,
       -0.09060922, -0.22522209,  0.10481714, -0.22879652,  0.04757809,
        0.54954042,  0.00357443,  0.00357443,  0.00357443,  0.00714887,
        0.00357443,  0.00357443,  0.00714887,  0.00357443,  0.00357443,
        0.00714887,  0.00357443,  0.00357443,  0.00714887])

        M_known_indptr = np.array(
        [0,  9, 18, 27, 36, 45, 54, 63, 72, 75, 78, 81, 84])

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,  6,  7,
        9,  0,  1,  2,  3,  4,  5,  6,  7, 10,  0,  1,  2,  3,  4,  5,  6,
        7, 11,  0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,
        6,  7,  9,  0,  1,  2,  3,  4,  5,  6,  7, 10,  0,  1,  2,  3,  4,
        5,  6,  7, 11,  0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
        # fmt: on

        M.sum_duplicates()
        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        A = discr.assemble_stiff_matrix(sd)

        # fmt: off
        A_known_data = np.array(
        [ 1.3017767, -0.1767767, -0.0732233, -0.0517767, -0.1767767,
       -0.125    , -0.125    , -0.0732233, -0.5      , -0.1767767,
        1.3017767, -0.0517767, -0.0732233, -0.125    , -0.1767767,
       -0.0732233, -0.125    , -0.5      , -0.0732233, -0.0517767,
        1.3017767, -0.1767767, -0.125    , -0.0732233, -0.1767767,
       -0.125    , -0.5      , -0.0517767, -0.0732233, -0.1767767,
        1.3017767, -0.0732233, -0.125    , -0.125    , -0.1767767,
       -0.5      , -0.1767767, -0.125    , -0.125    , -0.0732233,
        1.3017767, -0.0732233, -0.1767767, -0.0517767, -0.5      ,
       -0.125    , -0.1767767, -0.0732233, -0.125    , -0.0732233,
        1.3017767, -0.0517767, -0.1767767, -0.5      , -0.125    ,
       -0.0732233, -0.1767767, -0.125    , -0.1767767, -0.0517767,
        1.3017767, -0.0732233, -0.5      , -0.0732233, -0.125    ,
       -0.125    , -0.1767767, -0.0517767, -0.1767767, -0.0732233,
        1.3017767, -0.5      , -0.5      , -0.5      ,  1.       ,
       -0.5      , -0.5      ,  1.       , -0.5      , -0.5      ,
        1.       , -0.5      , -0.5      ,  1.       ])

        A_known_indptr = np.array(
        [ 0,  9, 18, 27, 36, 45, 54, 63, 72, 75, 78, 81, 84])

        A_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,  6,  7,
        9,  0,  1,  2,  3,  4,  5,  6,  7, 10,  0,  1,  2,  3,  4,  5,  6,
        7, 11,  0,  1,  2,  3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,
        6,  7,  9,  0,  1,  2,  3,  4,  5,  6,  7, 10,  0,  1,  2,  3,  4,
        5,  6,  7, 11,  0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
        # fmt: on

        A.sum_duplicates()
        self.assertTrue(np.allclose(A.data, A_known_data))
        self.assertTrue(np.allclose(A.indptr, A_known_indptr))
        self.assertTrue(np.allclose(A.indices, A_known_indices))

        self.assertEqual(discr.ndof(sd), sd.num_nodes)

        D = discr.assemble_diff_matrix(sd)

        # fmt: off
        D_known_data = np.array(
        [-1., -1., -1.,  1., -1., -1., -1., -1., -1.,  1., -1., -1., -1.,
        1., -1., -1.,  1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.]
        )

        D_known_indices = np.array(
        [ 0,  4,  8,  0,  5, 10,  1,  6, 12,  1,  7, 14,  2,  4,  9,  3,  5,
        11,  2,  6, 13,  3,  7, 15,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        D_known_indptr = np.array(
        [ 0,  3,  6,  9, 12, 15, 18, 21, 24, 26, 28, 30, 32]
        )
        # fmt: on

        self.assertTrue(np.allclose(D.data, D_known_data))
        self.assertTrue(np.allclose(D.indptr, D_known_indptr))
        self.assertTrue(np.allclose(D.indices, D_known_indices))

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [0.125     , 0.33333333, 0.125     , 0.33333333, 0.125     ,
        0.33333333, 0.125     , 0.33333333, 0.125     , 0.33333333,
        0.125     , 0.33333333, 0.125     , 0.33333333, 0.125     ,
        0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333]
        )

        P_known_indices = np.array(
        [0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4]
        )

        P_known_indptr = np.array(
        [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 17, 18, 19, 20]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

        fun = lambda x: x[0] + x[1]
        vals = discr.interpolate(sd, fun)

        # fmt: off
        vals_known = np.array(
        [0.29289322, 0.70710678, 1.29289322, 1.70710678, 0.29289322,
        1.29289322, 0.70710678, 1.70710678, 0.        , 1.        ,
        1.        , 2.        ]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))

        b_nodes = sd.tags["domain_boundary_nodes"]
        vals = discr.assemble_nat_bc(sd, lambda x: np.ones(1), b_nodes)

        # fmt: off
        vals_known = np.array(
        [0.56066017, 0.56066017, 0.41421356, 0.41421356, 0.56066017,
        0.56066017, 0.41421356, 0.41421356, 0.29289322, 0.29289322,
        0.        , 0.        ]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            sd.dim,
        )


if __name__ == "__main__":
    unittest.main()
