import unittest
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class VecPwConstantsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")
        self.assertTrue(discr.ndof(sd) == sd.num_cells * dim)

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")
        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_lumped_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")

        M = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")
        D = discr.assemble_diff_matrix(sd).todense()
        D_known = sps.csc_array((0, discr.ndof(sd))).todense()

        self.assertTrue(np.allclose(D, D_known))

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")

        self.assertRaises(NotImplementedError, discr.assemble_stiff_matrix, sd)

    def test_interpolate(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")

        func = lambda x: np.sin(x)  # Example function
        vals = discr.interpolate(sd, func)

        # fmt: off
        vals_known = np.array(
        [0.04089934, 0.02073702, 0.09252211, 0.07729623, 0.04089934,
        0.02073702, 0.09252211, 0.07729623, 0.02073702, 0.04089934,
        0.02073702, 0.04089934, 0.07729623, 0.09252211, 0.07729623,
        0.09252211])
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")
        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.]
        )

        P_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        P_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces
        vals = discr.assemble_nat_bc(sd, func, b_faces)

        vals_known = np.zeros(discr.ndof(sd))

        self.assertTrue(np.allclose(vals, vals_known))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_error_l2(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwConstants("P0")
        # fmt: off
        num_sol = np.array(
        [0.04089934, 0.02073702, 0.09252211, 0.07729623, 0.04089934,
        0.02073702, 0.09252211, 0.07729623, 0.02073702, 0.04089934,
        0.02073702, 0.04089934, 0.07729623, 0.09252211, 0.07729623,
        0.09252211]
        )
        # fmt: on
        ana_sol = lambda x: np.sin(x)

        err = discr.error_l2(sd, num_sol, ana_sol, etype="standard")
        err_known = 7.951651025069553e-08

        self.assertTrue(np.allclose(err, err_known))

        err = discr.error_l2(sd, num_sol, ana_sol)
        err_known = 0.5254595621486393

        self.assertTrue(np.allclose(err, err_known))


if __name__ == "__main__":
    unittest.main()
