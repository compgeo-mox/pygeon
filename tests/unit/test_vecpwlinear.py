import unittest
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class VecPwLinearsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")
        self.assertTrue(discr.ndof(sd) == sd.num_cells * dim * (dim + 1))

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")
        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.08333333, 0.04166667, 0.04166667, 0.04166667, 0.08333333,
        0.04166667, 0.04166667, 0.04166667, 0.08333333, 0.08333333,
        0.04166667, 0.04166667, 0.04166667, 0.08333333, 0.04166667,
        0.04166667, 0.04166667, 0.08333333, 0.08333333, 0.04166667,
        0.04166667, 0.04166667, 0.08333333, 0.04166667, 0.04166667,
        0.04166667, 0.08333333, 0.08333333, 0.04166667, 0.04166667,
        0.04166667, 0.08333333, 0.04166667, 0.04166667, 0.04166667,
        0.08333333]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  3,  4,
         5,  6,  7,  8,  6,  7,  8,  6,  7,  8,  9, 10, 11,  9, 10, 11,  9,
        10, 11]
        )

        M_known_indptr = np.array(
        [ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_lumped_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        M = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 6

        M_known_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        M_known_indptr = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        self.assertRaises(NotImplementedError, discr.assemble_diff_matrix, sd)

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        self.assertRaises(NotImplementedError, discr.assemble_stiff_matrix, sd)

    def test_interpolate(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        func = lambda x: np.sin(x)  # Example function
        self.assertRaises(NotImplementedError, discr.interpolate, sd, func)

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.ones(discr.ndof(sd)) / (sd.dim + 1)

        P_known_indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

        P_known_indptr = np.arange(discr.ndof(sd)+1)
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces
        vals = discr.assemble_nat_bc(sd, func, b_faces)

        vals_known = np.zeros(discr.ndof(sd))

        self.assertTrue(np.allclose(vals, vals_known))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_error_l2(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([1] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.VecPwLinears("P1")
        # fmt: off
        num_sol = np.empty(0)
        # fmt: on
        ana_sol = lambda x: np.sin(x)

        self.assertRaises(NotImplementedError, discr.error_l2, sd, num_sol, ana_sol)


if __name__ == "__main__":
    unittest.main()
