import unittest
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class PwLinearsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")
        assert discr.ndof(sd) == sd.num_cells * (dim + 1)

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")
        M = discr.assemble_mass_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.02083333, 0.01041667, 0.01041667, 0.01041667, 0.02083333,
        0.01041667, 0.01041667, 0.01041667, 0.02083333, 0.02083333,
        0.01041667, 0.01041667, 0.01041667, 0.02083333, 0.01041667,
        0.01041667, 0.01041667, 0.02083333, 0.02083333, 0.01041667,
        0.01041667, 0.01041667, 0.02083333, 0.01041667, 0.01041667,
        0.01041667, 0.02083333, 0.02083333, 0.01041667, 0.01041667,
        0.01041667, 0.02083333, 0.01041667, 0.01041667, 0.01041667,
        0.02083333, 0.02083333, 0.01041667, 0.01041667, 0.01041667,
        0.02083333, 0.01041667, 0.01041667, 0.01041667, 0.02083333,
        0.02083333, 0.01041667, 0.01041667, 0.01041667, 0.02083333,
        0.01041667, 0.01041667, 0.01041667, 0.02083333, 0.02083333,
        0.01041667, 0.01041667, 0.01041667, 0.02083333, 0.01041667,
        0.01041667, 0.01041667, 0.02083333, 0.02083333, 0.01041667,
        0.01041667, 0.01041667, 0.02083333, 0.01041667, 0.01041667,
        0.01041667, 0.02083333]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  3,  4,
         5,  6,  7,  8,  6,  7,  8,  6,  7,  8,  9, 10, 11,  9, 10, 11,  9,
        10, 11, 12, 13, 14, 12, 13, 14, 12, 13, 14, 15, 16, 17, 15, 16, 17,
        15, 16, 17, 18, 19, 20, 18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22,
        23, 21, 22, 23]
        )

        M_known_indptr = np.array(
        [0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
        51, 54, 57, 60, 63, 66, 69, 72]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_lumped_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        M = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
        0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
        0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
        0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667,
        0.04166667, 0.04166667, 0.04166667, 0.04166667]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        self.assertRaises(
            NotImplementedError,
            discr.assemble_diff_matrix,
            sd,
        )

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        self.assertRaises(
            NotImplementedError,
            discr.assemble_diff_matrix,
            sd,
        )

    def test_interpolate(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        self.assertRaises(
            NotImplementedError, discr.interpolate, sd, lambda x: np.sin(x)
        )

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        self.assertRaises(NotImplementedError, discr.eval_at_cell_centers, sd)

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces

        self.assertRaises(NotImplementedError, discr.assemble_nat_bc, sd, func, b_faces)

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears("P1b")

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )


if __name__ == "__main__":
    unittest.main()
