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

        discr = pg.PwLinears()
        assert discr.ndof(sd) == 24

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()
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

        discr = pg.PwLinears()

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

        discr = pg.PwLinears()

        B = discr.assemble_diff_matrix(sd)
        self.assertTrue(B.nnz == 0)

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()

        stiff = discr.assemble_stiff_matrix(sd)
        self.assertTrue(stiff.nnz == 0)

    def test_interpolate(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()

        interp = discr.interpolate(sd, lambda x: x[0])
        P = discr.eval_at_cell_centers(sd)
        known = sd.cell_centers[0]

        self.assertTrue(np.allclose(P @ interp, known))

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()

        known_func = np.ones(discr.ndof(sd))

        P = discr.eval_at_cell_centers(sd)

        self.assertTrue(np.allclose(P @ known_func, np.ones(sd.num_cells)))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()

        func = lambda x: np.sin(x[0])
        b_faces = np.array([0, 1, 3])

        b = discr.assemble_nat_bc(sd, func, b_faces)
        self.assertTrue(np.all(b == 0))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        sd.compute_geometry()

        discr = pg.PwLinears()

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )


if __name__ == "__main__":
    unittest.main()
