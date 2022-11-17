""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class NedelecTest(unittest.TestCase):
    def test_interpolation(self):
        N, dim = 4, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        def r_constant(x):
            return np.array([1.0, 2.0, np.pi])

        for discr in [pg.Nedelec0("flow"), pg.Nedelec1("flow")]:
            interp_r = discr.interpolate(sd, r_constant)
            eval_r = discr.eval_at_cell_centers(sd) * interp_r
            eval_r = np.reshape(eval_r, (3, -1), order="F")

            known_r = np.array([r_constant(x) for x in sd.cell_centers.T]).T
            self.assertAlmostEqual(np.linalg.norm(eval_r - known_r), 0)


if __name__ == "__main__":
    unittest.main()
