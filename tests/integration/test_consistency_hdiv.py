import unittest
import numpy as np

import pygeon as pg

"""
Module contains tests to validate the consistency between H(div) discretizations.
"""


class HdivMatricesTest(unittest.TestCase):
    def test_mass(self):
        for dim in [2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            rt0 = pg.RT0()
            M_rt0 = rt0.assemble_mass_matrix(sd)

            bdm1 = pg.BDM1()
            M_bdm1 = bdm1.assemble_mass_matrix(sd)
            P = bdm1.proj_from_RT0(sd)

            difference = M_rt0 - P.T @ M_bdm1 @ P

            self.assertTrue(np.allclose(difference.data, 0))

    def test_interp_eval_constants(self):
        for dim in [2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            f = lambda x: np.array([2, 3, -1])

            rt0 = pg.RT0()
            P = rt0.eval_at_cell_centers(sd)
            f_rt0 = P @ rt0.interpolate(sd, f)

            bdm1 = pg.BDM1()
            P = bdm1.eval_at_cell_centers(sd)
            f_bdm1 = P @ bdm1.interpolate(sd, f)

            self.assertTrue(np.allclose(f_rt0, f_bdm1))


if __name__ == "__main__":
    unittest.main()
