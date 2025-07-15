import unittest

import numpy as np
import pygeon as pg


class ProjectionsTest(unittest.TestCase):
    def test_lagrange1(self):
        for dim in [1, 2, 3]:
            for degree in [1, 2]:
                sd = pg.unit_grid(0.5, dim, as_mdg=False)
                sd.compute_geometry()

                l1 = pg.Lagrange1()
                l1_mass = l1.assemble_mass_matrix(sd)

                pi = pg.proj_to_PwPolynomials(l1, sd, degree)

                poly = pg.get_PwPolynomials(degree, pg.SCALAR)()
                poly_mass = poly.assemble_mass_matrix(sd)

                diff = pi.T @ poly_mass @ pi - l1_mass

                self.assertTrue(np.allclose(diff.data, 0))

    def test_rt0(self):
        for dim in [1, 2, 3]:
            for degree in [1, 2]:
                sd = pg.unit_grid(0.5, dim, as_mdg=False)
                sd.compute_geometry()

                rt0 = pg.RT0()
                rt0_mass = rt0.assemble_mass_matrix(sd)

                pi = pg.proj_to_PwPolynomials(rt0, sd, degree)

                poly = pg.get_PwPolynomials(degree, pg.VECTOR)()
                poly_mass = poly.assemble_mass_matrix(sd)

                diff = pi.T @ poly_mass @ pi - rt0_mass

                self.assertTrue(np.allclose(diff.data, 0))


if __name__ == "__main__":
    unittest.main()
