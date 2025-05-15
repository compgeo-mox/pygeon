"""Module contains vector Lagrangean2 fem tests."""

import unittest

import numpy as np

import pygeon as pg


class VecLagrange2Test(unittest.TestCase):
    def test_range_discr_class(self):
        l2 = pg.VecLagrange2()
        for dim in [1, 2, 3]:
            self.assertRaises(
                NotImplementedError,
                l2.get_range_discr_class,
                dim,
            )

    def test_proj_to_pwquadratics(self):
        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            l2 = pg.VecLagrange2()
            proj_l2 = l2.proj_to_pwQuadratics(sd)
            mass_l2 = l2.assemble_mass_matrix(sd)

            p2 = pg.VecPwQuadratics()
            mass_p1 = p2.assemble_mass_matrix(sd)

            diff = proj_l2.T @ mass_p1 @ proj_l2 - mass_l2

            self.assertTrue(np.allclose(diff.data, 0.0))


if __name__ == "__main__":
    unittest.main()
