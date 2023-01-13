""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VBDM1Test(unittest.TestCase):
    def test0(self):
        N, dim = 1, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.VBDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_lumped_matrix(sd, None)

        discr_rt0 = pp.RT0("flow")
        data = pp.initialize_default_data(sd, {}, "flow", {})
        discr_rt0.discretize(sd, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_rt0.mass_matrix_key]

        E = discr_bdm1.proj_from_RT0(sd)

        check = E.T * mass_bdm1 * E - mass_rt0

        self.assertEqual(check.nnz, 0)

if __name__ == "__main__":
    VBDM1Test().test0()


