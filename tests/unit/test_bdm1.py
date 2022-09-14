""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class BDM1Test(unittest.TestCase):
    def test0(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_matrix(sd, None)

        discr_rt0 = pp.RT0("flow")
        data = pp.initialize_default_data(sd, {}, "flow", {})
        discr_rt0.discretize(sd, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_rt0.mass_matrix_key]

        E = sps.bmat([[sps.eye(sd.num_faces)]*2])

        check = E * mass_bdm1 * E.T - mass_rt0

        import pdb; pdb.set_trace()

    def test1(self):
        N, dim = 2, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_matrix(sd, None)

        discr_rt0 = pp.RT0("flow")
        data = pp.initialize_default_data(sd, {}, "flow", {})
        discr_rt0.discretize(sd, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_rt0.mass_matrix_key]

        E = sps.bmat([[sps.eye(sd.num_faces)]*3])

        check = E * mass_bdm1 * E.T - mass_rt0

        import pdb; pdb.set_trace()


if __name__ == "__main__":
    BDM1Test().test1()
    #unittest.main()

