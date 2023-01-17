""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VBDM1Test(unittest.TestCase):
    def test_on_Octgrid(self, N=2):
        dim = 2
        sd = pg.OctGrid([N] * dim, [1, 1])
        pg.convert_from_pp(sd)
        sd.nodes /= 40

        sd.compute_geometry()

        discr_vbdm1 = pg.VBDM1("flow")

        discr_p0 = pg.PwConstants("flow")
        mass_p0 = discr_p0.assemble_mass_matrix(sd, None)

        mass_vbdm1 = discr_vbdm1.assemble_lumped_matrix(sd, None)
        div = mass_p0 * discr_vbdm1.assemble_diff_matrix(sd)

        spp = sps.bmat([[mass_vbdm1, -div.T], [div, None]], "csc")
        rhs = np.zeros(spp.shape[0])
        rhs[-sd.num_cells :] = np.ones(sd.num_cells)

        vx = sps.linalg.spsolve(spp, rhs)
        vp = vx[-sd.num_cells :]

        # discr_bdm1 = pg.BDM1("flow")
        # mass_bdm1 = discr_bdm1.assemble_lumped_matrix(sd, None)
        # diff_bdm1 = discr_bdm1.assemble_diff_matrix(sd)

        # spp = sps.bmat([[mass_bdm1, -diff_bdm1.T],
        #                 [diff_bdm1, None]])
        # rhs = np.zeros(spp.shape[0])
        # rhs[-sd.num_cells:] = sd.cell_volumes

        # x = sps.linalg.spsolve(spp, rhs)
        # p = x[-sd.num_cells:]

        proj = discr_p0.eval_at_cell_centers(sd)
        print(vp @ mass_p0 @ vp, (proj * vp).max())
        discr_p0.interpolate()

        save = pp.Exporter(sd, "solref")
        save.write_vtu([("p", proj * vp)])

    def test_conv(self):
        for N in np.power(2.0, np.arange(7)):
            self.test_on_Octgrid(N.astype(int))


if __name__ == "__main__":
    VBDM1Test().test_on_Octgrid()
