""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VLagrange1Test(unittest.TestCase):
    def test_on_pentagon(self, N=2):
        dim = 2

        # nodes = np.array([[0, 3, 3, 3.0 / 2, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        # indptr = np.arange(0, 11, 2)
        # indices = np.roll(np.repeat(np.arange(5), 2), -1)
        # face_nodes = sps.csc_matrix((np.ones(10), indices, indptr))
        # cell_faces = sps.csc_matrix(np.ones((5, 1)))

        # sd = pg.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        # sd.compute_geometry()

        # sd = pp.CartGrid([2] * dim, [1, 1])
        # pg.convert_from_pp(sd)

        # sd = pg.OctGrid([10, 10])
        sd = pg.VoronoiGrid(0.05, 250, 0)
        sd.compute_geometry()

        discr = pg.VLagrange1("flow")

        mass = discr.assemble_mass_matrix(sd)
        stiff = discr.assemble_stiff_matrix(sd)

        eval = discr.eval_at_cell_centers(sd)
        rhs = sd.cell_volumes * eval

        ls = pg.LinearSystem(stiff, rhs)
        ls.flag_ess_bc(sd.tags["domain_boundary_nodes"], np.zeros(discr.ndof(sd)))

        u = ls.solve()

        u_cc = eval * u

        exp = pp.Exporter(sd, "voronoi", binary=False)
        exp.write_vtu(("u", u_cc))
        print("")

        # discr_p0 = pg.PwConstants("flow")
        # mass_p0 = discr_p0.assemble_mass_matrix(sd, None)

        # mass_vbdm1 = discr_vbdm1.assemble_lumped_matrix(sd, None)
        # div = mass_p0 * discr_vbdm1.assemble_diff_matrix(sd)

        # spp = sps.bmat([[mass_vbdm1, -div.T], [div, None]], "csc")
        # rhs = np.zeros(spp.shape[0])
        # rhs[-sd.num_cells :] = np.ones(sd.num_cells)

        # vx = sps.linalg.spsolve(spp, rhs)
        # vp = vx[-sd.num_cells :]

        # discr_bdm1 = pg.BDM1("flow")
        # mass_bdm1 = discr_bdm1.assemble_lumped_matrix(sd, None)
        # diff_bdm1 = discr_bdm1.assemble_diff_matrix(sd)

        # spp = sps.bmat([[mass_bdm1, -diff_bdm1.T],
        #                 [diff_bdm1, None]])
        # rhs = np.zeros(spp.shape[0])
        # rhs[-sd.num_cells:] = sd.cell_volumes

        # x = sps.linalg.spsolve(spp, rhs)
        # p = x[-sd.num_cells:]

        # proj = discr_p0.eval_at_cell_centers(sd)
        # print(vp @ mass_p0 @ vp, (proj * vp).max())
        # discr_p0.interpolate()

        # save = pp.Exporter(sd, "solref")
        # save.write_vtu([("p", proj * vp)])

    # def test_conv(self):
    #     for N in np.power(2.0, np.arange(7)):
    #         self.test_on_Octgrid(N.astype(int))


if __name__ == "__main__":
    VLagrange1Test().test_on_pentagon()
