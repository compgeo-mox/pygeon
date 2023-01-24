""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class VLagrange1Test(unittest.TestCase):
    def test_on_pentagon(self):

        nodes = np.array([[0, 3, 3, 3.0 / 2.0, 0], [0, 0, 2, 4, 4], np.zeros(5)])
        indptr = np.arange(0, 11, 2)
        indices = np.roll(np.repeat(np.arange(5), 2), -1)
        face_nodes = sps.csc_matrix((np.ones(10), indices, indptr))
        cell_faces = sps.csc_matrix(np.ones((5, 1)))

        sd = pg.Grid(2, nodes, face_nodes, cell_faces, "pentagon")
        sd.compute_geometry()

        discr = pg.VLagrange1("flow")
        diam = sd.cell_diameters()[0]
        loc_nodes = np.arange(5)

        # Test the three matrices from Hitchhikers sec 4.2
        B = discr.assemble_loc_L2proj_rhs(sd, 0, diam, loc_nodes)
        B_known = (
            np.array(
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [-8.0, 4.0, 8.0, 4.0, -8.0],
                    [-6.0, -6.0, 3.0, 6.0, 3.0],
                ]
            )
            / 20
        )
        self.assertTrue(np.allclose(B, B_known))

        D = discr.assemble_loc_dofs_of_monomials(sd, 0, diam, loc_nodes)
        D_known = (
            np.array(
                [
                    [1470.0, -399.0, -532.0],
                    [1470.0, 483.0, -532.0],
                    [1470.0, 483.0, 56.0],
                    [1470.0, 42.0, 644.0],
                    [1470.0, -399.0, 644.0],
                ]
            )
            / 1470
        )
        self.assertTrue(np.allclose(D, D_known))

        G = discr.assemble_loc_L2proj_lhs(sd, 0, diam, loc_nodes)
        G_known = (
            np.array([[1050.0, 30.0, 40.0], [0.0, 441.0, 0.0], [0.0, 0.0, 441.0]])
            / 1050
        )
        self.assertTrue(np.allclose(G, G_known))

    def test_on_Cart_grid(self):
        dim = 2
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


if __name__ == "__main__":
    unittest.main()
