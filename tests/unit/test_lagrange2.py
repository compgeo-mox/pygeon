""" Module contains a unit test for the Lagrangean P2 discretization.
"""

import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Lagrange2Test(unittest.TestCase):
    def test_3d(self):  # for dim in range(1, 4):
        dim = 3
        mdg = pg.unit_grid(dim, 0.5)
        sd = mdg.subdomains()[0]

        # sd = pp.StructuredTetrahedralGrid([10] * 3)
        # sd = pp.CartGrid(100, [1])
        # pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.Lagrange2()
        A = discr.assemble_stiffness_matrix(sd, None)

        source = np.ones(sd.num_nodes + sd.num_ridges)
        M = discr.assemble_mass_matrix(sd, None)
        f = M @ source

        ess_bc = np.hstack(
            (sd.tags["domain_boundary_nodes"], sd.tags["domain_boundary_ridges"])
        )
        ess_vals = np.zeros_like(ess_bc, dtype=float)

        ridge_centers = sd.nodes @ np.abs(sd.ridge_peaks) / 2
        x = np.hstack((sd.nodes, ridge_centers))
        true_sol = np.sum(x * (1 - x), axis=0) / (2 * dim)

        ess_vals[ess_bc] = true_sol[ess_bc]

        LS = pg.LinearSystem(A, f)
        LS.flag_ess_bc(ess_bc, ess_vals)

        u = LS.solve()

        self.assertTrue(np.allclose(u, true_sol))

        P = discr.eval_at_cell_centers(sd)

        pass


if __name__ == "__main__":
    Lagrange2Test().test_3d()
