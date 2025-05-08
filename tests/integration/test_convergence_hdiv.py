import unittest
import numpy as np
import scipy.sparse as sps

import pygeon as pg

"""
Module contains tests to validate the consistency between H(div) discretizations.
"""


class HdivConvergenceTest(unittest.TestCase):
    def test_rt1_2d_lumped(self):
        # Provide the solution
        def q_func(x):
            return np.array([x[1] * x[0], 2 * x[0], 0])

        def p_func(x):
            return x[0]

        def g_func(x):
            return q_func(x) + np.array([1, 0, 0])

        def div_func(x):
            return x[1]

        discr_q = pg.RT1()
        discr_p = pg.PwLinears()
        h_list = 2.0 ** (-np.arange(2, 5))

        error_p = np.zeros_like(h_list)
        error_q = np.zeros_like(h_list)

        for ind, h in enumerate(h_list):
            sd = pg.unit_grid(2, h, as_mdg=False)
            sd.compute_geometry()

            # assemble the saddle point problem
            face_mass = discr_q.assemble_lumped_matrix(sd)
            cell_mass = discr_p.assemble_mass_matrix(sd, None)
            div = cell_mass @ discr_q.assemble_diff_matrix(sd)

            spp = sps.bmat([[face_mass, -div.T], [div, None]], format="csc")

            # set the boundary conditions
            b_faces = sd.tags["domain_boundary_faces"]
            bc_val = -discr_q.assemble_nat_bc(sd, p_func, b_faces)

            rhs = np.zeros(spp.shape[0])
            rhs[bc_val.size :] += discr_p.source_term(sd, div_func)
            rhs[: bc_val.size] += bc_val
            rhs[: bc_val.size] += discr_q.source_term(sd, g_func)

            # solve the problem
            ls = pg.LinearSystem(spp, rhs)
            x = ls.solve()

            q = x[: bc_val.size]
            p = x[-discr_p.ndof(sd) :]

            error_p[ind] = discr_p.error_l2(sd, p, p_func)
            error_q[ind] = discr_q.error_l2(sd, q, q_func)

        rates_p = np.log(error_p[1:] / error_p[:-1]) / np.log(h_list[1:] / h_list[:-1])
        rates_q = np.log(error_q[1:] / error_q[:-1]) / np.log(h_list[1:] / h_list[:-1])

        self.assertTrue(np.all(rates_p >= 1.9))
        self.assertTrue(np.all(rates_q >= 1.9))


if __name__ == "__main__":
    unittest.main()
