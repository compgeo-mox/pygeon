import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class RT1Test(unittest.TestCase):

    def test_linear_distribution_cart_1D(self):
        N, dim = 5, 1
        sd = pp.CartGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.linear_distribution_test(sd)

    def test_linear_distribution_struct_2D(self):
        N, dim = 5, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.linear_distribution_test(sd)

    def test_linear_distribution_struct_3D(self):
        N, dim = 3, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        self.linear_distribution_test(sd)

    def test_linear_distribution_unstr_2D(self):
        N, dim = 5, 2
        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        sd.compute_geometry()

        self.linear_distribution_test(sd)

    def test_linear_distribution_unstr_3D(self):
        N, dim = 3, 3
        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        sd.compute_geometry()

        self.linear_distribution_test(sd)

    def linear_distribution_test(self, sd):
        discr_q = pg.RT1()
        discr_p = pg.PwLinears()

        # Provide the solution
        def q_func(x):
            return np.array([-1, 2, 1])

        def p_func(x):
            return -x @ q_func(x)

        # assemble the saddle point problem
        face_mass = discr_q.assemble_mass_matrix(sd)
        cell_mass = discr_p.assemble_mass_matrix(sd, None)
        div = cell_mass @ discr_q.assemble_diff_matrix(sd)

        spp = sps.bmat([[face_mass, -div.T], [div, None]], format="csc")

        # set the boundary conditions
        b_faces = sd.tags["domain_boundary_faces"]
        bc_val = -discr_q.assemble_nat_bc(sd, p_func, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[: bc_val.size] += bc_val

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-discr_p.ndof(sd) :]

        known_q = discr_q.interpolate(sd, q_func)
        known_p = discr_p.interpolate(sd, p_func)

        self.assertTrue(np.allclose(p, known_p))
        self.assertTrue(np.allclose(q, known_q))

    def test_interpolation_and_evaluation(self):
        N, dim = 3, 3
        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        sd.compute_geometry()

        def q_func(x):
            return np.array([-x[1], 2 * x[0], x[2]])

        discr = pg.RT1()
        Pi = discr.eval_at_cell_centers(sd)
        interp = discr.interpolate(sd, q_func)

        q_at_cc = (Pi @ interp).reshape((3, -1))
        q_known_at_cc = np.array([q_func(x) for x in sd.cell_centers.T]).T

        self.assertTrue(np.allclose(q_at_cc, q_known_at_cc))

    def test_norm_of_known_function(self):
        N, dim = 3, 3
        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        sd.compute_geometry()

        def q_func(x):
            return np.array([-x[1], 2 * x[0], x[2] - 1])

        discr = pg.RT1()
        interp = discr.interpolate(sd, q_func)
        M = discr.assemble_mass_matrix(sd)

        computed_norm = interp @ M @ interp

        self.assertTrue(np.isclose(computed_norm, 2))


if __name__ == "__main__":
    unittest.main()
