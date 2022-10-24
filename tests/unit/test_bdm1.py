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

        E = discr_bdm1.proj_from_RT0(sd)

        check = E.T * mass_bdm1 * E - mass_rt0

        self.assertEqual(check.nnz, 0)

    def test_interpolation_2D(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()
        discr_bdm1 = pg.BDM1("flow")

        def q_linear(x):
            return x

        interp_q = discr_bdm1.interpolate(sd, q_linear)
        eval_q = discr_bdm1.eval_at_cell_centers(sd) * interp_q
        eval_q = np.reshape(eval_q, (3, -1), order="F")

        known_q = np.array([q_linear(x) for x in sd.cell_centers.T]).T
        self.assertAlmostEqual(np.linalg.norm(eval_q - known_q), 0)

    def test_interpolation_3D(self):
        N, dim = 2, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()
        discr_bdm1 = pg.BDM1("flow")

        def q_linear(x):
            return x

        interp_q = discr_bdm1.interpolate(sd, q_linear)
        eval_q = discr_bdm1.eval_at_cell_centers(sd) * interp_q
        eval_q = np.reshape(eval_q, (3, -1), order="F")

        known_q = np.array([q_linear(x) for x in sd.cell_centers.T]).T
        self.assertAlmostEqual(np.linalg.norm(eval_q - known_q), 0)

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

        E = discr_bdm1.proj_from_RT0(sd)

        check = E.T * mass_bdm1 * E - mass_rt0
        self.assertAlmostEqual(np.linalg.norm(check.data), 0)

    def test3(self):
        N, dim = 5, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        self.linear_distribution_test(sd)

    def test4(self):
        N, dim = 5, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        self.linear_distribution_test(sd)

    def linear_distribution_test(self, sd):

        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        discr_p0 = pg.PwConstants("flow")

        # face_mass = discr_bdm1.assemble_lumped_matrix(sd, None)
        face_mass = discr_bdm1.assemble_mass_matrix(sd, None)
        cell_mass = discr_p0.assemble_mass_matrix(sd, None)

        div = cell_mass * discr_bdm1.assemble_diff_matrix(sd)

        # assemble the saddle point problem
        spp = sps.bmat([[face_mass, -div.T], [div, None]], format="csc")

        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

        def p_0(x):
            return x[0]

        bc_val = -discr_bdm1.assemble_nat_bc(sd, p_0, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[: bc_val.size] += bc_val

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-sd.num_cells :]

        face_proj = discr_bdm1.eval_at_cell_centers(sd)
        cell_proj = discr_p0.eval_at_cell_centers(sd)

        cell_q = (face_proj * q).reshape((3, -1), order="F")
        cell_p = cell_proj * p

        known_q = np.zeros(cell_q.shape)
        known_q[0, :] = -1.0
        known_p = sd.cell_centers[0, :]

        self.assertAlmostEqual(np.linalg.norm(cell_q - known_q), 0)
        self.assertAlmostEqual(np.linalg.norm(cell_p - known_p), 0)


if __name__ == "__main__":

    unittest.main()
