import unittest
import numpy as np
import porepy as pp
import pygeon as pg


class ElasticityTest(unittest.TestCase):
    def setup(self, dim, h):
        labda = 1
        mu = 1
        sd = pg.unit_grid(dim, h, as_mdg=False)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("elasticity")

        sym_sym = vec_p1.assemble_symgrad_symgrad_matrix(sd)
        div_div = vec_p1.assemble_div_div_matrix(sd)

        A = mu * sym_sym + labda * div_div
        return sd, vec_p1, A

    def test_rigid_body_motion_2d(self):
        sd, vec_p1, A = self.setup(2, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 2)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

    def test_rigid_body_motion_3d(self):
        sd, vec_p1, A = self.setup(3, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 3)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.1 + x[1]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

    def test_footstep(self):
        sd, vec_p1, A = self.setup(2, 0.5)

        bottom = np.hstack([np.isclose(sd.nodes[1, :], 0)] * 2)
        top = np.isclose(sd.face_centers[1, :], 1)

        fun = lambda _: np.array([0, -1])  # [1, 0]
        b = vec_p1.assemble_nat_bc(sd, fun, top)

        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(bottom, np.zeros(vec_p1.ndof(sd)))
        u = ls.solve()

        # fmt: off
        u_known = np.array(
            [ 0.        ,  0.        ,  0.1776907 , -0.17951352,  0.        ,
              0.16157526, -0.00424376, -0.167963  , -0.07453339,  0.04330445,
             -0.03649198,  0.03261989,  0.        ,  0.        , -0.63391078,
             -0.63835749,  0.        , -0.31189007, -0.61793605, -0.31482189,
             -0.42682694, -0.36033026, -0.17899184, -0.1394673 ])
        # fmt: on

        self.assertTrue(np.allclose(u, u_known))

        # proj = vec_p1.eval_at_cell_centers(sd)
        # cell_u = proj @ u
        # cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
        # cell_u = cell_u.reshape((3, -1))

        # save = pp.Exporter(sd, "sol")
        # save.write_vtu([("cell_u", cell_u)])


if __name__ == "__main__":
    unittest.main()
