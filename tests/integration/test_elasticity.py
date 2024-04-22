import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class ElasticityTestPrimal(unittest.TestCase):
    """
    Test case class for elasticity module.

    This class contains test methods for various scenarios in the elasticity module.
    Each test method sets up the necessary parameters, assembles the linear system,
    solves it, and compares the obtained solution with the expected solution.

    """

    def setup(self, dim, h):
        """
        Set up the test case for elasticity.

        Args:
            dim (int): The dimension of the problem.
            h (float): The grid spacing.

        Returns:
            Tuple: A tuple containing the following elements:
                - sd (pg.Mesh): The structured mesh.
                - vec_p1 (pg.VecLagrange1): The Lagrange finite element space.
                - A (pg.Matrix): The assembled elasticity matrix.
        """
        data = {"lambda": 1, "mu": 0.5}
        sd = pg.unit_grid(dim, h, as_mdg=False)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("elasticity")

        A = vec_p1.assemble_stiff_matrix(sd, data)

        return sd, vec_p1, A, data

    def test_rigid_body_motion_2d(self):
        """
        Test case for 2D rigid body motion.

        This test case verifies the correctness of the solution for 2D rigid body motion.
        It sets up the problem, applies essential boundary conditions, solves the linear system,
        and checks if the computed solution matches the expected solution.

        Returns:
            None
        """
        sd, vec_p1, A, data = self.setup(2, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 2)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

        sigma = vec_p1.compute_stress(sd, u, data)

        self.assertTrue(np.allclose(sigma, 0))

    def test_rigid_body_motion_3d(self):
        """
        Test case for simulating rigid body motion in 3D.

        This test sets up a linear system and solves it to simulate rigid body motion in 3D.
        It verifies that the computed solution matches the expected solution within a tolerance.

        Returns:
            None
        """
        sd, vec_p1, A, data = self.setup(3, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 3)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.1 + x[1]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

        sigma = vec_p1.compute_stress(sd, u, data)

        self.assertTrue(np.allclose(sigma, 0))

    def test_footing_2d(self):
        """
        Test case for simulating a 2D footing using elasticity.

        This method tests the 2D footing scenario in the elasticity module.
        It sets up the necessary parameters, assembles the linear system,
        solves it, and compares the obtained solution with the expected solution.

        Returns:
            None
        """
        sd, vec_p1, A, data = self.setup(2, 0.5)

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

        sigma = vec_p1.compute_stress(sd, u, data)

        self.assertTrue(np.all(np.trace(sigma, axis1=1, axis2=2) <= 0))

    def test_footing_3d(self):
        """
        Test case for simulating a 3D footing using elasticity.

        This test sets up a 3D problem with a footing and solves it using elasticity equations.
        It verifies that the solution has a non-positive z component.

        Returns:
            None
        """
        sd, vec_p1, A, data = self.setup(3, 0.5)

        bottom = np.hstack([np.isclose(sd.nodes[2, :], 0)] * 3)
        top = np.isclose(sd.face_centers[2, :], 1)

        fun = lambda _: np.array([0, 0, -1])  # [1, 0]
        b = vec_p1.assemble_nat_bc(sd, fun, top)

        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(bottom, np.zeros(vec_p1.ndof(sd)))
        u = ls.solve()

        self.assertTrue(np.all(u[-sd.num_nodes :] <= 0))

        sigma = vec_p1.compute_stress(sd, u, data)

        self.assertTrue(np.all(np.trace(sigma, axis1=1, axis2=2) <= 0))


class ElasticityTestMixed(unittest.TestCase):
    def run_elasticity_2d(self, u_boundary, N):
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "elasticity"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)
        p0 = pg.PwConstants(key)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        Ms = vec_bdm1.assemble_mass_matrix(sd, data)
        Mu = vec_p0.assemble_mass_matrix(sd)
        Mr = p0.assemble_mass_matrix(sd)

        div = Mu @ vec_bdm1.assemble_diff_matrix(sd)
        asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)

        # fmt: off
        spp = sps.bmat([[   Ms, div.T, asym.T],
                        [ -div,  None,   None],
                        [-asym,  None,   None]], format = "csc")
        # fmt: on

        b_faces = sd.tags["domain_boundary_faces"]
        bc = vec_bdm1.assemble_nat_bc(sd, u_boundary, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[: vec_bdm1.ndof(sd)] = bc

        x = sps.linalg.spsolve(spp, rhs)

        split_idx = np.cumsum([vec_bdm1.ndof(sd), vec_p0.ndof(sd)])
        sigma, u, r = np.split(x, split_idx)

        cell_sigma = vec_bdm1.eval_at_cell_centers(sd) @ sigma
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = p0.eval_at_cell_centers(sd) @ r

        return cell_sigma, cell_u, cell_r, sd

    def test_elasticity_rbm_2d(self):
        N = 3
        u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0], 0])
        cell_sigma, cell_u, cell_r, sd = self.run_elasticity_2d(u_boundary, N)

        key = "elasticity"
        vec_p0 = pg.VecPwConstants(key)
        interp = vec_p0.interpolate(sd, u_boundary)
        u_known = vec_p0.eval_at_cell_centers(sd) @ interp

        self.assertTrue(np.allclose(cell_sigma, 0))
        self.assertTrue(np.allclose(cell_u, u_known))
        self.assertTrue(np.allclose(cell_r, -1))

    def test_elasticity_2d(self):
        N = 3
        u_boundary = lambda x: np.array([x[0], x[1], 0])
        cell_sigma, cell_u, cell_r, sd = self.run_elasticity_2d(u_boundary, N)

        key = "elasticity"
        vec_p0 = pg.VecPwConstants(key)
        interp = vec_p0.interpolate(sd, u_boundary)
        u_known = vec_p0.eval_at_cell_centers(sd) @ interp

        cell_sigma = cell_sigma.reshape((6, -1))

        self.assertTrue(np.allclose(cell_sigma[0], 2))
        self.assertTrue(np.allclose(cell_sigma[1], 0))
        self.assertTrue(np.allclose(cell_sigma[2], 0))
        self.assertTrue(np.allclose(cell_sigma[3], 0))
        self.assertTrue(np.allclose(cell_sigma[4], 2))
        self.assertTrue(np.allclose(cell_sigma[5], 0))
        self.assertTrue(np.allclose(cell_u, u_known))
        self.assertTrue(np.allclose(cell_r, 0))

    def run_elasticity_3d(self, u_boundary, N):
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "elasticity"
        vec_bdm1 = pg.VecBDM1(key)
        vec_p0 = pg.VecPwConstants(key)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        Ms = vec_bdm1.assemble_mass_matrix(sd, data)
        Mu = vec_p0.assemble_mass_matrix(sd)
        Mr = Mu

        div = Mu @ vec_bdm1.assemble_diff_matrix(sd)
        asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)

        # fmt: off
        spp = sps.bmat([[   Ms, div.T, asym.T],
                        [ -div,  None,   None],
                        [-asym,  None,   None]], format = "csc")
        # fmt: on

        b_faces = sd.tags["domain_boundary_faces"]
        bc = vec_bdm1.assemble_nat_bc(sd, u_boundary, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[: vec_bdm1.ndof(sd)] = bc

        x = sps.linalg.spsolve(spp, rhs)

        split_idx = np.cumsum([vec_bdm1.ndof(sd), vec_p0.ndof(sd)])
        sigma, u, r = np.split(x, split_idx)

        cell_sigma = vec_bdm1.eval_at_cell_centers(sd) @ sigma
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = vec_p0.eval_at_cell_centers(sd) @ r

        return cell_sigma, cell_u, cell_r, sd

    def test_elasticity_rbm_3d(self):
        N = 3
        u_boundary = lambda x: np.array([-0.5 - x[1], -0.5 + x[0] - x[2], -0.5 + x[1]])
        cell_sigma, cell_u, cell_r, sd = self.run_elasticity_3d(u_boundary, N)

        key = "elasticity"
        vec_p0 = pg.VecPwConstants(key)
        interp = vec_p0.interpolate(sd, u_boundary)
        u_known = vec_p0.eval_at_cell_centers(sd) @ interp

        self.assertTrue(np.allclose(cell_sigma, 0))
        self.assertTrue(np.allclose(cell_u, u_known))

        cell_r = cell_r.reshape((3, -1))

        self.assertTrue(np.allclose(cell_r[0], 1))
        self.assertTrue(np.allclose(cell_r[1], 0))
        self.assertTrue(np.allclose(cell_r[2], 1))

    def test_elasticity_3d(self):
        N = 3
        u_boundary = lambda x: np.array([x[0], x[1], x[2]])
        cell_sigma, cell_u, cell_r, sd = self.run_elasticity_3d(u_boundary, N)

        key = "elasticity"
        vec_p0 = pg.VecPwConstants(key)
        interp = vec_p0.interpolate(sd, u_boundary)
        u_known = vec_p0.eval_at_cell_centers(sd) @ interp

        cell_sigma = cell_sigma.reshape((9, -1))

        self.assertTrue(np.allclose(cell_sigma[0], 2.5))
        self.assertTrue(np.allclose(cell_sigma[1], 0))
        self.assertTrue(np.allclose(cell_sigma[2], 0))
        self.assertTrue(np.allclose(cell_sigma[3], 0))
        self.assertTrue(np.allclose(cell_sigma[4], 2.5))
        self.assertTrue(np.allclose(cell_sigma[5], 0))
        self.assertTrue(np.allclose(cell_sigma[6], 0))
        self.assertTrue(np.allclose(cell_sigma[7], 0))
        self.assertTrue(np.allclose(cell_sigma[8], 2.5))

        self.assertTrue(np.allclose(cell_u, u_known))
        self.assertTrue(np.allclose(cell_r, 0))


if __name__ == "__main__":
    unittest.main()
