import unittest
import numpy as np
import porepy as pp
import pygeon as pg


class ElasticityTest(unittest.TestCase):
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
        labda = 1
        mu = 0.5
        sd = pg.unit_grid(dim, h, as_mdg=False)
        sd.compute_geometry()

        vec_p1 = pg.VecLagrange1("elasticity")

        sym_sym = vec_p1.assemble_symgrad_symgrad_matrix(sd)
        div_div = vec_p1.assemble_div_div_matrix(sd)

        A = 2 * mu * sym_sym + labda * div_div
        return sd, vec_p1, A, labda, mu

    def test_rigid_body_motion_2d(self):
        """
        Test case for 2D rigid body motion.

        This test case verifies the correctness of the solution for 2D rigid body motion.
        It sets up the problem, applies essential boundary conditions, solves the linear system,
        and checks if the computed solution matches the expected solution.

        Returns:
            None
        """
        sd, vec_p1, A, labda, mu = self.setup(2, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 2)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

        sigma = vec_p1.compute_stress(sd, u, labda, mu)

        self.assertTrue(np.allclose(sigma, 0))

    def test_rigid_body_motion_3d(self):
        """
        Test case for simulating rigid body motion in 3D.

        This test sets up a linear system and solves it to simulate rigid body motion in 3D.
        It verifies that the computed solution matches the expected solution within a tolerance.

        Returns:
            None
        """
        sd, vec_p1, A, labda, mu = self.setup(3, 0.125)

        b_nodes = np.hstack([sd.tags["domain_boundary_nodes"]] * 3)
        bc_fun = lambda x: np.array([0.5 - x[1], 0.5 + x[0] - x[2], 0.1 + x[1]])
        u_ex = vec_p1.interpolate(sd, bc_fun)

        ls = pg.LinearSystem(A)
        ls.flag_ess_bc(b_nodes, u_ex)
        u = ls.solve()

        self.assertTrue(np.allclose(u, u_ex))

        sigma = vec_p1.compute_stress(sd, u, labda, mu)

        self.assertTrue(np.allclose(sigma, 0))

    def test_footstep_2d(self):
        """
        Test case for simulating a 2D footstep using elasticity.

        This method tests the 2D footstep scenario in the elasticity module.
        It sets up the necessary parameters, assembles the linear system,
        solves it, and compares the obtained solution with the expected solution.

        Returns:
            None
        """
        sd, vec_p1, A, _, _ = self.setup(2, 0.5)

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

    def test_footstep_3d(self):
        """
        Test case for simulating a 3D footstep using elasticity.

        This test sets up a 3D problem with a footstep and solves it using elasticity equations.
        It verifies that the solution has a non-positive z component.

        Returns:
            None
        """
        sd, vec_p1, A, _, _ = self.setup(3, 0.5)

        bottom = np.hstack([np.isclose(sd.nodes[2, :], 0)] * 3)
        top = np.isclose(sd.face_centers[2, :], 1)

        fun = lambda _: np.array([0, 0, -1])  # [1, 0]
        b = vec_p1.assemble_nat_bc(sd, fun, top)

        ls = pg.LinearSystem(A, b)
        ls.flag_ess_bc(bottom, np.zeros(vec_p1.ndof(sd)))
        u = ls.solve()

        self.assertTrue(np.all(u[-sd.num_nodes :] <= 0))


if __name__ == "__main__":
    unittest.main()
