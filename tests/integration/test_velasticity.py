import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class VElasticityTestPrimal(unittest.TestCase):
    """
    Test case class for elasticity module solved with the virtual element method.

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
        data = {pp.PARAMETERS: {"elasticity": {"lambda": 1, "mu": 0.5}}}
        sd = pp.CartGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_p1 = pg.VecVLagrange1("elasticity")

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
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.65562914e-01,
        -7.98448066e-18,  1.65562914e-01, -1.74077578e-01, -2.01775116e-16,
         1.74077578e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -3.07000946e-01, -2.75780511e-01, -3.07000946e-01, -6.32923368e-01,
        -6.19678335e-01, -6.32923368e-01])
        # fmt: on

        self.assertTrue(np.allclose(u, u_known))

        sigma = vec_p1.compute_stress(sd, u, data)

        self.assertTrue(np.all(np.trace(sigma, axis1=1, axis2=2) <= 0))


if __name__ == "__main__":
    unittest.main()
