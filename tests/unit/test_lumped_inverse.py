import unittest

import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class LumpedInverseTest(unittest.TestCase):
    def setUp(self):
        # Create a sparse matrix M and a right-hand side matrix B
        M = np.array(
            [
                [2, 1, 0, 0, 0],
                [1, 2, 0, 0, 0],
                [0, 0, 6, 1, 0],
                [0, 0, 1, 8, 2],
                [0, 0, 0, 2, 2],
            ]
        )
        M_sparse = sps.csc_array(M)

        B = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        B_sparse = sps.csc_array(B)

        # Create a right-hand side vector b
        b_vector = np.array([1, 2, 3, 0, 1])
        b_matrix = np.tile(b_vector, (3, 1)).T

        return M_sparse, B_sparse, b_vector, b_matrix

    def test_block_diag_solver_dense_vec(self):
        M, _, b, _ = self.setUp()

        # Solve the system of equations using the block diagonal solver
        sol = pg.block_diag_solver_dense(M, b)

        # Check if the solution is correct
        expected_sol = sps.linalg.spsolve(M, b)
        self.assertTrue(np.allclose(sol, expected_sol))

        # Check if the shape of the solution is correct
        self.assertEqual(sol.shape, (5,))

        # Check if the solution is a 2D array
        self.assertEqual(sol.ndim, 1)

    def test_block_diag_solver_dense_mat(self):
        M, _, _, B = self.setUp()

        # Solve the system of equations using the block diagonal solver
        sol = pg.block_diag_solver_dense(M, B)

        # Check if the solution is correct
        expected_sol = sps.linalg.spsolve(M, B)
        self.assertTrue(np.allclose(sol, expected_sol))

        # Check if the shape of the solution is correct
        self.assertEqual(sol.shape, (5, 3))

        # Check if the solution is a 2D array
        self.assertEqual(sol.ndim, 2)

    def test_block_diag_solver(self):
        M, B, _, _ = self.setUp()

        # Solve the system of equations using the block diagonal solver
        sol = pg.block_diag_solver(M, B)

        # Check if the solution is correct
        expected_sol = sps.linalg.spsolve(M, B.todense())
        self.assertTrue(np.allclose(sol.toarray(), expected_sol))

        # Check if the shape of the solution is correct
        self.assertTrue(np.allclose(sol.shape, (5, 6)))

        # Check if the solution is a sparse matrix
        self.assertTrue(sps.issparse(sol))

    def test_assemble_inverse(self):
        M, _, _, _ = self.setUp()

        # Solve the system of equations using the block diagonal solver
        invM = pg.assemble_inverse(M)
        expected_invM = np.linalg.inv(M.toarray())

        # Check if the solution is correct
        self.assertTrue(np.allclose(invM.toarray(), expected_invM))


if __name__ == "__main__":
    unittest.main()
