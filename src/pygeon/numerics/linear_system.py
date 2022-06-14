import numpy as np
import scipy.sparse as sps


class LinearSystem:
    """
    Class for storing a linear system consisting of the matrix and its
    right-hand side. The class keeps track of essential boundary conditions
    and reduces the system appropriately before solving.

    Attributes:
        A (sps.spmatrix, n x n): The left-hand side matrix
        b (np.array-like): The right-hand side vector
        is_dof (np.array, bool): Determines whether an entry is a degree of freedom.
            If False then it will be overwritten by an essential bc.
        ess_vals (np.array, (n, )): The values of the essential bcs.
    """

    def __init__(self, A, b=None) -> None:
        self.A = A

        if b is None:
            b = np.zeros(A.shape[0])
        self.b = b

        self.reset_bc()

    def reset_bc(self):
        self.is_dof = np.ones(self.b.size, dtype=bool)
        self.ess_vals = np.zeros(self.b.size)

    def flag_ess_bc(self, is_ess_dof, ess_vals):
        self.is_dof[is_ess_dof] = False
        self.ess_vals[is_ess_dof] += ess_vals[is_ess_dof]

    def reduce_system(self):
        R_0 = create_restriction(self.is_dof)
        A_0 = R_0 * self.A * R_0.T
        b_0 = R_0 * (self.b - self.A * self.ess_vals)

        return A_0, b_0, R_0

    def solve(self, solver=sps.linalg.spsolve):
        A_0, b_0, R_0 = self.reduce_system()

        sol_0 = solver(A_0, b_0)

        sol = R_0.T * sol_0 + self.ess_vals

        return sol


def create_restriction(keep_dof):
    """
    Helper function to create the restriction mapping

    Parameters:
        keep_dof (np.array, bool): True for the dofs of the system,
            False for the overwritten values

    Returns:
        sps.csr_matrix: the restriction mapping.
    """
    R = sps.diags(keep_dof, dtype=np.int).tocsr()
    return R[R.indices, :]
