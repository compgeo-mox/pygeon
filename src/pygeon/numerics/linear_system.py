import numpy as np
import scipy.sparse as sps


class LinearSystem:
    def __init__(self, A, b=None) -> None:
        self.A = A

        if b is None:
            b = np.zeros(A.shape[0])
        self.b = b

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
    R = sps.diags(keep_dof, dtype=np.int).tocsr()
    return R[R.indices, :]
