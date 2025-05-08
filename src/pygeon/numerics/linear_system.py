"""Module for the LinearSystem class."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps


class LinearSystem:
    """
    Class for storing a linear system consisting of the matrix and its
    right-hand side. The class keeps track of essential boundary conditions
    and reduces the system appropriately before solving.

    Attributes:
        A (sps.csc_array, n x n): The left-hand side matrix
        b (np.array-like): The right-hand side vector
        is_dof (np.array, bool): Determines whether an entry is a degree of freedom.
            If False then it will be overwritten by an essential bc.
        ess_vals (np.array, (n, )): The values of the essential bcs.
    """

    def __init__(self, A: sps.csc_array, b: Optional[np.ndarray] = None) -> None:
        """
        Initialize a LinearSystem object.

        Args:
            A (sps.csc_array): The coefficient matrix of the linear system.
            b (np.ndarray, optional): The right-hand side vector of the linear system.
                Defaults to None.

        Returns:
            None
        """
        self.A = A

        if b is None:
            b = np.zeros(A.shape[0])
        self.b = b

        self.reset_bc()

    def reset_bc(self) -> None:
        """
        Reset the boundary conditions.

        This method sets the degrees of freedom (is_dof) to True for all elements
        in the b vector, and sets the essential values (ess_vals) to zero for all
        elements in the b vector.
        """
        self.is_dof = np.ones(self.b.shape[0], dtype=bool)
        self.ess_vals = np.zeros(self.b.shape[0])

    def flag_ess_bc(self, is_ess_dof: np.ndarray, ess_vals: np.ndarray) -> None:
        """
        Flags the essential boundary conditions for the degrees of freedom specified
        by `is_ess_dof`.

        Args:
            is_ess_dof (np.ndarray): Boolean array indicating the degrees of freedom
                to flag as essential.
            ess_vals (np.ndarray): Array of essential values corresponding to the
                flagged degrees of freedom.

        Returns:
            None
        """
        self.is_dof[is_ess_dof] = False
        self.ess_vals[is_ess_dof] += ess_vals[is_ess_dof]

    def reduce_system(self) -> Tuple[sps.csc_array, np.ndarray, sps.csc_array]:
        """
        Reduces the linear system by applying a restriction operator and returning
        the reduced system.

        Returns:
            A tuple containing the reduced matrix A, the reduced vector b, and the
                restriction operator R.
        """
        R_0 = create_restriction(self.is_dof)
        A_0 = R_0 @ self.A @ R_0.T
        b_0 = R_0 @ (self.b - self.A @ self.repeat_ess_vals())

        return A_0, b_0, R_0

    def solve(self, solver: Callable = sps.linalg.spsolve) -> np.ndarray:
        """
        Solve the linear system of equations.

        Args:
            solver (Callable): The solver function to use. Defaults to
                sps.linalg.spsolve.

        Returns:
            np.ndarray: The solution to the linear system of equations.
        """
        A_0, b_0, R_0 = self.reduce_system()
        sol_0 = solver(A_0.tocsc(), b_0)
        sol = self.repeat_ess_vals() + R_0.T @ sol_0

        return sol

    def repeat_ess_vals(self) -> Union[np.ndarray, sps.csc_array]:
        """
        Repeat the essential values of the linear system.

        If the input vector `b` has dimension 1, the method returns the essential values
        as is. Otherwise, it repeats the essential values for each column of `b`.

        Returns:
            numpy.ndarray or scipy.sparse.csc_array: The repeated essential values.
        """
        if self.b.ndim == 1:
            return self.ess_vals
        elif not np.any(self.ess_vals):
            return sps.csc_array(self.b.shape)
        else:
            ess_vals = sps.csr_array(np.atleast_2d(self.ess_vals))
            return sps.vstack([ess_vals] * self.b.shape[1]).T


def create_restriction(keep_dof: np.ndarray) -> sps.csc_array:
    """
    Helper function to create the restriction mapping

    Args:
        keep_dof (np.ndarray): Boolean array indicating which degrees of freedom (dofs)
            to keep. True for the dofs of the system, False for the overwritten values.

    Returns:
        sps.csc_array: The restriction mapping matrix.
    """
    R = sps.diags_array(keep_dof, dtype=int).tocsr()
    return R[R.indices, :].tocsc()
