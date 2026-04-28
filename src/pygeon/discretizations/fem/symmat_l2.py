"""Module for the discretizations of the matrix L2 space."""

from functools import cache

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class SymMatPwPolynomials(pg.MatPwPolynomials):
    """
    Base class for matrix-valued piecewise polynomial discretizations.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order: int = pg.MATRIX
    """Matrix-valued discretization"""

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom for the symmetric matrix-valued
        piecewise polynomial discretization.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            int: The number of degrees of freedom.
        """
        return self.ndof_per_cell(sd) * sd.num_cells

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom for each cell in the symmetric
        matrix-valued piecewise polynomial discretization.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            int: The number of degrees of freedom per cell.
        """
        return int((sd.dim + 1) / (2 * sd.dim) * super().ndof_per_cell(sd))

    @cache
    def assemble_sym_adj_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix from the symmetric matrix-valued piecewise
        polynomial space to the full matrix-valued piecewise polynomial space.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        # Number of dofs of the underlying scalar polynomial space. Full matrix dofs
        # are ordered component-wise.
        scalar_ndof = super().ndof(sd) // (sd.dim**2)

        # Local expansion map from symmetric components to full matrix components.
        # Symmetric dofs are ordered by upper triangular entries in row-major order:
        # 2D (4x3): cols = (0,0), (0,1), (1,1)
        # 3D (9x6): cols = (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        match sd.dim:
            case 2:
                sym_adj = np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=float,
                )
            case 3:
                sym_adj = np.array(
                    [
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            case _:
                raise ValueError("The grid should be either two or three-dimensional")

        # Repeat the same component map for all scalar dofs.
        result = sps.kron(sym_adj, sps.eye_array(scalar_ndof)).tocsc()
        result.eliminate_zeros()
        return result

    @cache
    def assemble_sym_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix from the full matrix-valued piecewise
        polynomial space to the symmetric matrix-valued piecewise polynomial space.
        Off-diagonal entries are averaged: sym_{ij} = (full_{ij} + full_{ji}) / 2.
        This is the left inverse of assemble_sym_adj_matrix, meaning that
        assemble_sym_matrix @ assemble_sym_adj_matrix = I

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        # Number of dofs of the underlying scalar polynomial space.
        scalar_ndof = super().ndof(sd) // (sd.dim**2)

        # Local averaging map from full matrix components to symmetric components.
        # Transpose of the expansion map, with off-diagonal rows scaled by 1/2.
        match sd.dim:
            case 2:
                sym = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 0.5, 0.5, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            case 3:
                sym = np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0.5, 0, 0.5, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, 0, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0.5, 0, 0.5, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            case _:
                raise ValueError("The grid should be either two or three-dimensional")

        # Repeat the same component map for all scalar dofs.
        result = sps.kron(sym, sps.eye_array(scalar_ndof)).tocsc()
        result.eliminate_zeros()
        return result

    def interpolate(self, sd: pg.Grid, func: callable) -> np.ndarray:
        """
        Interpolates a given matrix-valued function to the symmetric matrix-valued
        piecewise polynomial space. The input function is assumed to be symmetric, but
        this is not enforced.

        Args:
            sd (pg.Grid): The grid.
            func (callable): The function to interpolate.

        Returns:
            np.ndarray: The interpolated function, with shape (ndof,).
        """
        val = super().interpolate(sd, func)  # shape (ndof_full,)
        return self.assemble_sym_matrix(sd) @ val  # shape (ndof_sym,)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
             sps.csc_array: The evaluation matrix.
        """
        val = super().eval_at_cell_centers(sd)  # shape (n_cells, ndof_full)
        return val @ self.assemble_sym_adj_matrix(sd)  # shape (n_cells, ndof_sym)


class SymMatPwConstants(SymMatPwPolynomials, pg.MatPwConstants):
    """
    A class representing the discretization using symmetric matrix piecewise constant
    functions.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the symmetric matrix discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the symmetric matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwConstants(keyword)

    def mat_invert(self, sd: pg.Grid, val: np.ndarray) -> np.ndarray:
        """
        Inverts a matrix-valued function in the symmetric matrix piecewise constant
        space.

        Args:
            sd (pg.Grid): The grid.
            val (np.ndarray): The matrix-valued function to invert. It is assumed to be
                symmetric and piecewise constant and can be provided with shape (ndof,).

        Returns:
            np.ndarray: The inverted matrix-valued function, with the same shape as val.
        """
        val = super().mat_invert(sd, self.assemble_sym_adj_matrix(sd) @ val)
        return self.assemble_sym_matrix(sd) @ val


class SymMatPwLinears(SymMatPwPolynomials):
    """
    A class representing the discretization using symmetric matrix piecewise linear
    functions.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the symmetric matrix discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the symmetric matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwLinears(keyword)


class SymMatPwQuadratics(SymMatPwPolynomials):
    """
    A class representing the discretization using symmetric matrix piecewise quadratic
    functions.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the symmetric matrix discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the symmetric matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwQuadratics(keyword)
