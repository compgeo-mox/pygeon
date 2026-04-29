"""Module for the discretizations of the matrix L2 space."""

from functools import cache
from typing import Callable, Type, cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class SymMatPwPolynomials(pg.Discretization):
    """
    Base class for symmetric matrix-valued piecewise polynomial discretizations.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order: int = pg.MATRIX
    """Matrix-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the symmetric matrix discretization class.
        The base discretization class is pg.VecPwPolynomials.

        Args:
            keyword (str): The keyword for the symmetric matrix discretization class.
                Default is pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr = pg.get_PwPolynomials(self.poly_order, pg.MATRIX)(keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.

        Args:
            sd (pg.Grid): The grid or a subclass.

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
        scalar_space = pg.get_PwPolynomials(self.poly_order, pg.SCALAR)()
        return (sd.dim + 1) * sd.dim // 2 * scalar_space.ndof_per_cell(sd)

    @cache
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix from the symmetric matrix-valued piecewise
        polynomial space to the full matrix-valued piecewise polynomial space.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        # Local expansion map from symmetric components to full matrix components.
        # Symmetric dofs are ordered by upper triangular entries in row-major order:
        # 2D (4x3): cols = (0,0), (0,1), (1,1)
        # 3D (9x6): cols = (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        num_entries = (sd.dim + 1) * sd.dim // 2

        # Create a matrix with the symmetric numbering in each entry
        indices = np.empty((sd.dim, sd.dim), dtype=int)
        i, j = np.triu_indices(sd.dim)
        indices[i, j] = np.arange(num_entries)
        indices[j, i] = np.arange(num_entries)

        # Create the transfer matrix from symmetric to full matrix numbering
        sym_adj = np.zeros((sd.dim**2, num_entries))
        sym_adj[np.arange(sd.dim**2), indices.flatten()] = 1

        # Number of dofs of the underlying scalar polynomial space. Full matrix dofs
        # are ordered component-wise.
        scalar_ndof = self.base_discr.ndof(sd) // (sd.dim**2)

        # Repeat the same component map for all scalar dofs.
        return sps.kron(sym_adj, sps.eye_array(scalar_ndof)).tocsc()

    @cache
    def assemble_symmetrizing_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix from the full matrix-valued piecewise
        polynomial space to the symmetric matrix-valued piecewise polynomial space.
        Off-diagonal entries are averaged: sym_{ij} = (full_{ij} + full_{ji}) / 2.
        This is the left inverse of proj_to_PwPolynomials, meaning that
        assemble_symmetrizing_matrix @ proj_to_PwPolynomials = I

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        # Local averaging map from full matrix components to symmetric components.
        # Transpose of the expansion map, with off-diagonal rows scaled by 1/2.
        num_entries = (sd.dim + 1) * sd.dim // 2

        # Create a matrix with the symmetric numbering in each entry
        indices = np.empty((sd.dim, sd.dim), dtype=int)
        i, j = np.triu_indices(sd.dim)
        indices[i, j] = np.arange(num_entries)
        indices[j, i] = np.arange(num_entries)

        # Create the transfer matrix from full to symmetric matrix numbering
        sym = np.zeros((num_entries, sd.dim**2))
        sym[indices.flatten(), np.arange(sd.dim**2)] = 0.5
        sym[indices.diagonal(), :] *= 2

        # Number of dofs of the underlying scalar polynomial space.
        scalar_ndof = self.base_discr.ndof(sd) // (sd.dim**2)

        # Repeat the same component map for all scalar dofs.
        return sps.kron(sym, sps.eye_array(scalar_ndof)).tocsc()

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to +1 order discretization in the space of matrix
        valued piecewise polynomials, without symmetrization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_higher_PwPolynomials(sd)
        return proj @ self.proj_to_PwPolynomials(sd)

    def interpolate(self, sd: pg.Grid, func: Callable) -> np.ndarray:
        """
        Interpolates a given matrix-valued function to the symmetric matrix-valued
        piecewise polynomial space. The full matrix-valued interpolant is projected
        onto the symmetric subspace, i.e. the symmetric part is taken.

        Args:
            sd (pg.Grid): The grid.
            func (callable): The function to interpolate.

        Returns:
            np.ndarray: The interpolated function, with shape (ndof,).
        """
        val = self.base_discr.interpolate(sd, func)  # shape (ndof_full,)
        return self.assemble_symmetrizing_matrix(sd) @ val  # shape (ndof_sym,)

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        This method takes a grid object and returns the differential matrix
        corresponding to the given grid.

        Args:
            sd (pg.Grid): The grid object or its subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array((0, self.ndof(sd)))

    def assemble_stiff_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.
            data (dict | None): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def assemble_nat_bc(
        self,
        sd: pg.Grid,
        _func: Callable[[np.ndarray], np.ndarray],
        _b_faces: np.ndarray,
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition vector, equal to zero.

        Args:
            sd (pg.Grid): The grid object.
            func (Callable[[np.ndarray], np.ndarray]): The function defining the
                 natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition vector.
        """
        return np.zeros(self.ndof(sd))

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no zero discretization available in PyGeoN.
        """
        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")


class SymMatPwConstants(SymMatPwPolynomials):
    """
    A class representing the discretization using symmetric matrix piecewise constant
    functions.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

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
        base_discr = cast(pg.MatPwConstants, self.base_discr)
        val = base_discr.mat_invert(sd, self.proj_to_PwPolynomials(sd) @ val)
        return self.assemble_symmetrizing_matrix(sd) @ val


class SymMatPwLinears(SymMatPwPolynomials):
    """
    A class representing the discretization using symmetric matrix piecewise linear
    functions.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""


class SymMatPwQuadratics(SymMatPwPolynomials):
    """
    A class representing the discretization using symmetric matrix piecewise quadratic
    functions.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""
