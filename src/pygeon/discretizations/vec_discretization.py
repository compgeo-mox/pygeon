""" Module for the vector discretization class. """

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VecDiscretization(pg.Discretization):
    """
    A class representing a vector discretization.

    Attributes:
        keyword (str): The keyword for the vector discretization class.
        scalar_discr (pg.Discretization): The scalar discretization object.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles and returns the mass matrix for the lowest order Lagrange element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix given by the row sums on the diagonal.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the finite element solution at the cell centers of the given grid.

        assemble_nat_bc(
            sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
        ) -> np.ndarray:
            Assembles the natural boundary condition vector.
    """

    def __init__(self, keyword: str, scalar_discr: pg.Discretization) -> None:
        """
        Initialize the vector discretization class.

        Args:
            keyword (str): The keyword for the vector discretization class.
            scalar_discr (pg.Discretization): The scalar discretization object.

        Returns:
            None
        """
        super().__init__(keyword)
        # a local discr class for performing some of the computations
        self.scalar_discr = scalar_discr(keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case, it returns the product of the number of nodes and
        the dimension of the grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            int: The number of degrees of freedom.
        """
        return self.scalar_discr.ndof(sd) * sd.dim

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles and returns the mass matrix for the lowest order Lagrange element.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly.

        Returns:
            sps.csc_matrix: The mass matrix obtained from the discretization.
        """
        mass = self.scalar_discr.assemble_mass_matrix(sd, data)
        return sps.block_diag([mass] * sd.dim, format="csc")

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        diff = self.scalar_discr.assemble_diff_matrix(sd)
        return sps.block_diag([diff] * sd.dim, format="csc")

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict, optional): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        lumped_mass = self.scalar_discr.assemble_lumped_matrix(sd, data)
        return sps.block_diag([lumped_mass] * sd.dim, format="csc")

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
            func (Callable): a function that returns the function values at coordinates

        Returns:
            np.ndarray: the values of the degrees of freedom
        """
        interp = [
            self.scalar_discr.interpolate(sd, lambda x: func(x)[d])
            for d in np.arange(sd.dim)
        ]
        return np.hstack(interp)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_matrix: The finite element solution evaluated at the cell centers.
        """
        P = self.scalar_discr.eval_at_cell_centers(sd)
        return sps.block_diag([P] * sd.dim, format="csc")

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
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
        nat_bc = [
            self.scalar_discr.assemble_nat_bc(sd, lambda x: func(x)[d], b_faces)
            for d in np.arange(sd.dim)
        ]
        return np.hstack(nat_bc)
