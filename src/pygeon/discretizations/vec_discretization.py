"""Module for the vector discretization class."""

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VecDiscretization(pg.Discretization):
    """
    A class representing a vector discretization.

    Attributes:
        keyword (str): The keyword for the vector discretization class.
        base_discr (pg.Discretization): The base discretization object.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles and returns the mass matrix for the lowest order Lagrange element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Assembles the lumped mass matrix given by the row sums on the diagonal.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a function onto the finite element space.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Evaluate the finite element solution at the cell centers of the given grid.

        assemble_nat_bc(
            sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
        ) -> np.ndarray:
            Assembles the natural boundary condition vector.
    """

    def __init__(self, keyword: str):
        """
        Initializes the VecDiscretization class.

        Args:
            keyword (str): The keyword for the vector discretization class.
        """
        super().__init__(keyword)
        self.base_discr: pg.Discretization

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
        return self.base_discr.ndof(sd) * sd.dim

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for the lowest order Lagrange element.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        mass = self.base_discr.assemble_mass_matrix(sd, data)
        return sps.block_diag([mass] * sd.dim).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        diff = self.base_discr.assemble_diff_matrix(sd)
        return sps.block_diag([diff] * sd.dim).tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict, optional): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        lumped_mass = self.base_discr.assemble_lumped_matrix(sd, data)
        return sps.block_diag([lumped_mass] * sd.dim).tocsc()

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
            self.base_discr.interpolate(sd, lambda x: func(x)[d])
            for d in np.arange(sd.dim)
        ]
        return np.hstack(interp)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_array: The finite element solution evaluated at the cell centers.
        """
        P = self.base_discr.eval_at_cell_centers(sd)
        return sps.block_diag([P] * sd.dim).tocsc()

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
            self.base_discr.assemble_nat_bc(sd, lambda x: func(x)[d], b_faces)
            for d in np.arange(sd.dim)
        ]
        return np.hstack(nat_bc)
