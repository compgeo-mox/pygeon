import abc
from typing import Callable, Tuple

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Discretization(pp.numerics.discretization.Discretization):
    """
    Abstract class for pygeon discretization methods.
    For full compatibility, a child class requires the following methods:
        ndof
        assemble_mass_matrix
        assemble_diff_matrix
        interpolate
        eval_at_cell_centers
        assemble_nat_bc
        get_range_discr_class
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the Discretization object.

        Args:
            keyword (str): The keyword used to identify the discretization method.

        Attributes:
            mass_matrix_key (str): The keyword used to identify the mass matrix
                term in the discretization matrix dictionary.
            diff_matrix_key (str): The keyword used to identify the diffusion matrix
                term in the discretization matrix dictionary.
            stiff_matrix_key (str): The keyword used to identify the stiffness matrix
                term in the discretization matrix dictionary.
            lumped_matrix_key (str): The keyword used to identify the lumped matrix
                term in the discretization matrix dictionary.
        """
        super().__init__(keyword)

        self.mass_matrix_key = "mass"
        self.diff_matrix_key = "diff"
        self.stiff_matrix_key = "stiff"
        self.lumped_matrix_key = "lumped"

    def discretize(self, sd: pg.Grid, data: dict) -> None:
        """
        Discretizes the given grid using the specified data.

        Args:
            sd (pg.Grid): The grid to be discretized.
            data (dict): The data required for discretization.

        Returns:
            None
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        matrix_dictionary[self.mass_matrix_key] = self.assemble_mass_matrix(sd, data)
        matrix_dictionary[self.diff_matrix_key] = self.assemble_diff_matrix(sd)
        matrix_dictionary[self.stiff_matrix_key] = self.assemble_stiff_matrix(sd, data)
        matrix_dictionary[self.lumped_matrix_key] = self.assemble_lumped_matrix(
            sd, data
        )

    @abc.abstractmethod
    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None) -> sps.csc_matrix:
        """
        Assembles the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict, optional): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The mass matrix.
        """

        pass

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the lumped mass matrix given by the row sums on the diagonal.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict, optional): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        diag_mass = np.sum(self.assemble_mass_matrix(sd, data), axis=0)
        return sps.diags(np.asarray(diag_mass).flatten()).tocsc()

    @abc.abstractmethod
    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        pass

    def assemble_stiff_matrix(self, sd: pg.Grid, data: dict = None) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix.

        This method takes a grid object `sd` and an optional data dictionary `data` as input.
        It first calls the `assemble_diff_matrix` method to obtain the differential matrix `B`.
        Then, it creates an instance of the range discretization class using the `dim` attribute of `sd`.
        Next, it calls the `assemble_mass_matrix` method of the range discretization class
        to obtain the mass matrix `A`.
        Finally, it returns the product of the transpose of `B`, `A`, and `B`.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict, optional): Optional data dictionary. Defaults to None.

        Returns:
            sps.csc_matrix: The stiffness matrix.
        """
        B = self.assemble_diff_matrix(sd)

        discr = self.get_range_discr_class(sd.dim)(self.keyword)
        A = discr.assemble_mass_matrix(sd, data)

        return B.T * A * B

    @abc.abstractmethod
    def interpolate(self, sd: pg.Grid, func: Callable) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
            func (Callable): a function that returns the function values at coordinates

        Returns:
            np.ndarray: the values of the degrees of freedom
        """
        pass

    @abc.abstractmethod
    def eval_at_cell_centers(self, sd: pg.Grid) -> np.ndarray:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            np.ndarray: The evaluation matrix.
        """
        pass

    def source_term(self, sd: pg.Grid, func: Callable) -> np.ndarray:
        """
        Assembles the source term by interpolating the given function
        and multiplying by the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The evaluation matrix.
        """

        return self.assemble_mass_matrix(sd) * self.interpolate(sd, func)

    @abc.abstractmethod
    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable, b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (Tr q, p)_Gamma

        Args:
            sd (pg.Grid): The grid object.
            func (Callable): The function representing the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        pass

    @abc.abstractmethod
    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the discretization class that contains the range of the differential

        Args:
            dim (int): The dimension of the range

        Returns:
            pg.Discretization: The discretization class containing the range of the differential
        """
        pass

    def assemble_matrix_rhs(
        self, sd: pg.Grid, data: dict = None
    ) -> Tuple(sps.csc_matrix, np.ndarray):
        """
        Assembles a mass matrix and returns it along with a zero rhs vector.

        Args:
            sd (pg.Grid): The grid on which the matrix is assembled.
            data (dict, optional): Additional data for the assembly process. Defaults to None.

        Returns:
            Tuple(sps.csc_matrix, np.ndarray): The assembled mass matrix and zero rhs vector.
        """
        return self.assemble_mass_matrix(sd, data), np.zeros(self.ndof(sd))

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable,
        relative: bool = True,
        etype: str = "standard",
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable): Function that represents the analytical solution.
            relative (bool, optional): Compute the relative error or not. Defaults to True.
            etype (str, optional): Type of error computed. For "standard", the current implementation. Defaults to "standard".

        Returns:
            float: The computed error.
        """
        int_sol = self.interpolate(sd, ana_sol)
        mass = self.assemble_mass_matrix(sd)

        norm = (int_sol @ mass @ int_sol.T) if relative else 1

        diff = num_sol - int_sol
        return np.sqrt(diff @ mass @ diff.T / norm)
