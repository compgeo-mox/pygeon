""" Module for the discretizations of the vector L2 space. """

from typing import Callable, Optional

import numpy as np

import pygeon as pg


class VecPwConstants(pg.VecDiscretization):
    """
    A class representing the discretization using vector piecewise constant functions.

    Attributes:
        keyword (str): The keyword for the vector discretization class.

    Methods:
        get_range_discr_class(self, dim: int) -> pg.Discretization:
            Returns the discretization class for the range of the differential.

        assemble_nat_bc(self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition vector, equal to zero.
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        super().__init__(keyword, pg.PwConstants)

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the differential.
        """
        return self.scalar_discr.get_range_discr_class(dim)

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
        return np.zeros(self.ndof(sd))

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: Optional[bool] = True,
        etype: Optional[str] = "specific",
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to "specific".

        Returns:
            float: The computed error.
        """

        err2 = 0
        num_sol = num_sol.reshape((sd.dim, -1))
        for d in np.arange(sd.dim):
            ana_sol_dim = lambda x: ana_sol(x)[d]
            num_sol_dim = num_sol[d]

            err2_dim = self.scalar_discr.error_l2(
                sd, num_sol_dim, ana_sol_dim, relative, etype
            )
            err2 += err2_dim**2
        return np.sqrt(err2)


class VecPwLinears(pg.VecDiscretization):
    """
    A class representing the discretization using vector piecewise linear functions.

    Attributes:
        keyword (str): The keyword for the vector discretization class.

    Methods:
        get_range_discr_class(self, dim: int) -> pg.Discretization:
            Returns the discretization class for the range of the differential.

    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        super().__init__(keyword, pg.PwLinears)

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the differential.
        """
        return self.scalar_discr.get_range_discr_class(dim)

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
        return np.zeros(self.ndof(sd))
