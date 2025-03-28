"""Module for the discretizations of the vector L2 space."""

from typing import Callable, Optional, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VecPieceWisePolynomial(pg.VecDiscretization):
    """
    A class representing an abstract vector piecewise polynomial discretization.

    Attributes:
        keyword (str): The keyword for the vector discretization class.
        scalar_discr (pg.Discretization): The scalar discretization class.

    Methods:
        get_range_discr_class(dim: int) -> Type[pg.Discretization]:
            Returns the discretization class for the range of the differential.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces:
            np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition vector, equal to zero.

    """

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the
            differential.
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


class VecPwConstants(VecPieceWisePolynomial):
    """
    A class representing the discretization using vector piecewise constant functions.

    Attributes:
        keyword (str): The keyword for the vector discretization class.
        scalar_discr (pg.Discretization): The scalar discretization class.

    Methods:
        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray],
            np.ndarray], relative: Optional[bool] = True, etype:
            Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a
            function.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        self.scalar_discr: pg.PwConstants
        super().__init__(keyword, pg.PwConstants)

    def proj_to_pwLinears(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix to the vector piecewise linear space.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.scalar_discr.proj_to_pwLinears(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: bool = True,
        etype: str = "specific",
        data: Optional[dict] = None,
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a
        function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to
                "specific".

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
            err2 += np.square(err2_dim)
        return np.sqrt(err2)


class VecPwLinears(VecPieceWisePolynomial):
    """
    A class representing the discretization using vector piecewise linear functions.

    Attributes:
        keyword (str): The keyword for the vector discretization class.
        scalar_discr (pg.Discretization): The scalar discretization class.

    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        self.scalar_discr: pg.PwLinears
        super().__init__(keyword, pg.PwLinears)
