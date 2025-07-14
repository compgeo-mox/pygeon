"""Module for the discretizations of the vector L2 space."""

from typing import Callable, Optional, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VecPieceWisePolynomial(pg.VecDiscretization):
    """
    A class representing an abstract vector piecewise polynomial discretization.
    """

    def local_dofs_of_cell(
        self, sd: pg.Grid, c: int, ambient_dim: int = -1
    ) -> np.ndarray:
        """
        Compute the local degrees of freedom (DOFs) of a cell in a vector-valued
        finite element discretization.

        Args:
            sd (pg.Grid): The grid object representing the discretization domain.
            c (int): The index of the cell for which the local DOFs are to be computed.
                ambient_dim (int, optional): The ambient dimension of the space. If not
                provided, it defaults to the dimension of the grid (`sd.dim`).

        Returns:
            np.ndarray: An array containing the local DOFs of the specified cell,
                adjusted for the vector-valued nature of the discretization.
        """
        if ambient_dim == -1:
            ambient_dim = sd.dim

        n_base = self.base_discr.ndof(sd)

        dof_base = self.base_discr.local_dofs_of_cell(sd, c)  # type: ignore[attr-defined]
        shift = np.repeat(n_base * np.arange(ambient_dim), dof_base.size)

        dof_base = np.tile(dof_base, ambient_dim)

        return dof_base + shift

    def ndof_per_cell(self, sd: pg.Grid) -> int:
        """
        Computes the number of degrees of freedom (DOF) per cell for the given grid.

        This method calculates the total number of DOFs per cell by multiplying
        the number of DOFs per cell from the base discretization by the spatial
        dimension of the grid.

        Args:
            sd (pg.Grid): The grid object representing the spatial discretization.

        Returns:
            int: The total number of degrees of freedom per cell.
        """
        return self.base_discr.ndof_per_cell(sd) * sd.dim  # type: ignore[attr-defined]

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class for the range of the differential.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The discretization class for the range of the
            differential.
        """
        return self.base_discr.get_range_discr_class(dim)

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
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.PwConstants = pg.PwConstants(keyword)

    def proj_to_pwLinears(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the projection matrix to the vector piecewise linear space.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_pwLinears(sd)
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

            err2_dim = self.base_discr.error_l2(
                sd, num_sol_dim, ana_sol_dim, relative, etype
            )
            err2 += np.square(err2_dim)
        return np.sqrt(err2)


class VecPwLinears(VecPieceWisePolynomial):
    """
    A class representing the discretization using vector piecewise linear functions.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.PwLinears = pg.PwLinears(keyword)

    def proj_to_pwConstants(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a piece-wise vector function to a piecewise
        vector constant function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        proj = self.base_discr.proj_to_pwConstants(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def proj_to_pwQuadratics(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the vector P1 discretization to the vector P2 discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_pwQuadratics(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()


class VecPwQuadratics(VecPieceWisePolynomial):
    """
    A class representing the discretization using vector piecewise quadratic functions.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.PwQuadratics = pg.PwQuadratics(keyword)
