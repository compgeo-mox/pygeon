"""Module for the discretizations of the vector L2 space."""

from __future__ import annotations

from typing import Callable, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class VecPwPolynomials(pg.VecDiscretization):
    """
    A class representing an abstract vector piecewise polynomial discretization.
    """

    poly_order = None
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    base_discr: pg.PwPolynomials | pg.VecPwPolynomials

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

        dof_base = self.base_discr.local_dofs_of_cell(sd, c)
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
        return self.base_discr.ndof_per_cell(sd) * sd.dim

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

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to +1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_higher_PwPolynomials(sd)
        return self.vectorize(sd.dim, proj)

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Projects the discretization to -1 order discretization.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_to_lower_PwPolynomials(sd)
        return self.vectorize(sd.dim, proj)


class VecPwConstants(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise constant functions.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

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
        self.base_discr = pg.PwConstants(keyword)


class VecPwLinears(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise linear functions.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

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
        self.base_discr = pg.PwLinears(keyword)


class VecPwQuadratics(VecPwPolynomials):
    """
    A class representing the discretization using vector piecewise quadratic functions.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

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
        self.base_discr = pg.PwQuadratics(keyword)
