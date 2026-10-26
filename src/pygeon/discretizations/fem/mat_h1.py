"""Module for the discretizations of the matrix-valued H1 space."""

from typing import Type

import scipy.sparse as sps

import pygeon as pg


class MatLagrange1(pg.MatDiscretization):
    """
    MatLagrange1 is a matrix-valued H1-conforming finite element space of order 1.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix-valued Lagrange1 discretization class.
        The base discretization class is pg.VecLagrange1.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.VecLagrange1 = pg.VecLagrange1(keyword)

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the advection matrix for the matrix-valued Lagrange1
        finite elements, which is given component-wise by
        :math:`(\boldsymbol{\beta} \cdot \nabla u, v)_\Omega`, for each component
        :math:`u, v` in :class:`pg.Lagrange1`.

        The data dictionary contains the vector field :math:`\boldsymbol{\beta}`
        accessible via pg.VECTOR_FIELD. It is given as a vector field, assumed
        constant per cell (:class:`pg.VecPwConstants`). If not provided, it
        defaults to :math:`(0, 0, 0)`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data for scaling, in particular
                pg.VECTOR_FIELD (advection velocity field).

        Returns:
            sps.csc_array: The assembled advection matrix.
        """
        adv_matrix = self.base_discr.assemble_adv_matrix(sd, data)
        return self.vectorize(sd.dim, adv_matrix)

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no range discretization for the matrix
                Lagrange1 in PyGeoN.
        """
        raise NotImplementedError
