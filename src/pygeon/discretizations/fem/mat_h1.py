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
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.VecLagrange1 = pg.VecLagrange1(keyword)

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the advection matrix for Lagrange1 finite
        elements, which is given by
        :math:`(\boldsymbol{\beta} \cdot \nabla u, v)_\Omega`, for
        :math:`u,v \in \mathbb{L}_1(\Omega)`.

        The data dictionary contains the vector field :math:`\boldsymbol{\beta}`
        accessible via pg.VECTOR-FIELD. It is a given as a vector field, assumed
        constant per cell :math:`\in [\mathbb{P}_0(\Omega)]^d`. If not provided, it
        defaults to :math:`(0, 0, 0)`.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict | None): Optional data for scaling, in particular
            pg.VECTOR-FIELD (advection velocity field).

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

        Returns:
            pg.Discretization: The range discretization class.
        """
        raise NotImplementedError
