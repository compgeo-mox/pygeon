"""Module for the discretizations of the matrix H1 space."""

from typing import Type

import pygeon as pg


class MatLagrange1(pg.VecDiscretization):
    poly_order = 1
    tensor_order = pg.MATRIX

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.VecLagrange1.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecLagrange1(keyword)  # type: ignore[assignment]

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of
                the differential.

        Raises:
            NotImplementedError: If there is no range discretization for the matrix
                Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the matrix Lagrangian 1 in PyGeoN"
        )
