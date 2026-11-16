"""Module for the discretizations of the vector L2 space."""

from __future__ import annotations

import pygeon as pg


class VecPwMConstants(pg.VDiscretization, pg.VecPwConstants):
    """
    A class representing the vector-valued piecewise constant monomials.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwMConstants.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr = pg.PwMConstants(keyword)


class VecPwMLinears(pg.VDiscretization, pg.VecPwLinears):
    """
    A class representing the vector-valued piecewise linear monomials
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.PwMLinears.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr = pg.PwMLinears(keyword)
