"""Module for the discretizations of the H(div) space."""

from typing import Type

import pygeon as pg


class VecVRT0(pg.VecDiscretization):
    """
    VecVRT0 is a tensor-valued discretization class for the virtual Raviart-Thomas RT0
    element.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector virtual RT0 discretization class.
        The base discretization class is pg.VRT0.

        We are considering the following structure of the stress tensor in 2D:

        .. math::

            \\sigma = \\begin{bmatrix}
                \\sigma_{xx} & \\sigma_{xy} \\\\
                \\sigma_{yx} & \\sigma_{yy}
            \\end{bmatrix}

        which is represented in the code unrolled row-wise as a vector of length 4:

        .. math::

            \\sigma = [\\sigma_{xx}, \\sigma_{xy}, \\sigma_{yx}, \\sigma_{yy}]

        While in 3D the stress tensor can be written as:

        .. math::

            \\sigma = \\begin{bmatrix}
                \\sigma_{xx} & \\sigma_{xy} & \\sigma_{xz} \\\\
                \\sigma_{yx} & \\sigma_{yy} & \\sigma_{yz} \\\\
                \\sigma_{zx} & \\sigma_{zy} & \\sigma_{zz}
            \\end{bmatrix}

        where its vectorized structure of length 9 is given by:

        .. math::

            \\sigma = [\\sigma_{xx}, \\sigma_{xy}, \\sigma_{xz},
                       \\sigma_{yx}, \\sigma_{yy}, \\sigma_{yz},
                       \\sigma_{zx}, \\sigma_{zy}, \\sigma_{zz}]

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.VRT0 = pg.VRT0(keyword)

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.VecPwConstants
