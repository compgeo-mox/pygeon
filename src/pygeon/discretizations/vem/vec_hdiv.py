"""Module for the discretizations of the H(div) space."""

from typing import Type

import pygeon as pg


class VecVRT0(pg.VecDiscretization):
    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector virtual RT0 discretization class.
        The base discretization class is pg.VRT0.

        We are considering the following structure of the stress tensor in 2d

        sigma = [[sigma_xx, sigma_xy],
                 [sigma_yx, sigma_yy]]

        which is represented in the code unrolled row-wise as a vector of length 4

        sigma = [sigma_xx, sigma_xy,
                 sigma_yx, sigma_yy]

        While in 3d the stress tensor can be written as

        sigma = [[sigma_xx, sigma_xy, sigma_xz],
                 [sigma_yx, sigma_yy, sigma_yz],
                 [sigma_zx, sigma_zy, sigma_zz]]

        where its vectorized structure of length 9 is given by

        sigma = [sigma_xx, sigma_xy, sigma_xz,
                 sigma_yx, sigma_yy, sigma_yz,
                 sigma_zx, sigma_zy, sigma_zz]

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
