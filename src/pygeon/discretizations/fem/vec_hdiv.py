"""Module for the discretizations of the matrix-valued H(div) space."""

from typing import Type

import scipy.sparse as sps

import pygeon as pg


class VecBDM1(pg.MatDiscretization):
    """
    Matrix-valued finite element discretization with each row in :class:`pg.BDM1`.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        r"""
        Initialize the vector BDM1 discretization class.
        The base discretization class is pg.BDM1.

        We are considering the following structure of the stress tensor in 2D:

        .. math::

            \sigma = \begin{bmatrix}
                \sigma_{xx} & \sigma_{xy} \\
                \sigma_{yx} & \sigma_{yy}
            \end{bmatrix}

        which is represented in the code unrolled row-wise as a vector of length 4:

        .. math::

            \sigma = [\sigma_{xx}, \sigma_{xy}, \sigma_{yx}, \sigma_{yy}]

        While in 3D the stress tensor can be written as:

        .. math::

            \sigma = \begin{bmatrix}
                \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
                \sigma_{yx} & \sigma_{yy} & \sigma_{yz} \\
                \sigma_{zx} & \sigma_{zy} & \sigma_{zz}
            \end{bmatrix}

        where its vectorized structure of length 9 is given by:

        .. math::

            \sigma = [\sigma_{xx}, \sigma_{xy}, \sigma_{xz},
                       \sigma_{yx}, \sigma_{yy}, \sigma_{yz},
                       \sigma_{zx}, \sigma_{zy}, \sigma_{zz}]

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.BDM1 = pg.BDM1(keyword)

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = self.base_discr.proj_to_RT0(sd)
        return sps.kron(sps.eye_array(sd.dim), proj).tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Include the vector RT0 finite element space in the vector BDM1 one.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_from_RT0(sd)
        return sps.kron(sps.eye_array(sd.dim), proj).tocsc()

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            Type[pg.Discretization]: The range discretization class.
        """
        return pg.VecPwConstants


class VecRT0(pg.MatDiscretization):
    """
    Matrix-valued finite element discretization with each row in :class:`pg.RT0`.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        r"""
        Initialize the vector RT0 discretization class.
        The base discretization class is pg.RT0.

        We are considering the following structure of the stress tensor in 2D:

        .. math::

            \sigma = \begin{bmatrix}
                \sigma_{xx} & \sigma_{xy} \\
                \sigma_{yx} & \sigma_{yy}
            \end{bmatrix}

        which is represented in the code unrolled row-wise as a vector of length 4:

        .. math::

            \sigma = [\sigma_{xx}, \sigma_{xy}, \sigma_{yx}, \sigma_{yy}]

        While in 3D the stress tensor can be written as:

        .. math::

            \sigma = \begin{bmatrix}
                \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
                \sigma_{yx} & \sigma_{yy} & \sigma_{yz} \\
                \sigma_{zx} & \sigma_{zy} & \sigma_{zz}
            \end{bmatrix}

        where its vectorized structure of length 9 is given by:

        .. math::

            \sigma = [\sigma_{xx}, \sigma_{xy}, \sigma_{xz},
                       \sigma_{yx}, \sigma_{yy}, \sigma_{yz},
                       \sigma_{zx}, \sigma_{zy}, \sigma_{zz}]

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.RT0 = pg.RT0(keyword)

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            Type[pg.Discretization]: The range discretization class.
        """
        return pg.VecPwConstants


class VecRT1(pg.MatDiscretization):
    """
    Matrix-valued finite element discretization with each row in :class:`pg.RT1`.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT1 discretization class.
        The base discretization class is pg.RT1.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.RT1 = pg.RT1(keyword)

    def get_range_discr_class(self, _dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            Type[pg.Discretization]: The range discretization class.
        """
        return pg.VecPwLinears
