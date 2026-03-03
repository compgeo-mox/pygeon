"""Module for the discretizations of the vector H1 space."""

from typing import Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecLagrange1(pg.VecDiscretization):
    """
    Vector Lagrange finite element discretization for H1 space.

    This class represents a finite element discretization for the H1 space using
    vector Lagrange elements. It provides methods for assembling various matrices
    and operators, such as the mass matrix, divergence matrix, symmetric gradient
    matrix, and more.

    Convention for the ordering is first all the x, then all the y, and (if dim = 3)
    all the z.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4, and in 3D, it has a length of 9.

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

    The strain tensor follows the same approach.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    base_discr: pg.Lagrange1  # To please mypy

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.Lagrange1.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.Lagrange1(keyword)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The div matrix obtained from the discretization.
        """
        return self.assemble_broken_div_matrix(sd)

    def assemble_div_div_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Returns the div-div matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by the Lame' parameter lambda.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Additional data, the Lame' parameter lambda.
                Defaults to None.

        Returns:
            csc_array: Sparse (sd.num_nodes, sd.num_nodes) Div-div matrix obtained from
            the discretization.
        """
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        p0 = pg.PwConstants(self.keyword)

        div = self.assemble_div_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)

        return div.T @ (lambda_ * mass) @ div

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the symmetric gradient matrix operator for the
        lowest order vector Lagrange element

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_array: The sparse symmetric gradient matrix operator.
        """
        grad = self.assemble_broken_grad_matrix(sd)
        sym = pg.MatPwConstants().assemble_symmetrizing_matrix(sd)

        return sym @ grad

    def assemble_symgrad_symgrad_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Returns the symgrad-symgrad matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by twice the Lame' parameter
        mu.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Additional data, the Lame' parameter mu. Defaults to
                None.

        Returns:
            sps.csc_array: Sparse symgrad-symgrad matrix of shape (sd.num_nodes,
            sd.num_nodes). The matrix obtained from the discretization.
        """
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        data_mu = pp.initialize_data({}, self.keyword, {pg.WEIGHT: 2 * mu})

        matp0 = pg.MatPwConstants(self.keyword)

        symgrad = self.assemble_symgrad_matrix(sd)
        mass = matp0.assemble_mass_matrix(sd, data_mu)

        return symgrad.T @ mass @ symgrad

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        div = self.assemble_div_matrix(sd)
        symgrad = self.assemble_symgrad_matrix(sd)

        return sps.block_array([[symgrad], [div]]).tocsc()

    def assemble_elasticity_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the elasticity matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid on which the finite element method is defined.
            data (dict | None): Additional data required for the assembly process.

        Returns:
            sps.csc_array: The assembled global elasticity matrix.
        """
        # compute the two parts of the global stiffness matrix
        sym_sym = self.assemble_symgrad_symgrad_matrix(sd, data)
        div_div = self.assemble_div_div_matrix(sd, data)

        # return the global stiffness matrix
        return sym_sym + div_div

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of
            the differential.

        Raises:
            NotImplementedError: If there is no range discretization for the vector
            Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector Lagrangian 1 in PyGeoN"
        )

    def compute_stress(
        self,
        sd: pg.Grid,
        u: np.ndarray,
        data: dict,
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field.

        Args:
            sd (pg.Grid): The spatial discretization object.
            u (ndarray): The displacement field.
            data (dict): Data for the computation including the Lame parameters accessed
                with the keys pg.LAME_LAMBDA and pg.LAME_MU.
                Both float and np.ndarray are accepted.

        Returns:
            ndarray: The stress tensor.
        """
        # construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        p0 = pg.PwConstants(self.keyword)
        proj = p0.eval_at_cell_centers(sd)

        # retrieve Lamé parameters
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        mu = np.tile(mu, np.square(sd.dim))
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        # compute the two terms and split on each component
        sigma = np.array(np.split(2 * mu * (symgrad @ u), np.square(sd.dim)))
        sigma[:: (sd.dim + 1)] += lambda_ * (div @ u)

        # compute the actual dofs
        sigma = sigma @ proj

        # create the indices to re-arrange the components for the second
        # order tensor
        idx = np.arange(np.square(sd.dim)).reshape((sd.dim, -1), order="F")

        return sigma[idx].T


class VecLagrange2(pg.VecDiscretization):
    """
    VecLagrange2 is a vector discretization class that extends the functionality of
    the pg.VecDiscretization base class. It utilizes the pg.Lagrange2 scalar
    discretization class for its operations.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The base discretization class is pg.Lagrange2.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.Lagrange2 = pg.Lagrange2(keyword)

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of the
            differential.

        Raises:
            NotImplementedError: If there is no range discretization for the vector
            Lagrangian 2 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector Lagrangian 2 in PyGeoN"
        )
