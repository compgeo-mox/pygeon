"""Module for the discretizations of the vector H1 space."""

from typing import Type, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecLagrange(pg.VecDiscretization):
    r"""
    Abstract class collecting the features shared by the vector Lagrange finite element
    discretizations of degree k: the operators of linear elasticity, where the
    symmetric gradient and the divergence map to the matrix and scalar piecewise
    polynomials of degree k - 1.

    Convention for the ordering is first all the x, then all the y, and (if dim = 3)
    all the z.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4, and in 3D, it has a length of 9.

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

    The strain tensor follows the same approach.
    """

    tensor_order = pg.VECTOR
    """Vector-valued discretization"""

    base_discr: pg.Lagrange  # To please mypy

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Returns the divergence matrix operator :math:`\nabla \cdot u`, mapping to the
        scalar piecewise polynomials of degree k - 1.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The div matrix obtained from the discretization.
        """
        return self.assemble_broken_div_matrix(sd)

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Returns the symmetric gradient matrix operator
        :math:`\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)`, mapping to the
        matrix piecewise polynomials of degree k - 1.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The sparse symmetric gradient matrix operator.
        """
        grad = self.assemble_broken_grad_matrix(sd)
        grad_space = cast(
            pg.MatPwPolynomials, self.get_broken_grad_discr()(self.keyword)
        )
        sym = grad_space.assemble_symmetrizing_matrix(sd)

        return sym @ grad

    def assemble_stiff_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the linear elasticity stiffness matrix

        .. math::

            2\mu (\varepsilon(u), \varepsilon(v))_\Omega
            + \lambda (\nabla \cdot u, \nabla \cdot v)_\Omega

        where :math:`\mu` and :math:`\lambda` are the Lamé parameters.

        Args:
            sd (pg.Grid): The grid on which the finite element method is defined.
            data (dict | None): The data with the Lamé parameters, accessed with the
                keys pg.LAME_MU and pg.LAME_LAMBDA.

        Returns:
            sps.csc_array: The assembled global elasticity matrix.
        """
        # Compute the two parts of the global stiffness matrix
        # We start with the symgrad-symgrad component
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        data_mu = pp.initialize_data({}, self.keyword, {pg.WEIGHT: 2 * mu})

        grad_space = self.get_broken_grad_discr()(self.keyword)
        mu_mass = grad_space.assemble_mass_matrix(sd, data_mu)

        symgrad = self.assemble_symgrad_matrix(sd)
        sym_sym = symgrad.T @ mu_mass @ symgrad

        # And now the div-div component
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)
        data_lambda = pp.initialize_data({}, self.keyword, {pg.WEIGHT: lambda_})

        scalar_space = pg.get_PwPolynomials(self.poly_order - 1, pg.SCALAR)(
            self.keyword
        )
        lambda_mass = scalar_space.assemble_mass_matrix(sd, data_lambda)

        div = self.assemble_div_matrix(sd)
        div_div = div.T @ (lambda_mass) @ div

        # Return the resulting matrix
        return (sym_sym + div_div).tocsc()

    def compute_stress(
        self,
        sd: pg.Grid,
        u: np.ndarray,
        data: dict,
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field at the cell centers as
        a (3, 3, num_cells) tensor.

        Args:
            sd (pg.Grid): The grid.
            u (np.ndarray): The displacement field.
            data (dict): Data for the computation including the Lamé parameters accessed
                with the keys pg.LAME_LAMBDA and pg.LAME_MU.
                Both float and np.ndarray are accepted.

        Returns:
            np.ndarray: The stress tensor.
        """
        # Construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        grad_space = cast(
            pg.MatPwPolynomials, self.get_broken_grad_discr()(self.keyword)
        )
        proj = grad_space.eval_at_cell_centers(sd)
        ident = grad_space.assemble_trace_matrix(sd).T

        # Retrieve Lamé parameters, repeated for each dof of the matrix space
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)
        lambda_ = pg.get_cell_data(sd, data, self.keyword, pg.LAME_LAMBDA)

        num_reps = symgrad.shape[0] // sd.num_cells
        mu = np.tile(mu, num_reps)
        lambda_ = np.tile(lambda_, num_reps)

        # Combine all the terms
        sigma = 2 * mu * (symgrad @ u) + lambda_ * (ident @ div @ u)

        # Reshape and return
        return (proj @ sigma).reshape((pg.AMBIENT_DIM, pg.AMBIENT_DIM, -1))

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no range discretization for the vector
                Lagrangian in PyGeoN.
        """
        raise NotImplementedError(
            f"There's no range discr for the vector Lagrangian {self.poly_order} in "
            "PyGeoN"
        )


class VecLagrange1(VecLagrange):
    """
    Vector Lagrange finite element discretization of degree 1, with the dofs at the
    nodes.
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
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr = pg.Lagrange1(keyword)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        # Avoid division by zero for 0D grids (and handle the no-dof case consistently).
        if sd.dim == 0 or self.ndof(sd) == 0:
            return sps.csc_array((pg.AMBIENT_DIM, 0))

        Pi = super().eval_at_cell_centers(sd)

        # We need to map back from reference to physical coordinates
        R = sps.kron(sd.rotation_matrix.T, sps.eye_array(Pi.shape[0] // sd.dim), "csc")

        return (R @ Pi).tocsc()

    def assemble_adv_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the advection matrix for the vector Lagrange1 finite
        elements, which is given component-wise by
        :math:`(\boldsymbol{\beta} \cdot \nabla \boldsymbol{u}, \boldsymbol{v})_\Omega`,
        for :math:`\boldsymbol{u}, \boldsymbol{v} \in [\mathbb{L}_1(\Omega)]^d`.

        The data dictionary contains the vector field :math:`\boldsymbol{\beta}`
        accessible via pg.VECTOR_FIELD. It is given as a vector field, assumed
        constant per cell :math:`\in [\mathbb{P}_0(\Omega)]^d`. If not provided, it
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


class VecLagrange2(VecLagrange):
    """
    Vector Lagrange finite element discretization of degree 2, with the dofs at the
    nodes and at the edges.
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
                Defaults to pg.UNITARY_DATA.
        """
        super().__init__(keyword)
        self.base_discr: pg.Lagrange2 = pg.Lagrange2(keyword)
