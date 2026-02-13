"""Module for the discretizations of the H(div) space."""

from typing import Type, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VecHDiv(pg.VecDiscretization):
    """Base class for vector-valued discretizations in the H(div) space.
    This class provides methods for assembling mass matrices, trace matrices,
    asymmetric matrices, and lumped matrices for vector-valued finite element
    discretizations in the H(div) space.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order = pg.MATRIX
    """Matrix-valued discretization"""

    def _apply_pwpolynomials_method(
        self, sd: pg.Grid, method_name: str, *args, **kwargs
    ) -> sps.csc_array:
        """
        Generic helper to apply a PwPolynomials method with projection.

        This method projects to PwPolynomials space, calls the specified method,
        and returns the result with projection applied: P.T @ result @ P

        Args:
            sd (pg.Grid): The grid.
            method_name (str): Name of the method to call on PwPolynomials.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            sps.csc_array: P.T @ result @ P where result is from the PwPolynomials
            method.
        """
        P = self.proj_to_PwPolynomials(sd)
        pwp = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        method = getattr(pwp, method_name)
        result = method(sd, *args, **kwargs)
        return P.T @ result @ P

    def assemble_mass_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the elasticity inner product matrix, which is given by
        :math:`(A \\sigma, \\tau)` where

        .. math::

            A \\sigma = \\frac{1}{2\\mu} \\left[ \\sigma - c
            \\text{Tr}(\\sigma) I\\right]

        with :math:`\\mu` and :math:`\\lambda` the Lamé constants and

        .. math::

            c = \\frac{\\lambda}{2\\mu + d \\lambda}

        where :math:`d` is the dimension.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        method_name = "assemble_mass_matrix_elasticity"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_deviator_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the mass matrix for vector BDM1 for an incompressible
        material, which is given by (A sigma, tau) where
        A sigma = (sigma - coeff * Trace(sigma) * I) / (2 mu)
        with mu the Lamé constants and coeff = 1 / dim

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        mu = pg.get_cell_data(sd, data, self.keyword, pg.LAME_MU)

        param = {pg.LAME_LAMBDA: np.inf, pg.LAME_MU: mu}
        data_ = pp.initialize_data({}, self.keyword, param)

        return self.assemble_mass_matrix_elasticity(sd, data_)

    def assemble_mass_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles and returns the Cosserat inner product, which is given by
        :math:`(A \\sigma, \\tau)` where

        .. math::

            A \\sigma = \\frac{1}{2\\mu} \\left( \\text{sym}(\\sigma)
            - c \\text{Tr}(\\sigma) I \\right)
            + \\frac{1}{2\\mu_c} \\text{skw}(\\sigma)

        with :math:`\\mu` and :math:`\\lambda` the Lamé constants,
        :math:`\\mu_c` the coupling Lamé modulus, and

        .. math::

            c = \\frac{\\lambda}{2\\mu + d \\lambda}

        where :math:`d` is the dimension.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        method_name = "assemble_mass_matrix_cosserat"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_lumped_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        method_name = "assemble_lumped_matrix_elasticity"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_lumped_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix with Cosserat terms for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        method_name = "assemble_lumped_matrix_cosserat"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_asym_matrix(
        self, sd: pg.Grid, as_pwconstant: bool = False
    ) -> sps.csc_array:
        """
        Assemble the asymmetric matrix for the given grid.

        This method constructs an asymmetric matrix by projecting to
        matrix piecewise polynomials and combining it with the
        discretization's asymmetric matrix.

        Args:
            sd (pg.Grid): The grid object representing the spatial discretization.
            as_pwconstant (bool): Compute the operator with the range on the piece-wise
                polynomials (default), otherwise the mapping is on the piece-wise
                constant.

        Returns:
            sps.csc_array: The assembled asymmetric matrix in compressed sparse column
            format.
        """
        P = self.proj_to_PwPolynomials(sd)
        mat_discr = pg.get_PwPolynomials(self.poly_order, pg.MATRIX)(self.keyword)
        mat_discr = cast(pg.MatPwLinears | pg.MatPwQuadratics, mat_discr)
        asym = mat_discr.assemble_asym_matrix(sd) @ P

        if as_pwconstant:
            tensor_order = sd.dim - 2

            range_space = pg.get_PwPolynomials(self.poly_order, tensor_order)(
                self.keyword
            )
            P0 = pg.proj_to_PwPolynomials(range_space, sd, 0)

            asym = P0 @ asym

        return asym

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the vector HDiv.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.

        Note:
            This method should be implemented in subclasses.
        """
        P = self.proj_to_PwPolynomials(sd)
        pwp = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        trace = pwp.assemble_trace_matrix(sd)
        return trace @ P


class VecBDM1(VecHDiv):
    """
    VecBDM1 is a class that represents the vector BDM1 (Brezzi-Douglas-Marini) finite
    element method. It provides methods for assembling matrices like the mass matrix,
    the trace matrix, the asymmetric matrix and the differential matrix. It also
    provides methods for evaluating the solution at cell centers, interpolating a given
    function onto the grid, assembling the natural boundary condition term, and more.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector BDM1 discretization class.
        The base discretization class is pg.BDM1.

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
        self.base_discr: pg.BDM1 = pg.BDM1(keyword)

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = self.base_discr.proj_to_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        proj = self.base_discr.proj_from_RT0(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential

        Args:
            dim (int): The dimension of the range.

        Returns:
            pg.Discretization: The discretization class containing the range of the
            differential
        """
        return pg.VecPwConstants


class VecRT0(VecHDiv):
    """
    VecRT0 is a tensor-valued discretization class for the Raviart-Thomas RT0 finite
    element, specialized for handling stress tensors in 2D and 3D.
    This class provides methods for assembling trace and asymmetric matrices
    for vector RT0 discretizations, as well as retrieving the appropriate range
    discretization class.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT0 discretization class.
        The base discretization class is pg.RT0.

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
        self.base_discr: pg.RT0 = pg.RT0(keyword)

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.VecPwConstants


class VecRT1(VecHDiv):
    """
    VecRT1 is a vector Raviart-Thomas finite element discretization class of order 1.

    This class is designed for matrix-valued finite element discretizations in the
    H(div) space, specifically using the Raviart-Thomas elements of order 1 (RT1).
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector RT1 discretization class.
        The base discretization class is pg.RT1.

        Args:
            keyword (str): The keyword for the vector discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr: pg.RT1 = pg.RT1(keyword)

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.VecPwLinears
