"""Module for the matrix-valued discretization class."""

from typing import cast

import scipy.sparse as sps

import pygeon as pg


class MatDiscretization(pg.VecDiscretization):
    """
    Base class for matrix-valued discretizations. This class provides methods for
    assembling mass matrices, trace matrices, asymmetric matrices, and lumped matrices.
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
        r"""
        Assembles and returns the elasticity inner product matrix for
        :math:`\sigma \in` :class:`MatDiscretization` (matrix-valued), which is
        given by :math:`(A \sigma, \tau)_\Omega` where

        .. math::

            A \sigma = \frac{1}{2\mu} \left[ \sigma - c
            \text{Tr}(\sigma) I\right]

        with :math:`\mu` and :math:`\lambda` the Lamé constants and

        .. math::

            c = \frac{\lambda}{2\mu + d \lambda}

        where :math:`d` is the dimension. Both :math:`\sigma` and :math:`\tau`
        are in :class:`MatDiscretization`.

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
        r"""
        Assembles and returns the mass matrix for an incompressible material for
        :math:`\sigma \in` :class:`MatDiscretization`, which is given by
        :math:`(A \sigma, \tau)_\Omega` where

        .. math::

            A \sigma = \frac{1}{2\mu} \left( \sigma
            - \frac{1}{d} \text{Tr}(\sigma) I \right)

        with :math:`\mu` the shear Lamé constant. Both :math:`\sigma` and
        :math:`\tau` are in :class:`MatDiscretization`.

        Args:
            sd (pg.Grid): The grid.
            data (dict): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        method_name = "assemble_deviator_matrix"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_mass_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles and returns the Cosserat inner product, which is given by
        :math:`(A \sigma, \tau)` where

        .. math::

            A \sigma = \frac{1}{2\mu} \left( \text{sym}(\sigma)
            - c \text{Tr}(\sigma) I \right)
            + \frac{1}{2\mu_c} \text{skw}(\sigma)

        with :math:`\mu` and :math:`\lambda` the Lamé constants,
        :math:`\mu_c` the coupling Lamé modulus, and

        .. math::

            c = \frac{\lambda}{2\mu + d \lambda}

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
        r"""
        Assembles the lumped elasticity matrix for the given grid. This is a diagonal
        approximation of :math:`(A \sigma, \tau)` where :math:`A` is the elasticity
        compliance operator from :meth:`assemble_mass_matrix_elasticity`.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped elasticity matrix.
        """
        method_name = "assemble_lumped_matrix_elasticity"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_lumped_matrix_cosserat(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the lumped Cosserat matrix for the given grid. This is a diagonal
        approximation of :math:`(A \sigma, \tau)` where :math:`A` is the Cosserat
        compliance operator from :meth:`assemble_mass_matrix_cosserat`.

        Args:
            sd (pg.Grid): The grid object.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped Cosserat matrix.
        """
        method_name = "assemble_lumped_matrix_cosserat"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_asym_matrix(
        self, sd: pg.Grid, as_pwconstant: bool = False
    ) -> sps.csc_array:
        r"""
        Assembles the skew-symmetry (asymmetry) matrix
        :math:`\text{skw}(\sigma) = \frac{1}{2}(\sigma - \sigma^T)`.

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
        r"""
        Assembles and returns the trace matrix :math:`\text{Tr}(\sigma)`.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        P = self.proj_to_PwPolynomials(sd)
        pwp = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        pwp = cast(pg.MatPwPolynomials, pwp)
        trace = pwp.assemble_trace_matrix(sd)
        return trace @ P
