"""Module for the matrix-valued discretization class."""

from typing import cast

import scipy.sparse as sps

import pygeon as pg


class MatDiscretization(pg.VecDiscretization):
    """
    Base class for matrix-valued discretizations.
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
        :math:`\sigma \in` :class:`pg.MatDiscretization` (matrix-valued), which is
        given by :math:`(A \sigma, \tau)_\Omega` where

        .. math::

            A \sigma = \frac{1}{2\mu} \left[ \sigma - c
            \text{Tr}(\sigma) I\right]

        with :math:`\mu` and :math:`\lambda` the Lamé constants and

        .. math::

            c = \frac{\lambda}{2\mu + d \lambda}

        where :math:`d` is the dimension. Both :math:`\sigma` and :math:`\tau`
        are in :class:`pg.MatDiscretization`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Data for the assembly.

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
        :math:`\sigma \in` :class:`pg.MatDiscretization`, which is given by
        :math:`(A \sigma, \tau)_\Omega` where

        .. math::

            A \sigma = \frac{1}{2\mu} \left( \sigma
            - \frac{1}{d} \text{Tr}(\sigma) I \right)

        with :math:`\mu` the shear Lamé constant. Both :math:`\sigma` and
        :math:`\tau` are in :class:`pg.MatDiscretization`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Data for the assembly.

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
        :math:`\mu_c` the Cosserat couple modulus, and

        .. math::

            c = \frac{\lambda}{2\mu + d \lambda}

        where :math:`d` is the dimension.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Data for the assembly.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """
        method_name = "assemble_mass_matrix_cosserat"
        return self._apply_pwpolynomials_method(sd, method_name, data)

    def assemble_lumped_matrix_elasticity(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the lumped elasticity matrix for the given grid. This is an
        approximation by a quadrature rule of :math:`(A \sigma, \tau)` where :math:`A`
        is the elasticity compliance operator from
        :meth:`pg.MatDiscretization.assemble_mass_matrix_elasticity`.

        Args:
            sd (pg.Grid): The grid.
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
        Assembles the lumped Cosserat matrix for the given grid. This is an
        approximation by a quadrature rule of :math:`(A \sigma, \tau)` where :math:`A`
        is the Cosserat compliance operator from
        :meth:`pg.MatDiscretization.assemble_mass_matrix_cosserat`.

        Args:
            sd (pg.Grid): The grid.
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
        Assembles the asymmetry matrix, mapping :math:`\sigma` to
        :math:`\sigma_{21} - \sigma_{12}` in 2D and to
        :math:`(\sigma_{32} - \sigma_{23}, \sigma_{13} - \sigma_{31},
        \sigma_{21} - \sigma_{12})` in 3D, see
        :meth:`pg.MatPwPolynomials.assemble_asym_matrix`.

        The operator is obtained by projecting to the matrix piecewise polynomials and
        applying their asymmetry matrix.

        Args:
            sd (pg.Grid): The grid.
            as_pwconstant (bool): If True, the range is projected to the piecewise
                constants, otherwise it is in the piecewise polynomials of the same
                order. Defaults to False.

        Returns:
            sps.csc_array: The assembled asymmetry matrix.
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
