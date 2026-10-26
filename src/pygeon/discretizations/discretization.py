"""Module for the discretization class."""

from __future__ import annotations

import abc
from math import comb
from typing import Callable, Type, cast

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class Discretization(abc.ABC):
    """
    Abstract class for PyGeoN discretization methods.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order: int
    """Tensor order of the basis functions"""

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the Discretization object.

        Args:
            keyword (str): The keyword used to identify the discretization method.
                Defaults to pg.UNITARY_DATA.
        """
        self.keyword = keyword

    def __repr__(self) -> str:
        """
        Return a string representation of the Discretization object.

        The string includes the type of the discretization and, if available,
        the keyword associated with it.

        Returns:
            str: A string representation of the Discretization object.
        """
        s = f"Discretization of type {self.__class__.__name__}"
        if hasattr(self, "keyword"):
            s += f" with keyword {self.keyword}"
        return s

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method, given by
        the dofs per entity contracted with the number of entities of the grid.

        Spaces whose dofs are not shared between neighboring elements, such as the
        discontinuous ones, override this method.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            int: The number of degrees of freedom.
        """
        return int(self.ndof_per_entity(sd.dim) @ sd.num_entities())

    @abc.abstractmethod
    def ndof_per_entity(self, dim: int) -> np.ndarray:
        """
        Returns the number of degrees of freedom of a single element, ordered by
        the dimension of the entity they are associated with as [0, 1, 2, 3].

        The entries count all the dofs of the element that lie on entities of a
        given dimension, so that their sum is the number of dofs per element. The
        implementations assume a simplicial element.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: The number of degrees of freedom per entity, of size 4.
        """

    def ndof_per_element(self, dim: int) -> int:
        """
        Returns the number of degrees of freedom of a single element, obtained by
        contracting the dofs per entity with the number of entities of a simplex
        of dimension dim.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            int: The number of degrees of freedom per element.
        """
        num_entities = np.array([comb(dim + 1, k + 1) for k in range(4)])
        return int(self.ndof_per_entity(dim) @ num_entities)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the mass matrix by projecting to the corresponding piecewise
        polynomial space.

        For a discretization space :math:`V_h`, the matrix represents the
        bilinear form

        .. math::

            (u, v)_\Omega, \quad u, v \in V_h,

        optionally weighted by physical parameters in ``data``. Both the domain
        and the range of this operator lie in :math:`V_h`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The mass matrix.
        """
        return self._assemble_inner_product(sd, data, "assemble_mass_matrix")

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the lumped mass matrix using the corresponding piecewise polynomial
        space.

        For a discretization space :math:`V_h`, the lumped matrix approximates the mass
        matrix :math:`(u, v)_\Omega,\ u, v \in V_h` with the lumped matrix of the
        piecewise polynomials, obtained with a quadrature rule, transferred to
        :math:`V_h` through :meth:`pg.Discretization.proj_to_PwPolynomials`. Both the
        domain and the range lie in :math:`V_h`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Dictionary with physical parameters for scaling.

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        return self._assemble_inner_product(sd, data, "assemble_lumped_matrix")

    def _assemble_inner_product(
        self, sd: pg.Grid, data: dict | None, pwp_method: str
    ) -> sps.csc_array:
        """
        General function to assemble an inner product matrix using the piecewise
        polynomial space.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Dictionary with physical parameters for scaling.
            pwp_method (str): Assembly method of the piecewise polynomial space.

        Returns:
            sps.csc_array: The inner product matrix.
        """
        # If there are no degrees of freedom, we return an empty matrix.
        if self.ndof(sd) == 0:
            return sps.csc_array((0, 0))

        # We project to the piecewise polynomials and use that mass matrix.
        Pi = self.proj_to_PwPolynomials(sd)
        pwp = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        M = getattr(pwp, pwp_method)(sd, data)

        return (Pi.T @ M @ Pi).tocsc()

    @abc.abstractmethod
    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        The matrix :math:`D` represents the discrete exterior derivative,
        mapping from the current finite element space :math:`V_h` to its range
        space :math:`W_h` (returned by :meth:`pg.Discretization.get_range_discr_class`).

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The differential matrix.
        """

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix, which is given by :math:`B^T A B`,
        where :math:`B` is the differential matrix (from the current space
        :math:`V_h` to its range space :math:`W_h`) and :math:`A` is the
        mass matrix of the range discretization :math:`W_h`.

        This method takes a grid object ``sd`` and an optional data dictionary
        ``data`` as input. It first calls the ``assemble_diff_matrix`` method to obtain
        the differential matrix ``B``.
        Then, it creates an instance of the range discretization class using the
        ``dim`` attribute of ``sd``. Next, it calls the ``assemble_mass_matrix`` method
        of the range discretization class to obtain the mass matrix ``A``.
        Finally, it returns the product of the transpose of ``B``, ``A``, and ``B``.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data dictionary. Defaults to None.

        Returns:
            sps.csc_array: The stiffness matrix.
        """
        B = self.assemble_diff_matrix(sd)

        discr = self.get_range_discr_class(sd.dim)(self.keyword)
        A = discr.assemble_mass_matrix(sd, data)

        return (B.T @ A @ B).tocsc()

    @abc.abstractmethod
    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """

    def eval_func_at_coords(
        self, func: Callable[[np.ndarray], np.ndarray], coords: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates a function at given coordinates. We assume that func(x) evaluates
        the given function for all coordinates in the columns of x at once; constant
        scalar and vector functions are broadcast to all the coordinates.

        Args:
            func (Callable): A function that returns the function values at coordinates.
            coords (np.ndarray): A coordinate array with shape (3, n).

        Returns:
            np.ndarray: The values of the function, with the points along the last
            axis.

        Raises:
            RuntimeError: If the function is neither vectorized nor constant.
        """
        # We first simply evaluate the function at the coordinates.
        interp = np.asarray(func(coords))

        # We can handle two cases: 1) the function is vectorized and returns an array
        # with n entries, one for each column in coords. 2) It returns a constant value,
        # either scalar or vector.

        # 1) If the function is properly vectorized, it returns an array with the last
        # axis indexing the evaluation points.
        if interp.ndim > 0 and interp.shape[-1] == coords.shape[-1]:
            return interp

        # 2) For constant functions, we populate the array ourselves.
        match interp.ndim:
            # Interpolate constant scalar functions, e.g. lambda _: 1.
            case 0:
                return np.tile(interp, coords.shape[1])

            # Interpolate constant vector functions, e.g. lambda _ : np.array([1, 0, 0])
            case 1:
                if interp.shape[0] == pg.AMBIENT_DIM:
                    return np.tile(interp, (coords.shape[1], 1)).T

        # Else, the input function does not meet the assumptions.
        raise RuntimeError(
            "Vectorize the function so that it can be evaluated for multiple columns"
            + " of the input array."
        )

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        # If there are no degrees of freedom, we return an empty matrix.
        if self.ndof(sd) == 0:
            return sps.csc_array((pg.AMBIENT_DIM**self.tensor_order, 0))

        Pi = self.proj_to_PwPolynomials(sd)
        Poly_space = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(
            self.keyword
        )

        return Poly_space.eval_at_cell_centers(sd) @ Pi

    def source_term(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assembles the source term by interpolating the given function
        and multiplying by the mass matrix.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): A function that returns the function values at coordinates.

        Returns:
            np.ndarray: The source term.
        """
        return self.assemble_mass_matrix(sd) @ self.interpolate(sd, func)

    @abc.abstractmethod
    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        r"""
        Assembles the natural boundary condition term
        :math:`(\text{tr}(q), p)_{\partial\Omega}`, where :math:`\text{tr}`
        denotes the appropriate trace operator for the current space :math:`V_h`
        onto :math:`\partial\Omega`.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): The function representing the natural boundary condition.
            b_faces (np.ndarray): The boundary faces, as a boolean mask or as indices.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """

    @abc.abstractmethod
    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            Type[pg.Discretization]: The range discretization class.
        """

    @abc.abstractmethod
    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the inclusion matrix that projects the finite element space onto
        the lowest order piecewise polynomial space without loss of information.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The inclusion matrix.
        """

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid.

        The broken gradient :math:`\nabla_h` acts element-wise on functions in
        the current finite element space :math:`V_h`, mapping to the piecewise
        polynomials of :meth:`pg.Discretization.get_broken_grad_discr`; see
        :meth:`pg.Discretization.proj_to_PwPolynomials` for the projection used.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        proj = self.proj_to_PwPolynomials(sd)
        pwp = pg.get_PwPolynomials(self.poly_order, self.tensor_order)(self.keyword)
        grad = pwp.assemble_broken_grad_matrix(sd)

        return grad @ proj

    def get_broken_grad_discr(self) -> Type[pg.VecPwPolynomials]:
        """
        Returns the discretization class of the range of the broken gradient, the
        piecewise polynomials of one degree less and one tensor order more than the
        discretization.

        Returns:
            Type[pg.VecPwPolynomials]: The discretization class of the broken
            gradient.
        """
        grad_order = self.tensor_order + 1
        grad_class = pg.get_PwPolynomials(self.poly_order - 1, grad_order)
        return cast(Type[pg.VecPwPolynomials], grad_class)

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: bool = True,
        poly_order: int | None = -1,
        data: dict | None = None,
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a
        function. The default behavior interpolates the function as a piecewise
        polynomial of order at least 1.

        This behavior can be overruled by setting poly_order = None. Then the error is
        computed using interpolation, which may result in unexpected superconvergence.

        Args:
            sd (pg.Grid): The grid.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable): Function that represents the analytical solution.
            relative (bool): Compute the relative error or not. Defaults to True.
            poly_order (int | None): If poly_order is specified as an integer, the
                error is evaluated in a piecewise polynomial space of that order. If it
                is None, we use interpolation. Defaults to -1, the piecewise polynomials
                of the order of the discretization, at least 1.
            data (dict | None): The data of the mass matrix in which the error is
                measured. Defaults to None.

        Returns:
            float: The computed error.
        """
        # Default case in which we use piecewise linears, at least.
        if poly_order == -1:
            poly_order = max(self.poly_order, 1)

        # If poly_order is None, then we use the interpolant of the space.
        if poly_order is None:
            int_sol = self.interpolate(sd, ana_sol)
            mass = self.assemble_mass_matrix(sd, data)

            norm = (int_sol @ mass @ int_sol.T) if relative else 1

            diff = num_sol - int_sol
            return np.sqrt(diff @ mass @ diff.T / norm)

        # If poly_order is specified, we compare the solutions in a piecewise polynomial
        # space
        else:
            poly_space = pg.get_PwPolynomials(poly_order, self.tensor_order)(
                self.keyword
            )
            proj_sol = pg.proj_to_PwPolynomials(self, sd, poly_order) @ num_sol

            return poly_space.error_l2(
                sd, proj_sol, ana_sol, relative, poly_order=None, data=data
            )
