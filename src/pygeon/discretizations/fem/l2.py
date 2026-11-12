"""Module for the discretizations of the L2 space."""

from math import comb
from typing import Callable, Type

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwPolynomials(pg.Discretization):
    r"""
    Class implementing the finite element discretization for piecewise polynomials
    of order k :math:`\mathbb{P}_k(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \subset \mathbb{R}^d`.

    **Degrees of freedom.** On each cell, the degrees of freedom are the values at
    the nodes of the Lagrange basis, :math:`\lambda = \alpha / k` in barycentric
    coordinates, see :meth:`pg.BernsteinBasis.lagrange_nodes`. The barycentric
    coordinate :math:`\lambda_i` is the one of the vertex :math:`x_i` of the cell,
    with the vertices in the order of ``sd.cell_nodes()``, i.e. of increasing global
    index. The nodes are in the order of
    :meth:`pg.BernsteinBasis.compute_multi_indices`: first the vertices, then the
    edges, the faces, and the interior, each subsimplex in the lexicographic order of
    its vertices, and within a subsimplex the multi-indices in decreasing
    lexicographic order. For instance, for the order 3 on a triangle the local
    degrees of freedom are

    ===== ============== ===================
    i     :math:`\alpha` node
    ===== ============== ===================
    0-2   (3, 0, 0), ... the vertices
    3, 4  (2, 1, 0),     the edge x0 x1
          (1, 2, 0)
    5, 6  (2, 0, 1),     the edge x0 x2
          (1, 0, 2)
    7, 8  (0, 2, 1),     the edge x1 x2
          (0, 1, 2)
    9     (1, 1, 1)      the barycenter
    ===== ============== ===================

    For the order 2 on a triangle, the basis functions are
    :math:`\lambda_i (2 \lambda_i - 1)` at the vertices and
    :math:`4 \lambda_i \lambda_j` at the midpoints of the edges, each equal to one at
    its node and zero at the other ones. The piecewise constants are the exception:
    their degree of freedom is the integral on the cell, see :meth:`dof_scaling`.

    **Numbering.** The degree of freedom of the local basis function i on the cell c
    is i * num_cells + c, see :meth:`local_dofs_of_cell`. Hence an operator given by
    the same local matrix L on every cell, up to a factor :math:`w_K` per cell, is
    assembled as

    .. math::

        L \otimes \operatorname{diag}(w_K),

    which is how all the generic methods are built.

    **Bernstein basis.** The local matrices are computed in the Bernstein basis of
    :class:`pg.BernsteinBasis`, where the mass matrix, the derivatives, and the
    evaluation have a closed form, on the reference simplex: since the map to each
    cell is affine, the barycentric coordinates are the same on all the cells, and
    only the factors :math:`w_K` depend on the geometry. A local matrix :math:`L^B`
    from the Bernstein basis of order n to the one of order m is changed to the
    Lagrange bases of the degrees of freedom as

    .. math::

        L = V_m L^B V_n^{-1},

    with :math:`V_n` the Bernstein--Vandermonde matrix of
    :meth:`pg.BernsteinBasis.to_lagrange`, from the Bernstein to the Lagrange
    coefficients, and :math:`V_n^{-1}` the one of
    :meth:`pg.BernsteinBasis.to_bernstein`.

    The generic methods hold for any order, so that a class of a higher order only
    needs to set :attr:`poly_order`. The classes of the orders 0, 1, and 2 override
    some of them, with the same basis.
    """

    poly_order: int
    """Polynomial degree of the basis functions"""

    tensor_order = pg.SCALAR
    """Scalar-valued discretization"""

    @staticmethod
    def dof_scaling(sd: pg.Grid, order: int) -> np.ndarray:
        """
        The factor between the degrees of freedom and the values at the nodes in each
        cell: the cell volumes for the piecewise constants, whose degrees of freedom
        are the integrals on the cells, and one for the other orders. The local
        matrices act on the values, so that the operators divide by it on the side of
        the domain and multiply by it on the side of the range.

        Args:
            sd (pg.Grid): The grid.
            order (int): The polynomial order.

        Returns:
            np.ndarray: The factor of each cell.
        """
        return sd.cell_volumes if order == 0 else np.ones(sd.num_cells)

    def ndof_per_entity(self, dim: int) -> np.ndarray:
        """
        Returns the number of degrees of freedom per geometric entity, ordered by
        the dimension of the entity as [0, 1, 2, 3].
        In this case, all the degrees of freedom of the element are associated with
        the cell.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: The number of degrees of freedom per entity.
        """
        ndof = comb(dim + self.poly_order, self.poly_order)
        return np.roll([ndof, 0, 0, 0], dim)  # cells

    def local_dofs_of_cell(self, sd: pg.Grid, c: int) -> np.ndarray:
        """
        Compute the local degrees of freedom (DOFs) indices for a cell.

        Args:
            sd (pg.Grid): The grid.
            c (int): Index of the cell.

        Returns:
            np.ndarray: Array of local DOF indices associated with the cell.
        """
        return sd.num_cells * np.arange(self.ndof_per_element(sd.dim)) + c

    def get_local_dof_multi_indices(self, dim: int) -> np.ndarray:
        r"""
        Returns the multi-indices of the local degrees of freedom: the row i is the
        multi-index :math:`\alpha` of the local degree of freedom i, the value at the
        node :math:`\lambda = \alpha / k`, with :math:`\lambda_j` the barycentric
        coordinate of the vertex j of the cell in the order of ``sd.cell_nodes()``.
        See the table in the docstring of the class and
        :meth:`pg.BernsteinBasis.compute_multi_indices`.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: The multi-indices, of shape (ndof_per_element, dim + 1).
        """
        return pg.BernsteinBasis(dim, self.poly_order).multi_indices

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Computes the mass matrix :math:`(\sigma u, v)_\Omega` for
        :math:`u, v \in \mathbb{P}_k(\Omega)` and :math:`\sigma` a scalar, constant on
        each cell. Both domain and range lie in :class:`pg.PwPolynomials`.

        The integral on a cell K is the one on the reference simplex S, of measure
        one, times :math:`|K|`, so that, with the local mass matrix M of
        :meth:`assemble_local_mass`,

        .. math::

            M_\Omega = M \otimes \operatorname{diag}(\sigma_K |K|),

        divided by the square of :meth:`dof_scaling` for the piecewise constants.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            data (dict | None): Dictionary with possible scaling.

        Returns:
            sps.csc_array: The mass matrix, of shape (ndof, ndof).
        """
        return self.assemble_from_local_mass(sd, self.assemble_local_mass(sd.dim), data)

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the lumped matrix for the given grid, which is an approximation,
        obtained with a quadrature rule, of the mass matrix
        :math:`(\sigma u, v)_\Omega` for :math:`u, v \in \mathbb{P}_k(\Omega)` and
        :math:`\sigma` a scalar. Both domain and range lie in
        :class:`pg.PwPolynomials` (L2 space). It is assembled as the mass matrix, see
        :meth:`assemble_mass_matrix`, from the local lumped mass matrix of
        :meth:`assemble_local_lumped_mass`.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        return self.assemble_from_local_mass(
            sd, self.assemble_local_lumped_mass(sd.dim), data
        )

    def assemble_from_local_mass(
        self, sd: pg.Grid, local_mass: np.ndarray, data: dict | None
    ) -> sps.csc_array:
        r"""
        Assembles a mass matrix from its local matrix on the reference simplex, of
        measure one, as :math:`M \otimes \operatorname{diag}(\sigma_K |K|)`, divided by
        the square of :meth:`dof_scaling` for the piecewise constants.

        Args:
            sd (pg.Grid): The grid.
            local_mass (np.ndarray): The local mass matrix.
            data (dict | None): Optional data dictionary, with the weight sigma.

        Returns:
            sps.csc_array: The mass matrix.
        """
        weight = pg.get_cell_data(sd, data, self.keyword, pg.WEIGHT)
        scaling = self.dof_scaling(sd, self.poly_order)
        diag_weight = sps.diags_array(sd.cell_volumes * weight / scaling**2)

        M = sps.csc_array(sps.kron(local_mass, diag_weight))
        M.eliminate_zeros()
        return M

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        For piecewise polynomials (L2), the differential maps to the zero space, so
        the matrix has no rows, since these are discontinuous functions with no global
        differential.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array((0, self.ndof(sd)))

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid, from
        :math:`\mathbb{P}_k(\Omega)` to :math:`[\mathbb{P}_{k-1}(\Omega)]^d`.

        On each cell K, a polynomial u is a polynomial in the barycentric coordinates
        :math:`\lambda_0, \dots, \lambda_d` of K, so that its gradient follows from the
        chain rule,

        .. math::

            \nabla u = \sum_{i = 0}^d \frac{\partial u}{\partial \lambda_i} \,
            \nabla \lambda_i.

        The two factors are assembled separately:

        1. The derivatives :math:`\partial u / \partial \lambda_i` do not depend on
           the cell: they map a polynomial of order k to one of order k - 1 in the
           same barycentric coordinates. In the Bernstein basis they are the sparse
           matrices :math:`D^B_i` of :meth:`pg.BernsteinBasis.derivatives`, and in
           the Lagrange bases of the degrees of freedom they are
           :math:`V_{k-1} D^B_i V_k^{-1}`, with the Bernstein--Vandermonde matrices
           :math:`V_k` of :meth:`pg.BernsteinBasis.to_lagrange`. Since the degrees of
           freedom are numbered (basis function) * num_cells + (cell), the same
           local matrix on all the cells is its Kronecker product with the identity,
           and the d + 1 derivatives stacked give the gradient in the barycentric
           coordinates, from :math:`\mathbb{P}_k(\Omega)` to
           :math:`[\mathbb{P}_{k-1}(\Omega)]^{d + 1}`,

           .. math::

               \partial = \begin{bmatrix} V_{k-1} D^B_0 V_k^{-1} \\ \vdots \\
               V_{k-1} D^B_d V_k^{-1} \end{bmatrix} \otimes I.

        2. The gradients :math:`\nabla \lambda_i` carry the geometry: they are
           constant on each cell, from :meth:`pg.Grid.compute_barycentric_gradients`
           in the ambient coordinates, and rotated to the plane or the line of the
           grid with ``sd.rotation_matrix``, as the components of
           :class:`pg.VecPwPolynomials`. The component a of the chain rule multiplies
           the derivative i by :math:`(\nabla \lambda_i)_a` on each cell, i.e. by the
           block matrix :math:`\Lambda`, from
           :math:`[\mathbb{P}_{k-1}(\Omega)]^{d + 1}` to
           :math:`[\mathbb{P}_{k-1}(\Omega)]^d`, with the diagonal blocks

           .. math::

               \Lambda_{ai} = \operatorname{diag}((\nabla \lambda_i)_a),

           whose values are repeated on all the degrees of freedom of each cell.

        The broken gradient is then

        .. math::

            \nabla_h = \Lambda \partial,

        with the components in blocks of rows, as in :class:`pg.VecPwPolynomials`.
        For k = 1 the gradient is in :class:`pg.PwConstants`, whose degrees of
        freedom are the integrals on the cells, so that :math:`\Lambda` is also
        multiplied by the cell volumes.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        dim, k = sd.dim, self.poly_order
        basis, low = pg.BernsteinBasis(dim, k), pg.BernsteinBasis(dim, k - 1)

        # The gradient in the barycentric coordinates, the same on every cell, from the
        # Lagrange coefficients of order k to the ones of order k - 1 through the
        # Bernstein basis, one block of rows per barycentric coordinate
        local = [
            low.to_lagrange() @ D_i @ basis.to_bernstein()
            for D_i in basis.derivatives()
        ]
        partial = sps.kron(np.vstack(local), sps.eye_array(sd.num_cells))

        # The gradients of the barycentric coordinates in the plane of the grid,
        # (dim, dim + 1, nc), scaled to the degrees of freedom of the order k - 1
        grad_lambda = np.einsum(
            "ak,kic->aic", sd.rotation_matrix, sd.compute_barycentric_gradients()
        )
        grad_lambda *= self.dof_scaling(sd, k - 1)

        # The block matrix Lambda, with the blocks diag((grad lambda_i)_a): the values
        # are constant on each cell, so repeated on all its degrees of freedom
        num_low = low.multi_indices.shape[0]
        Lambda = sps.block_array(
            [
                [sps.diags_array(np.tile(g, num_low)) for g in grad_a]
                for grad_a in grad_lambda
            ]
        )

        return sps.csc_array(Lambda @ partial)

    def assemble_stiff_matrix(
        self, sd: pg.Grid, _data: dict | None = None
    ) -> sps.csc_array:
        r"""
        Assembles the stiffness matrix for the given grid, representing the bilinear
        form :math:`(K \nabla u, \nabla v)_\Omega` for
        :math:`u,v \in \mathbb{P}_k(\Omega)` and :math:`K` a scalar.

        For piecewise polynomials, the stiffness matrix is the zero matrix since the
        differential operator is trivial.

        Args:
            sd (pg.Grid): The grid.
            data (dict | None): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        return sps.csc_array((self.ndof(sd), self.ndof(sd)))

    def assemble_nat_bc(
        self,
        sd: pg.Grid,
        _func: Callable[[np.ndarray], np.ndarray],
        _b_faces: np.ndarray,
    ) -> np.ndarray:
        r"""
        Assembles the natural boundary condition vector, equal to zero.

        For piecewise polynomials, the natural boundary condition is zero since these
        are discontinuous functions with no boundary trace.

        Args:
            sd (pg.Grid): The grid.
            func (Callable[[np.ndarray], np.ndarray]): The function defining the
                 natural boundary condition.
            b_faces (np.ndarray): The boundary faces, as a boolean mask or as indices.

        Returns:
            np.ndarray: The assembled natural boundary condition vector.
        """
        return np.zeros(self.ndof(sd))

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        r"""
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Raises:
            NotImplementedError: There is no zero discretization available in PyGeoN.
        """
        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")

    def assemble_local_mass(self, dim: int) -> np.ndarray:
        r"""
        Computes the local mass matrix :math:`(\varphi_i, \varphi_j)_S` for
        :math:`\varphi_i, \varphi_j` being the local basis functions of the
        piecewise polynomials on a d-simplex :math:`S` with measure 1, exactly. The
        mass matrix :math:`M^B` of the Bernstein basis, see
        :meth:`pg.BernsteinBasis.mass`, is a bilinear form, so that it changes to the
        Lagrange basis on both sides,

        .. math::

            M = V_k^{-\top} M^B V_k^{-1}.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local mass matrix for piecewise polynomials.
        """
        basis = pg.BernsteinBasis(dim, self.poly_order)
        return basis.to_bernstein().T @ basis.mass(basis) @ basis.to_bernstein()

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        r"""
        Computes the local lumped mass matrix for piecewise polynomials, an
        approximation by a quadrature rule of the form
        :math:`(\varphi_i, \varphi_j)_S`, with :math:`\varphi_i, \varphi_j` being
        the local basis functions of the piecewise polynomials on a d-simplex :math:`S`.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise polynomials.

        Raises:
            NotImplementedError: The lumped mass is provided only for the orders 0,
                1, and 2, by their classes.
        """
        raise NotImplementedError(
            f"No lumped mass for the piecewise polynomials of order {self.poly_order}"
        )

    def proj_to_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Construct the matrix for projecting a piecewise function to a piecewise
        polynomial function. The (identity) projection operator :math:`\Pi` takes a
        function from :math:`\mathbb{P}_k(\Omega)` and maps it to a piecewise
        polynomial function in :math:`\mathbb{P}_k(\Omega)`.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        return sps.eye_array(self.ndof(sd)).tocsc()

    def proj_to_lower_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to -1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_k(\Omega)` and maps it
        to a piecewise polynomial function of lower order in
        :math:`\mathbb{P}_{k-1}(\Omega)`, as the L2 projection on each cell.

        In the Bernstein bases, the projection :math:`p = \Pi u` of order k - 1 is
        the solution of :math:`(p, q)_K = (u, q)_K` for all q of order k - 1, i.e. of
        :math:`M^B_{k-1,k-1} c_p = M^B_{k-1,k} c_u`, with :math:`M^B_{m,n}` the mass
        matrix between the Bernstein bases of the orders m and n, see
        :meth:`pg.BernsteinBasis.mass`. The factor :math:`|K|` cancels out, so that
        the local matrix is the same on every cell and, in the Lagrange bases,

        .. math::

            \Pi = V_{k-1} \left( M^B_{k-1,k-1} \right)^{-1} M^B_{k-1,k} V_k^{-1}.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.

        Raises:
            NotImplementedError: For the piecewise constants, of the lowest order.
        """
        k, dim = self.poly_order, sd.dim
        if k == 0:
            raise NotImplementedError("No order below the piecewise constants")

        # On the reference simplex, the same on every cell: from the Lagrange to the
        # Bernstein coefficients of order k, the L2 projection onto the Bernstein
        # basis of order k - 1, and back to the Lagrange coefficients of order k - 1
        basis, low = pg.BernsteinBasis(dim, k), pg.BernsteinBasis(dim, k - 1)
        l2_projection = np.linalg.solve(low.mass(low), low.mass(basis))
        local = low.to_lagrange() @ l2_projection @ basis.to_bernstein()

        # From the degrees of freedom of the order k to the ones of the order k - 1
        scaling = self.dof_scaling(sd, k - 1) / self.dof_scaling(sd, k)
        proj = sps.kron(local, sps.diags_array(scaling))
        return sps.csc_array(proj)

    def proj_to_higher_PwPolynomials(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Projects the discretization to +1 order discretization. The projection operator
        :math:`\Pi` takes a function from :math:`\mathbb{P}_k(\Omega)` and maps it
        to a piecewise polynomial function of higher order in
        :math:`\mathbb{P}_{k+1}(\Omega)`, exactly, i.e. it is the inclusion.

        A polynomial of order k is also one of order k + 1, whose degrees of freedom
        are its values at the nodes :math:`\lambda_\beta` of order k + 1. From the
        degrees of freedom of order k, the Bernstein coefficients are
        :math:`V_k^{-1} u`, and the values follow from the Bernstein polynomials
        :math:`B_k` at those nodes, see :meth:`pg.BernsteinBasis.eval`, the same on
        every cell,

        .. math::

            \Pi = B_k(\lambda_\beta) V_k^{-1}.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The projection matrix.
        """
        k, dim = self.poly_order, sd.dim
        basis, high = pg.BernsteinBasis(dim, k), pg.BernsteinBasis(dim, k + 1)
        local = basis.eval(high.lagrange_nodes()) @ basis.to_bernstein()

        # From the degrees of freedom of the order k to the ones of the order k + 1
        scaling = self.dof_scaling(sd, k + 1) / self.dof_scaling(sd, k)
        proj = sps.kron(local, sps.diags_array(scaling))
        return sps.csc_array(proj)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the matrix for evaluating the discretization at the cell centers.
        The cell center of a simplex is its barycenter, the same point
        :math:`\lambda_c = (1, \dots, 1) / (d + 1)` in barycentric coordinates on
        every cell. As in :meth:`proj_to_higher_PwPolynomials`, the values there
        follow from the Bernstein coefficients :math:`V_k^{-1} u` and the Bernstein
        polynomials at :math:`\lambda_c`, the same on every cell,

        .. math::

            B_k(\lambda_c) V_k^{-1}.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        k, dim = self.poly_order, sd.dim
        basis = pg.BernsteinBasis(dim, k)
        barycenter = np.full((1, dim + 1), 1 / (dim + 1))
        local = basis.eval(barycenter) @ basis.to_bernstein()

        evaluation = sps.kron(local, sps.eye_array(sd.num_cells))
        return sps.csc_array(evaluation)

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        r"""
        Interpolates a function onto the finite element space through its values at
        points inside each cell K, so that a function discontinuous across the cells
        is sampled on a single side, in three steps:

        1. The points are the nodes of the Lagrange basis shrunk toward the
           barycenter :math:`\lambda_c = (1, \dots, 1) / (d + 1)`, in barycentric
           coordinates,

           .. math::

               \mu_\alpha = \lambda_c + s \left( \frac{\alpha}{k} - \lambda_c
               \right), \qquad s = \frac{1}{\sqrt{d + 2}}.

           They are unisolvent for :math:`\mathbb{P}_k` as the nodes, being their
           image by a homothety, and, for k = 1, they are the Gauss points of a
           quadrature rule of order 2.

        2. The function is evaluated at the points of each cell,
           :math:`x_K(\mu) = \sum_i \mu_i x_{K, i}`, see
           :meth:`pg.Grid.barycentric_to_coords`.

        3. The interpolant is the polynomial with these values: its Bernstein
           coefficients c solve :math:`B_k(\mu) c = f(x_K(\mu))`, see
           :meth:`pg.BernsteinBasis.eval`, so that its degrees of freedom are

           .. math::

               u_K = V_k B_k(\mu)^{-1} f(x_K(\mu)),

           with the same local matrix on every cell.

        It is exact for the polynomials of order k.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): A function that returns the function values at
                coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        k, dim = self.poly_order, sd.dim
        basis = pg.BernsteinBasis(dim, k)

        # The points inside the simplex, in barycentric coordinates
        center = 1 / (dim + 1)
        points = center + (basis.lagrange_nodes() - center) / np.sqrt(dim + 2)

        # Their coordinates in each cell, ordered as the dofs, with the cells running
        # faster, and the values of f there, (num points, num_cells)
        coords = sd.barycentric_to_coords(points)
        values = self.eval_func_at_coords(func, coords.reshape((3, -1)))
        values = values.reshape((-1, sd.num_cells))

        # From the values at the points to the ones at the nodes, through the
        # Bernstein coefficients
        local = basis.to_lagrange() @ np.linalg.inv(basis.eval(points))
        return (local @ values).ravel()


class PwConstants(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    lowest order :math:`\mathbb{P}_0(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \subset \mathbb{R}^d`.

    Each degree of freedom is the integral over the cell.
    """

    poly_order = 0
    """Polynomial degree of the basis functions"""

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise constants,
        which coincides with the local mass matrix since P0 has one dof per cell.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise constants.
        """
        return self.assemble_local_mass(dim)

    def assemble_broken_grad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        r"""
        Assembles the broken (element-wise) gradient matrix for the given grid,
        which is zero for the piecewise constants.

        The broken gradient :math:`\nabla_h` maps from :class:`pg.PwConstants` to
        zero, with the shape of a map to the dim components of
        :class:`pg.VecPwConstants`, as for the other orders.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The assembled broken gradient matrix.
        """
        ndof = self.ndof(sd)
        return sps.csc_array((sd.dim * ndof, ndof))

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix for evaluating the discretization at the cell centers:
        the value of each cell, i.e. its degree of freedom, the integral on the cell,
        divided by the cell volume. It holds on any grid, also not simplicial.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The evaluation matrix.
        """
        return sps.csc_array(sps.diags_array(1 / sd.cell_volumes))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        r"""
        Interpolates a function onto the finite element space, through its value at
        the cell center times the cell volume, :math:`u_K = f(x_K) |K|`. It holds on
        any grid, also not simplicial.

        Args:
            sd (pg.Grid): The grid.
            func (Callable): A function that returns the function values at
                coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        return self.eval_func_at_coords(func, sd.cell_centers) * sd.cell_volumes


class PwLinears(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    order 1 :math:`\mathbb{P}_1(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \subset \mathbb{R}^d`.
    """

    poly_order = 1
    """Polynomial degree of the basis functions"""

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        """
        Computes the local lumped mass matrix for piecewise linears, which is a
        diagonal approximation of the local mass matrix, by the quadrature rule at the
        vertices with the weight 1 / (dim + 1) each.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise linears.
        """
        return np.eye(dim + 1) / (dim + 1)

    def get_dof_lookup_array(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles a lookup matrix L with the property L[node, cell] = dof_index, with
        the sparsity of ``sd.cell_nodes()``.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The lookup matrix.
        """
        dof_array = sd.cell_nodes().astype("int")
        ndof = self.ndof(sd)
        dof_array.data = np.reshape(np.arange(ndof), (sd.num_cells, -1), "F").ravel()

        return dof_array


class PwQuadratics(PwPolynomials):
    r"""
    Class implementing the finite element discretization for piecewise polynomials of
    order 2 :math:`\mathbb{P}_2(\Omega) \subset L^2(\Omega)`, for a generic
    domain :math:`\Omega \subset \mathbb{R}^d`.
    """

    poly_order = 2
    """Polynomial degree of the basis functions"""

    def num_edges_per_cell(self, dim: int) -> int:
        """
        Compute the number of edges of a simplex of a given dimension.

        Args:
            dim (int): Dimension.

        Returns:
            int: The number of edges.
        """
        return dim * (dim + 1) // 2

    def get_local_edge_nodes(self, dim: int) -> np.ndarray:
        """
        Lists the local edge-node connectivity in the cell: the pairs of local nodes,
        in lexicographic order, read from the local degrees of freedom of the edges,
        see :meth:`pg.PwPolynomials.get_local_dof_multi_indices`.

        Args:
            dim (int): Dimension.

        Returns:
            np.ndarray: Row i contains the local indices of the nodes connected to the
            edge with local index i.
        """
        # The multi-indices of the edges follow the ones of the dim + 1 vertices, and
        # their nonzero entries are the two vertices of the edge
        edge_alphas = self.get_local_dof_multi_indices(dim)[dim + 1 :]
        return np.nonzero(edge_alphas)[1].reshape((-1, 2))

    def assemble_local_lumped_mass(self, dim: int) -> np.ndarray:
        r"""
        Computes the local lumped mass matrix for piecewise quadratics, which is
        an approximation of the local mass matrix by the quadrature rule, on the
        simplex of measure one, with the weight :math:`(d + 1) / (d + 2)` at the
        barycenter and :math:`1 / ((d + 1)(d + 2))` at each vertex. At the vertices,
        only the basis function of the vertex is not zero, so that their part is
        diagonal.

        Args:
            dim (int): The dimension of the grid.

        Returns:
            np.ndarray: Local lumped mass matrix for piecewise quadratics.
        """
        num_edges = self.num_edges_per_cell(dim)

        # Evaluate the basis function at the cell center
        node_bf_at_cc = np.full(dim + 1, 1 - dim)
        edge_bf_at_cc = np.full(num_edges, 4)
        vals_at_center = np.concatenate((node_bf_at_cc, edge_bf_at_cc)) / (dim + 1) ** 2
        center_weight = (dim + 1) / (dim + 2)

        L = center_weight * np.outer(vals_at_center, vals_at_center)

        # Evaluate the basis functions at the nodes
        vals_at_nodes = np.zeros(dim + 1 + num_edges)
        vals_at_nodes[: dim + 1] = 1
        node_weight = 1 / ((dim + 1) * (dim + 2))

        L += node_weight * np.diag(vals_at_nodes)

        return L
