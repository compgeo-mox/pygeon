"""The Bernstein basis on the reference simplex and its change to the Lagrange basis."""

from itertools import combinations_with_replacement
from math import factorial

import numpy as np

# The factorials of an array of integers, as floats
factorials = np.vectorize(factorial, otypes=[float])


class BernsteinBasis:
    r"""
    The Bernstein polynomials :math:`B^k_\alpha = k! / \alpha! \, \lambda^\alpha`, for
    the multi-indices :math:`\alpha \in \mathbb{N}^{d + 1}` with :math:`|\alpha| = k`,
    on a d-simplex in the barycentric coordinates :math:`\lambda`. Their mass matrix
    has a closed form from the exact integral of the products of barycentric
    coordinates of Vermolen and Segal (2018), and their derivatives in the barycentric
    coordinates are again Bernstein polynomials, see Kirby (2011) and Ainsworth,
    Andriamaro, and Davydov (2011).

    The Lagrange basis on the nodes :math:`\alpha / k`, of the same multi-indices, is
    related to them by the Bernstein--Vandermonde matrix, so that the operators of the
    Lagrange basis follow from the ones of the Bernstein basis, e.g. the mass matrix
    :math:`V^{-\top} M V^{-1}`. The matrices are small, so that they are computed
    at each call.

    Attributes:
        dim (int): The dimension of the simplex.
        order (int): The polynomial order k.
        multi_indices (np.ndarray): The multi-indices, one per row, ordered by the
            subsimplex that contains their node: the vertices, then the edges in the
            order of :meth:`pg.PwQuadratics.get_local_edge_nodes`, then the faces and
            the interior.
    """

    def __init__(self, dim: int, order: int) -> None:
        """
        Initialize the basis.

        Args:
            dim (int): The dimension of the simplex.
            order (int): The polynomial order k.
        """
        self.dim = dim
        self.order = order
        self.multi_indices = self.compute_multi_indices(dim, order)

    @staticmethod
    def compute_multi_indices(dim: int, order: int) -> np.ndarray:
        r"""
        The multi-indices :math:`\alpha \in \mathbb{N}^{d + 1}` with
        :math:`|\alpha| = k`, ordered by the subsimplex that contains the node
        :math:`\alpha / k`, i.e. the vertices with :math:`\alpha_i > 0`: first by its
        dimension (vertices, edges, faces, interior), then by its vertices in
        lexicographic order, and within the subsimplex by :math:`\alpha` in decreasing
        lexicographic order, from the node closest to its vertex of lowest index.

        Args:
            dim (int): The dimension of the simplex.
            order (int): The polynomial order k.

        Returns:
            np.ndarray: The multi-indices, one per row.

        Example:
            On a triangle, the order 1 gives the vertices, and the order 2 the
            vertices and then the midpoints of the edges. Each row is a node
            :math:`\lambda = \alpha / k`, i.e. the point
            :math:`\sum_i (\alpha_i / k) \, x_i`, and the Bernstein polynomial
            :math:`B^k_\alpha`: for the order 2, :math:`\lambda_0^2` at the vertex
            :math:`x_0` for [2, 0, 0], and :math:`2 \lambda_0 \lambda_1` at the
            midpoint of the edge :math:`x_0 x_1` for [1, 1, 0]::

                >>> BernsteinBasis.compute_multi_indices(2, 1)
                array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
                >>> BernsteinBasis.compute_multi_indices(2, 2)
                array([[2, 0, 0],
                       [0, 2, 0],
                       [0, 0, 2],
                       [1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1]])
        """
        # Each choice, with repetition, of k vertices gives the multi-index that
        # counts how many times each vertex is chosen
        alphas = [
            np.bincount(np.array(choice, dtype=int), minlength=dim + 1)
            for choice in combinations_with_replacement(range(dim + 1), order)
        ]

        # Sorted by the dimension of the subsimplex (vertices, edges, faces,
        # interior), then by the subsimplex, in the order of the combinations of its
        # vertices, and then decreasingly within the subsimplex
        def subsimplex_order(alpha: np.ndarray) -> tuple:
            support = np.flatnonzero(alpha)
            return support.size, tuple(support), tuple(-alpha)

        return np.array(sorted(alphas, key=subsimplex_order))

    def lagrange_nodes(self) -> np.ndarray:
        r"""
        The nodes of the Lagrange basis in barycentric coordinates, one per row,

        .. math::

            \lambda_\alpha = \frac{\alpha}{k},

        and the barycenter :math:`\lambda_i = 1 / (d + 1)` for k = 0.

        Returns:
            np.ndarray: The nodes.
        """
        if self.order == 0:
            return np.full((1, self.dim + 1), 1 / (self.dim + 1))
        return self.multi_indices / self.order

    def eval(self, points: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the Bernstein polynomials at the points given in barycentric
        coordinates,

        .. math::

            B^k_\alpha(\lambda) = \frac{k!}{\alpha!} \prod_{i = 0}^d
            \lambda_i^{\alpha_i}.

        Args:
            points (np.ndarray): The barycentric coordinates of the points, one per
                row.

        Returns:
            np.ndarray: The values, one row per point and one column per polynomial.
        """
        alphas = self.multi_indices
        # The powers lambda_i^alpha_i of each point (rows) for each multi-index
        # (columns), multiplied over the barycentric coordinates
        powers = points.reshape((-1, 1, self.dim + 1)) ** alphas
        return multinomial(alphas) * np.prod(powers, axis=2)

    def to_lagrange(self) -> np.ndarray:
        r"""
        The Bernstein--Vandermonde matrix, of the values of the Bernstein polynomials
        at the nodes of the Lagrange basis,

        .. math::

            V_{\beta \alpha} = B^k_\alpha(\lambda_\beta):

        it maps the Bernstein coefficients :math:`c` of a polynomial to its values
        :math:`u = V c` at the nodes, i.e. its Lagrange coefficients.

        Returns:
            np.ndarray: The matrix, one row per node and one column per polynomial.
        """
        return self.eval(self.lagrange_nodes())

    def to_bernstein(self) -> np.ndarray:
        r"""
        The inverse of the Bernstein--Vandermonde matrix, which maps the Lagrange
        coefficients :math:`u` of a polynomial to its Bernstein coefficients
        :math:`c = V^{-1} u`.

        Returns:
            np.ndarray: The matrix.
        """
        return np.linalg.inv(self.to_lagrange())

    def mass(self, other: "BernsteinBasis") -> np.ndarray:
        r"""
        The inner products of the polynomials of this basis, of order m (rows), with
        the ones of other, of order n (columns), on a d-simplex S with measure 1. The
        product of two Bernstein polynomials is a monomial,
        :math:`B^n_\beta B^m_\alpha = m! / \alpha! \, n! / \beta! \,
        \lambda^{\alpha + \beta}`, with the exact integral
        :math:`\int_S \lambda^\gamma = d! \, \gamma! / (d + |\gamma|)!` of Vermolen
        and Segal (2018), which gives

        .. math::

            M_{\alpha \beta} = (B^n_\beta, B^m_\alpha)_S
            = \frac{m! \, n!}{\alpha! \, \beta!}
            \frac{d! \, (\alpha + \beta)!}{(d + m + n)!}.

        Args:
            other (BernsteinBasis): The basis of the columns, of the same dimension.

        Returns:
            np.ndarray: The matrix.
        """
        alphas, betas = self.multi_indices, other.multi_indices

        # The exponents alpha + beta of all the pairs, (rows, columns, d + 1), and the
        # integrals of the monomials with them
        sums = alphas.reshape((-1, 1, self.dim + 1)) + betas
        integrals = np.prod(factorials(sums), axis=2) * factorial(self.dim)
        integrals /= factorial(self.dim + self.order + other.order)

        return np.outer(multinomial(alphas), multinomial(betas)) * integrals

    def derivatives(self) -> list[np.ndarray]:
        r"""
        The derivatives in the barycentric coordinates :math:`\lambda_i`, regarded as
        independent variables, from this basis to the one of order k - 1:
        :math:`\partial_{\lambda_i} B^k_{\beta + e_i} = k B^{k-1}_\beta`, and zero
        for the polynomials with :math:`\alpha_i = 0`, i.e. the matrices

        .. math::

            (D_i)_{\beta \alpha} =
            \begin{cases}
                k & \text{if } \alpha = \beta + e_i, \\
                0 & \text{otherwise},
            \end{cases}

        so that the gradient on a cell is
        :math:`\nabla u = \sum_i (D_i c) \, \nabla \lambda_i`, for the Bernstein
        coefficients :math:`c` of u.

        Returns:
            list[np.ndarray]: One matrix per barycentric coordinate, with one nonzero
            entry per row.
        """
        low = BernsteinBasis(self.dim, self.order - 1)

        # The differences alpha - beta of all the pairs, (rows beta, columns alpha,
        # d + 1): the derivative in lambda_i is k where the difference is e_i
        diffs = self.multi_indices - low.multi_indices.reshape((-1, 1, self.dim + 1))
        units = np.eye(self.dim + 1, dtype=int)
        return [self.order * np.all(diffs == unit, axis=2) for unit in units]


def multinomial(alphas: np.ndarray) -> np.ndarray:
    r"""
    The multinomial coefficients of the multi-indices along the last axis,

    .. math::

        \binom{|\alpha|}{\alpha} = \frac{|\alpha|!}{\alpha!}
        = \frac{|\alpha|!}{\prod_i \alpha_i!}.

    Args:
        alphas (np.ndarray): The multi-indices along the last axis.

    Returns:
        np.ndarray: The coefficients.
    """
    return factorials(alphas.sum(axis=-1)) / np.prod(factorials(alphas), axis=-1)
