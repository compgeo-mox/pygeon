import numpy as np
import pygeon as pg
import math


def factorial(n: float) -> int:
    """
    Compute the factorial of a float by first rounding to an int.
    Args:
        n (float): the input float

    Returns:
        int: the factorial n!
    """
    return math.factorial(int(n))


def integrate_monomial(alphas: np.ndarray) -> float:
    """
    Exact integration of products of monomials based on
    Vermolen et al.

    Args:
        alphas (np.ndarray): array of exponents alpha_i of the polynomial
            expressed as prod_i lambda_i ^ alpha_i

    Returns:
        float: the integral of the polynomial on a simplex with measure 1
    """
    dim = len(alphas) - 1
    map(math.factorial, alphas)
    fac_alph = [factorial(a_i) for a_i in alphas]
    return factorial(dim) * np.prod(fac_alph) / factorial(dim + np.sum(alphas))


def assemble_coeff_mat(expnts: np.ndarray) -> np.ndarray:
    """
    Compute the inner products of all monomials up to degree 2

    Args:
        expnts (np.ndarray): each column is an array of exponents
            alpha_i of the monomial basis function expressed as
            prod_i lambda_i ^ alpha_i.

    Returns:
        np.ndarray: the inner products of the monomials
            on a simplex with measure 1.
    """
    n_monomials = expnts.shape[1]
    coeff_mat = np.zeros((n_monomials, n_monomials))

    for i in np.arange(n_monomials):
        for j in np.arange(n_monomials):
            coeff_mat[i, j] = integrate_monomial(expnts[:, i] + expnts[:, j])

    return coeff_mat


def assemble_local_mass(dim: int) -> np.ndarray:
    """ """

    # Helper constants
    n_edges = dim * (dim + 1) // 2
    eye = np.eye(dim + 1)
    zero = np.zeros((n_edges, dim + 1))

    # Make a list of monomial up to degree 2,
    # by exponents, consisting of
    # - the linears lambda_i
    # - the cross-quadratics lambda_i \lambda_j
    # - the quadratics lambda_i^2
    if dim == 0:
        return np.ones((1, 1))
    elif dim == 1:
        quads = np.ones((2, 1))
    elif dim == 2:
        quads = 1 - eye
    elif dim == 3:
        quad_0 = np.vstack((np.zeros((1, 3)), 1 - np.eye(3)))
        quad_1 = np.vstack((np.ones((1, 3)), np.eye(3)[::-1]))
        quads = np.hstack((quad_0, quad_1))
    exponents = np.hstack((eye, quads, 2 * eye))

    # Compute the local mass matrix of the monomials
    coeff_mat = assemble_coeff_mat(exponents)

    # Our basis functions are given by
    # - nodes: lambda_i (2 lambda_i - 1)
    # - edges: 4 lambda_i lambda_j
    basis_nodes = np.vstack((-eye, zero, 2 * eye))
    basis_edges = np.zeros((2 * (dim + 1) + n_edges, n_edges))
    basis_edges[dim + 1 : dim + n_edges + 1] = 4 * np.eye(n_edges)
    basis = np.hstack((basis_nodes, basis_edges))

    return basis.T @ coeff_mat @ basis


if __name__ == "__main__":

    # for dim in range(4):
    #     M = assemble_local_mass(dim)
    #     print(M.sum())

    dim = 3
    mdg = pg.unit_grid(dim, 0.05)
    sd = mdg.subdomains()[0]

    opposite_nodes = sd.compute_opposite_nodes()

    for c in np.arange(sd.num_cells):
        loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
        faces = opposite_nodes.indices[loc]
        nodes = opposite_nodes.data[loc]
        signs = sd.cell_faces.data[loc]

        dphi = -sd.face_normals[:, faces] * signs / (dim * sd.cell_volumes[c])
        dphi_2 = pg.Lagrange1.local_grads(sd.nodes[:, nodes], dim)

        assert np.allclose(dphi, dphi_2)

        pass
