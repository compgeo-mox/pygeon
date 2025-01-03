import numpy as np
import pygeon as pg
import math


@staticmethod
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
    Vermolen and Segal (2018).

    Args:
        alphas (np.ndarray): array of exponents alpha_i of the polynomial
            expressed as prod_i lambda_i ^ alpha_i

    Returns:
        float: the integral of the polynomial on a simplex with measure 1
    """
    dim = len(alphas) - 1
    fac_alph = [factorial(a_i) for a_i in alphas]

    return factorial(dim) * np.prod(fac_alph) / factorial(dim + np.sum(alphas))


def assemble_monomial_mass(expnts: np.ndarray) -> np.ndarray:
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
    mass = np.zeros((n_monomials, n_monomials))

    for i in np.arange(n_monomials):
        for j in np.arange(n_monomials):
            mass[i, j] = integrate_monomial(expnts[:, i] + expnts[:, j])

    return mass


def get_num_edges(dim):
    return dim * (dim + 1) // 2


def get_local_edge_numbering(dim: int):

    n_nodes = dim + 1
    n_edges = get_num_edges(dim)
    loc_edges = np.empty((n_edges, 2), int)

    ind = 0
    for first_node in np.arange(n_nodes):
        for second_node in np.arange(first_node + 1, n_nodes):
            loc_edges[ind] = [first_node, second_node]
            ind += 1

    return loc_edges


def assemble_local_mass(dim: int) -> np.ndarray:
    """
    Computes the local mass matrix of the basis functions
    on a d-simplex with measure 1.

    Args:
        dim (int): The dimension of the simplex.

    Returns:
        np.ndarray: the local mass matrix.
    """

    # Helper constants
    n_edges = get_num_edges(dim)
    eye = np.eye(dim + 1)
    zero = np.zeros((n_edges, dim + 1))

    # Make a list of monomials up to degree 2,
    # by exponents, consisting of
    # - the linears lambda_i
    # - the quadratics lambda_i^2
    # - the cross-quadratics lambda_i \lambda_j
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
    monomial_mass = assemble_monomial_mass(exponents)

    # Our basis functions are given by
    # - nodes: lambda_i (2 lambda_i - 1)
    # - edges: 4 lambda_i lambda_j
    basis_nodes = np.vstack((-eye, zero, 2 * eye))
    basis_edges = np.zeros((2 * (dim + 1) + n_edges, n_edges))
    basis_edges[dim + 1 : dim + n_edges + 1] = 4 * np.eye(n_edges)
    basis = np.hstack((basis_nodes, basis_edges))

    return basis.T @ monomial_mass @ basis


def eval_grads_at_nodes(dphi):

    # Preallocaion

    # the gradient of our basis functions are given by
    # - nodes: grad lambda_i ( 4 lambda_i - 1 )
    # - edges: 4 lambda_i grad lambda_j + 4 lambda_j grad lambda_i

    # nodal dofs
    n_nodes = dphi.shape[1]
    Psi_nodes = np.zeros((n_nodes, 3 * n_nodes))
    for ind in np.arange(n_nodes):
        Psi_nodes[ind, 3 * ind : 3 * (ind + 1)] = 4 * dphi[:, ind]
    Psi_nodes[:n_nodes] -= np.tile(dphi.T, n_nodes)

    # edge dofs
    n_edges = get_num_edges(n_nodes - 1)
    if n_nodes == 2:
        Psi_edges = 4 * np.hstack((dphi[:, 1], dphi[:, 0]))
    elif n_nodes == 3:
        zero = np.zeros(3)
        Psi_edges = 4 * np.block(
            [
                [zero, dphi[:, 2], dphi[:, 1]],
                [dphi[:, 2], zero, dphi[:, 0]],
                [dphi[:, 1], dphi[:, 0], zero],
            ]
        )
    elif n_nodes == 4:
        zero = np.zeros(3)
        Psi_edges = 4 * np.block(
            [
                [zero, zero, dphi[:, 3], dphi[:, 2]],
                [zero, dphi[:, 3], zero, dphi[:, 1]],
                [zero, dphi[:, 2], dphi[:, 1], zero],
                [dphi[:, 3], zero, zero, dphi[:, 0]],
                [dphi[:, 2], zero, dphi[:, 0], zero],
                [dphi[:, 1], dphi[:, 0], zero, zero],
            ]
        )
    return np.vstack((Psi_nodes, Psi_edges))


def assemble_local_stiff(sd: pg.Grid, data: dict) -> np.ndarray:

    opposite_nodes = sd.compute_opposite_nodes()
    local_mass = pg.BDM1.local_inner_product(sd.dim)

    for c in np.arange(sd.num_cells):
        loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
        faces = opposite_nodes.indices[loc]
        nodes = opposite_nodes.data[loc]
        signs = sd.cell_faces.data[loc]

        dphi = -sd.face_normals[:, faces] * signs / (sd.dim * sd.cell_volumes[c])

        Psi = eval_grads_at_nodes(dphi)

        A = Psi @ local_mass @ Psi.T

        if sd.dim == 1:
            edges = np.array([c])
        elif sd.dim == 2:
            pass
        pass


if __name__ == "__main__":

    for dim in range(1, 4):
        print(get_local_edge_numbering(dim))
        # M = assemble_local_mass(dim)
        # print(M.sum())

    dim = 2
    mdg = pg.unit_grid(dim, 0.5)
    sd = mdg.subdomains()[0]

    assemble_local_stiff(sd, None)
    # )
