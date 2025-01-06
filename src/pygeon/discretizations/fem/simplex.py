import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp
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


def get_local_edge_numbering(dim: int) -> np.ndarray:

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
    # - the cross-quadratics lambda_i \lambda_j
    # - the quadratics lambda_i^2
    quads = np.zeros((dim + 1, n_edges))
    e_nodes = get_local_edge_numbering(dim)
    for ind, nodes in enumerate(e_nodes):
        quads[nodes, ind] = 1
    exponents = np.hstack((eye, quads, 2 * eye))

    # Compute the local mass matrix of the monomials
    monomial_mass = assemble_monomial_mass(exponents)

    # Our basis functions are given by
    # - nodes: lambda_i (2 lambda_i - 1)
    # - edges: 4 lambda_i lambda_j
    basis_nodes = np.vstack((-eye, zero, 2 * eye))
    basis_edges = np.zeros((2 * (dim + 1) + n_edges, n_edges))
    basis_edges[dim + 1 : dim + n_edges + 1, :] = 4 * np.eye(n_edges)
    basis = np.hstack((basis_nodes, basis_edges))

    return basis.T @ monomial_mass @ basis


def eval_grads_at_nodes(dphi, e_nodes):

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
    Psi_edges = np.zeros((n_edges, 3 * n_nodes))

    for ind, (e0, e1) in enumerate(e_nodes):
        Psi_edges[ind, 3 * e0 : 3 * (e0 + 1)] = 4 * dphi[:, e1]
        Psi_edges[ind, 3 * e1 : 3 * (e1 + 1)] = 4 * dphi[:, e0]

    return np.vstack((Psi_nodes, Psi_edges))


def get_global_edge_nrs(sd, c, faces):
    # Find global edge number
    if sd.dim == 1:
        # The only edge in 1d is the cell
        edges = np.array([c])
    elif sd.dim == 2:
        # The edges (0, 1), (0, 2), and (1, 2)
        # are the faces opposite nodes 2, 1, and 0, respectively.
        edges = faces[::-1]
    elif sd.dim == 3:
        # We first find the edges adjacent to the local faces
        cell_edges = np.abs(sd.face_ridges[:, faces]) @ np.ones((4, 1))
        edge_inds = np.where(cell_edges)[0]

        # Experimentally, we always find the following numbering
        edges = edge_inds[[5, 4, 2, 3, 1, 0]]

    # The edge dofs come after the nodal dofs
    return edges + sd.num_nodes


def assemble_mass(sd: pg.Grid, data: dict) -> np.ndarray:

    size = np.square((sd.dim + 1) + get_num_edges(sd.dim)) * sd.num_cells
    rows_I = np.empty(size, dtype=int)
    cols_J = np.empty(size, dtype=int)
    data_IJ = np.empty(size)
    idx = 0

    opposite_nodes = sd.compute_opposite_nodes()
    local_mass = assemble_local_mass(sd.dim)

    for c in np.arange(sd.num_cells):
        loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
        faces = opposite_nodes.indices[loc]
        nodes = opposite_nodes.data[loc]
        edges = get_global_edge_nrs(sd, c, faces)

        A = local_mass.ravel() * sd.cell_volumes[c]

        loc_ind = np.hstack((nodes, edges))

        cols = np.tile(loc_ind, (loc_ind.size, 1))
        loc_idx = slice(idx, idx + cols.size)
        rows_I[loc_idx] = cols.T.ravel()
        cols_J[loc_idx] = cols.ravel()
        data_IJ[loc_idx] = A.ravel()
        idx += cols.size

    # Assemble
    return sps.csc_array((data_IJ, (rows_I, cols_J)))


def assemble_stiff(sd: pg.Grid, data: dict) -> np.ndarray:

    size = np.square((sd.dim + 1) + get_num_edges(sd.dim)) * sd.num_cells
    rows_I = np.empty(size, dtype=int)
    cols_J = np.empty(size, dtype=int)
    data_IJ = np.empty(size)
    idx = 0

    opposite_nodes = sd.compute_opposite_nodes()
    local_mass = pg.BDM1.local_inner_product(sd.dim)
    e_nodes = get_local_edge_numbering(sd.dim)

    for c in np.arange(sd.num_cells):
        loc = slice(opposite_nodes.indptr[c], opposite_nodes.indptr[c + 1])
        faces = opposite_nodes.indices[loc]
        nodes = opposite_nodes.data[loc]
        edges = get_global_edge_nrs(sd, c, faces)

        signs = sd.cell_faces.data[loc]
        dphi = -sd.face_normals[:, faces] * signs / (sd.dim * sd.cell_volumes[c])
        Psi = eval_grads_at_nodes(dphi, e_nodes)

        A = Psi @ local_mass @ Psi.T * sd.cell_volumes[c]

        loc_ind = np.hstack((nodes, edges))

        cols = np.tile(loc_ind, (loc_ind.size, 1))
        loc_idx = slice(idx, idx + cols.size)
        rows_I[loc_idx] = cols.T.ravel()
        cols_J[loc_idx] = cols.ravel()
        data_IJ[loc_idx] = A.ravel()
        idx += cols.size

    # Assemble
    return sps.csc_array((data_IJ, (rows_I, cols_J)))


if __name__ == "__main__":

    # for dim in range(1, 4):
    #     #     print(get_local_edge_numbering(dim))
    #     M = assemble_local_mass(dim)
    # print(M.sum())

    dim = 3
    # mdg = pg.unit_grid(dim, 0.15)
    # sd = mdg.subdomains()[0]

    sd = pp.StructuredTetrahedralGrid([10] * 3)
    # sd = pp.CartGrid(100, [1])
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    A = assemble_stiff(sd, None)

    source = np.ones(sd.num_nodes + sd.num_ridges)
    M = assemble_mass(sd, None)
    f = M @ source

    ess_bc = np.hstack(
        (sd.tags["domain_boundary_nodes"], sd.tags["domain_boundary_ridges"])
    )
    ess_vals = np.zeros_like(ess_bc, dtype=float)

    ridge_centers = sd.nodes @ np.abs(sd.ridge_peaks) / 2
    x = np.hstack((sd.nodes, ridge_centers))
    true_sol = np.sum(x * (1 - x), axis=0) / (2 * dim)

    ess_vals[ess_bc] = true_sol[ess_bc]

    LS = pg.LinearSystem(A, f)
    LS.flag_ess_bc(ess_bc, ess_vals)

    u = LS.solve()

    print(np.linalg.norm(u - true_sol))
    pass
    # )
