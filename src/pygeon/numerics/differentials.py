import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

"""
Acknowledgements:
    The functionalities related to the curl computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budiša and Wietse M. Boon.
"""

# ---------------------------------- Aliases ---------------------------------- #


def div(grid):
    """
    Compute the divergence.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.GridBucket).

    Returns:
        sps.csr_matrix. The divergence operator.
    """
    return exterior_derivative(grid, 1)


def curl(grid):
    """
    Compute the curl.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.GridBucket).

    Returns:
        sps.csr_matrix. The curl operator.
    """
    return exterior_derivative(grid, 2)


def grad(grid):
    """
    Compute the gradient.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.GridBucket).

    Returns:
        sps.csr_matrix. The gradient operator.
    """
    return exterior_derivative(grid, 3)


# --------------------------- MD exterior derivative --------------------------- #


def exterior_derivative(grid, n_minus_k):
    """
    Compute the (mixed-dimensional) exterior derivative for the differential forms of order n - k.

    Parameters:
        grid (pp.Grid, pp.MortarGrid, or pp.GridBucket).
        n_minus_k (int): The difference between the ambient dimension and the order of the differential form.

    Returns:
        sps.csr_matrix. The differential operator.
    """

    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return _g_exterior_derivative(grid, n_minus_k)

    elif isinstance(grid, pp.GridBucket):
        return _gb_exterior_derivative(grid, n_minus_k)

    else:
        raise TypeError(
            "Input needs to be of type pp.Grid, pp.MortarGrid, or pp.GridBucket"
        )


def _g_exterior_derivative(grid, n_minus_k):
    """
    Compute the exterior derivative on a grid.

    Parameters:
        grid (pp.Grid or pp.MortarGrid): The grid.
        n_minus_k (int): The difference between the ambient dimension and the order of the differential form.

    Returns:
        sps.csr_matrix. The differential operator.
    """

    if n_minus_k == 0:
        return sps.csr_matrix((0, grid.num_cells))
    elif n_minus_k == 1:
        return grid.cell_faces.T
    elif n_minus_k == 2:
        return grid.face_ridges.T
    elif n_minus_k == 3:
        return grid.ridge_peaks.T
    elif n_minus_k == 4:
        return sps.csr_matrix((grid.num_peaks, 0))
    else:
        Warning("(n - k) is not between 0 and 4")
        return sps.csr_matrix((0, 0))


def _gb_exterior_derivative(gb, n_minus_k):
    """
    Compute the mixed-dimensional exterior derivative on a grid bucket.

    Parameters:
        grid (pp.GridBucket): The grid bucket.
        n_minus_k (int): The difference between the ambient dimension and the order of the differential form.
    """

    # Pre-allocation of the block-matrix
    bmat = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), dtype=sps.spmatrix
    )

    # Compute local differential operator
    for g, d_g in gb:
        node_nr = d_g["node_number"]
        bmat[node_nr, node_nr] = exterior_derivative(g, n_minus_k)

    # Compute mixed-dimensional jump operator
    for e, d_e in gb.edges():
        # Get mortar_grid and adjacent grids
        mg = d_e["mortar_grid"]
        grids = gb.nodes_of_edge(e)

        if grids[1].dim >= n_minus_k:
            # Get indices (node_numbers) in grid_bucket
            node_nrs = [gb.node_props(g, "node_number") for g in grids]

            # Place the jump term in the block-matrix
            bmat[node_nrs[0], node_nrs[1]] = exterior_derivative(mg, n_minus_k)

    return sps.bmat(bmat, format="csc") * pg.numerics.restrictions.zero_tip_dofs(
        gb, n_minus_k
    )
