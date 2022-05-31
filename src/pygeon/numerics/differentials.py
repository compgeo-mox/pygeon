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
    return exterior_derivative(grid, 1)


def curl(grid):
    return exterior_derivative(grid, 2)


def grad(grid):
    return exterior_derivative(grid, 3)


# --------------------------- MD exterior derivative --------------------------- #


def exterior_derivative(grid, n_minus_k):
    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return _g_exterior_derivative(grid, n_minus_k)

    elif isinstance(grid, pp.GridBucket):
        return _gb_exterior_derivative(grid, n_minus_k)

    else:
        raise TypeError(
            "Input needs to be of type pp.Grid, pp.MortarGrid, or pp.GridBucket"
        )


def _g_exterior_derivative(grid, n_minus_k):
    if n_minus_k == 1:
        return grid.cell_faces.T
    elif n_minus_k == 2:
        return grid.face_ridges.T
    elif n_minus_k == 3:
        return grid.ridge_peaks.T
    else:
        raise ValueError("(n - k) needs to be between 3 and 1")


def _gb_exterior_derivative(gb, n_minus_k):
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
