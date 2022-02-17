import numpy as np
import scipy.sparse as sps
import porepy as pp
from pygeon.geometry.geometry import signed_mortar_to_secondary

"""
Acknowledgements:
    The functionalities related to the curl computations are modified from
    github.com/anabudisa/md_aux_precond developed by Ana Budisa and Wietse M. Boon.
"""

# ----------------------------------div---------------------------------- #

def div(grid):
    if isinstance(grid, pp.Grid):
        return grid.cell_faces.T

    elif isinstance(grid, pp.GridBucket):
        return _gb_div(grid)


def _gb_div(gb):
    gb_div = np.empty(shape=(gb.size(), gb.size()), dtype=sps.spmatrix)

    # Local divergences
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_div[nn_g, nn_g] = div(g)

    # mortar contributions
    for e, d_e in gb.edges():
        # Get adjacent grids and mortar_grid
        g_down, g_up = gb.nodes_of_edge(e)
        mg = d_e['mortar_grid']

        # Get indices in grid_bucket
        nn_g_d = gb.node_props(g_down, 'node_number')
        nn_g_u = gb.node_props(g_up, 'node_number')
        nn_mg = d_e['edge_number'] + gb.num_graph_nodes()

        # Place in the matrix
        gb_div[nn_g_d, nn_mg] = - mg.mortar_to_secondary_int()
        gb_div[nn_g_u, nn_mg] = div(g_up) * signed_mortar_to_secondary(gb, e)

    return sps.bmat(gb_div, format='csc')


# ----------------------------------curl---------------------------------- #

def curl(grid):
    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return grid.face_edges.T

    elif isinstance(grid, pp.GridBucket):
        return _gb_curl(grid)


def _gb_curl(gb):
    gb_curl = np.empty(
        shape=(gb.size(), gb.num_graph_nodes()), dtype=sps.spmatrix)

    # Local curls
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_curl[nn_g, nn_g] = curl(g)

    # Jump terms
    for e, d_e in gb.edges():
        mg = d_e['mortar_grid']
        if mg.dim >= 1:
            # Get relevant grids
            g_down, g_up = gb.nodes_of_edge(e)

            # Get indices in grid_bucket
            nn_g_d = gb.node_props(g_down, 'node_number')
            nn_g_u = gb.node_props(g_up, 'node_number')

            # Place in the matrix
            gb_curl[nn_g_d, nn_g_u] = curl(mg)

    # Redistribute over interior and mortar dof
    for e, d_e in gb.edges():
        # Get grids
        g = gb.nodes_of_edge(e)[1]
        mg = d_e['mortar_grid']

        # Get indices in grid_bucket
        nn_g = gb.node_props(g, 'node_number')
        nn_mg = d_e['edge_number'] + gb.num_graph_nodes()

        Pi = mg.mortar_to_primary_int() * mg.primary_to_mortar_int()

        for ind in np.arange(np.size(gb_curl, 1)):
            loc_curl = gb_curl[nn_g, ind]

            if loc_curl is not None:
                gb_curl[nn_mg, ind] = signed_mortar_to_secondary(gb, e).T * loc_curl
                gb_curl[nn_g, ind] -= Pi * loc_curl

    return sps.bmat(gb_curl, format='csr')


# ----------------------------------grad---------------------------------- #

def grad(grid):
    if isinstance(grid, (pp.Grid, pp.MortarGrid)):
        return grid.edge_nodes.T
    elif isinstance(grid, pp.GridBucket):
        return _gb_grad(grid)


def _gb_grad(gb):
    gb_grad = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), dtype=sps.spmatrix)

    # Local grads
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_grad[nn_g, nn_g] = grad(g)

    # Jump terms
    for e, d_e in gb.edges():
        mg = d_e['mortar_grid']
        if mg.dim >= 2:
            # Get relevant grids
            g_down, g_up = gb.nodes_of_edge(e)

            # Get indices in grid_bucket
            nn_g_d = gb.node_props(g_down, 'node_number')
            nn_g_u = gb.node_props(g_up, 'node_number')

            # Place in the matrix
            gb_grad[nn_g_d, nn_g_u] = grad(mg)

    return sps.bmat(gb_grad, format='csr')