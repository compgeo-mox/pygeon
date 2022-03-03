import numpy as np
import scipy.sparse as sps

import porepy as pp
from pygeon.geometry.geometry import signed_mortar_to_primary

def hdiv_mass(grid, discr, data, data_key="flow"):
    if isinstance(grid, pp.Grid):
        data = pp.initialize_default_data(grid, {}, data_key)
        discr.discretize(grid, data)
        return data[pp.DISCRETIZATION_MATRICES][data_key]["mass"]
    elif isinstance(grid, pp.GridBucket):
        return _gb_hdiv_mass(grid, discr, data_key)

def _gb_hdiv_mass(gb, discr, data_key):
    gb_hdiv_mass = np.empty(shape=(gb.size(), gb.size()), dtype=sps.spmatrix)

    # Local mass matrices
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_hdiv_mass[nn_g, nn_g] = hdiv_mass(g, discr, d_g, data_key)

        # Take care of homogeneous, essential bcs at fracture tips
        Pi = sps.diags(g.tags['tip_faces'].astype(np.int))
        # Zero out rows
        gb_hdiv_mass[nn_g, nn_g] -= Pi * gb_hdiv_mass[nn_g, nn_g]
        # Zero out columns
        gb_hdiv_mass[nn_g, nn_g] -= gb_hdiv_mass[nn_g, nn_g] * Pi
        # Put a one on the diagonal
        gb_hdiv_mass[nn_g, nn_g] += Pi

    # Mortar 
    for e, d_e in gb.edges():
        # Get adjacent grids and mortar_grid
        g_up = gb.nodes_of_edge(e)[1]
        mg = d_e['mortar_grid']

        # Get indices in grid_bucket
        nn_g = gb.node_props(g_up, 'node_number')
        nn_mg = d_e['edge_number'] + gb.num_graph_nodes()

        # Local mortar mass matrix
        kn = 1 # TODO retrieve normal permeability from data
        gb_hdiv_mass[nn_mg, nn_mg] = sps.diags(mg.cell_volumes) / kn

        # Inner products of mortar extension into primary domain
        gb_hdiv_mass[nn_mg, nn_mg] += signed_mortar_to_primary(gb, e).T * \
            gb_hdiv_mass[nn_g, nn_g] * signed_mortar_to_primary(gb, e)
        
        gb_hdiv_mass[nn_g, nn_mg] = gb_hdiv_mass[nn_g, nn_g] * \
            signed_mortar_to_primary(gb, e)

        # Incorporate essential bc on fracture interfaces
        Pi = mg.mortar_to_primary_int() * mg.primary_to_mortar_int()
        gb_hdiv_mass[nn_g, nn_g] -= Pi * gb_hdiv_mass[nn_g, nn_g]
        gb_hdiv_mass[nn_g, nn_g] -= gb_hdiv_mass[nn_g, nn_g] * Pi
        gb_hdiv_mass[nn_g, nn_g] += Pi

        # Zero out rows in the off-diagonal blocks
        gb_hdiv_mass[nn_g, nn_mg] -= Pi * gb_hdiv_mass[nn_g, nn_mg]
        gb_hdiv_mass[nn_mg, nn_g] = gb_hdiv_mass[nn_g, nn_mg].T

    return sps.bmat(gb_hdiv_mass, format='csc')

def lumped_mass_TPFA(g):
    """
    Returns the lumped mass matrix L such that
    (div * L^-1 * div) is equivalent to a TPFA method
    """
    h_perp = np.zeros(g.num_faces)
    for (face, cell) in zip(*g.cell_faces.nonzero()):
        h_perp[face] += np.linalg.norm(g.face_centers[:,
                                       face] - g.cell_centers[:, cell])
    return sps.spdiags(h_perp / g.face_areas)

