import numpy as np
import scipy.sparse as sps

import porepy as pp

def mass_matrix(gb, discr, n_minus_k):
    if n_minus_k == 0:
        return P0_mass(gb)
    if n_minus_k == 1:
        return hdiv_mass(gb, discr)
    else:
        raise NotImplementedError


def P0_mass(gb:pp.GridBucket):
    bmat = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()),
        dtype=sps.spmatrix
        )
    
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        bmat[nn_g, nn_g] = sps.diags(g.cell_volumes)
    
    return sps.bmat(bmat)

def hdiv_mass(grid, discr, data=None, data_key="flow"):
    if isinstance(grid, pp.Grid):
        discr.discretize(grid, data)
        return data[pp.DISCRETIZATION_MATRICES][data_key]["mass"]
    elif isinstance(grid, pp.GridBucket):
        return _gb_hdiv_mass(grid, discr, data_key)

def _gb_hdiv_mass(gb, discr, data_key):
    gb_hdiv_mass = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), 
        dtype=sps.spmatrix
        )

    # Local mass matrices
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        gb_hdiv_mass[nn_g, nn_g] = hdiv_mass(g, discr, d_g, data_key)

    # Mortar 
    for e, d_e in gb.edges():
        # Get adjacent grids and mortar_grid
        g_up = gb.nodes_of_edge(e)[1]
        mg = d_e['mortar_grid']

        # Get indices in grid_bucket
        nn_g = gb.node_props(g_up, 'node_number')

        # Local mortar mass matrix
        kn = d_e['parameters'][data_key]['normal_diffusivity']
        gb_hdiv_mass[nn_g, nn_g] += mg.signed_mortar_to_primary * \
            sps.diags(1.0 / mg.cell_volumes / kn) * \
            mg.signed_mortar_to_primary.T

    return sps.bmat(gb_hdiv_mass, format='csc')

def lumped_mass_TPFA(g, return_inverse=False):
    """
    Returns the lumped mass matrix L such that
    (div * L^-1 * div) is equivalent to a TPFA method
    """
    h_perp = np.zeros(g.num_faces)
    for (face, cell) in zip(*g.cell_faces.nonzero()):
        h_perp[face] += np.linalg.norm(g.face_centers[:,
                                       face] - g.cell_centers[:, cell])
    
    if return_inverse:
        return sps.diags(g.face_areas / h_perp)
    else:
        return sps.diags(h_perp / g.face_areas)
