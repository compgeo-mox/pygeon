import numpy as np
import scipy.sparse as sps

import porepy as pp

# ---------------------------------- Aliases ---------------------------------- #


def P0_mass(grid, discr, data=None):
    return mass_matrix(grid, discr, 0, data)


def hdiv_mass(grid, discr, data=None):
    return mass_matrix(grid, discr, 1, data)


def hcurl_mass(grid, discr, data=None):
    return mass_matrix(grid, discr, 2, data)


def hgrad_mass(grid, discr, data=None):
    return mass_matrix(grid, discr, 3, data)


# ---------------------------------- General ---------------------------------- #


def mass_matrix(grid, discr, n_minus_k, data=None):
    if isinstance(grid, pp.Grid):
        return _g_mass(grid, discr, n_minus_k, data)
    elif isinstance(grid, pp.GridBucket):
        return _gb_mass(grid, discr, n_minus_k)


def _g_mass(g, discr, n_minus_k, data):
    if n_minus_k == 0:
        return sps.diags(g.cell_volumes)
    elif n_minus_k == 1:
        discr.discretize(g, data)
        return data[pp.DISCRETIZATION_MATRICES]["flow"]["mass"]
    else:
        raise NotImplementedError


def _gb_mass(gb, discr, n_minus_k, local_matrix=mass_matrix):
    bmat = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), dtype=sps.spmatrix
    )

    # Local mass matrices
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        bmat[nn_g, nn_g] = local_matrix(g, discr, n_minus_k, d_g)

    # Mortar contribution
    if n_minus_k > 0:
        for e, d_e in gb.edges():
            # Get adjacent grids and mortar_grid
            g_up = gb.nodes_of_edge(e)[1]
            mg = d_e["mortar_grid"]

            # Get indices in grid_bucket
            nn_g = gb.node_props(g_up, "node_number")

            # Local mortar mass matrix
            kn = d_e["parameters"]["flow"]["normal_diffusivity"]
            bmat[nn_g, nn_g] += (
                mg.signed_mortar_to_primary
                * sps.diags(1.0 / mg.cell_volumes / kn)
                * mg.signed_mortar_to_primary.T
            )

    return sps.bmat(bmat, format="csc")


# ---------------------------------- Lumped ---------------------------------- #


def lumped_mass_matrix(grid, discr, n_minus_k, data=None):
    if isinstance(grid, pp.Grid):
        return _g_lumped_mass(grid, n_minus_k)
    elif isinstance(grid, pp.GridBucket):
        return _gb_mass(grid, discr, n_minus_k, local_matrix=lumped_mass_matrix)


def _g_lumped_mass(g, n_minus_k):
    if n_minus_k == 0:
        return _g_mass(g, None, n_minus_k, None)
    elif n_minus_k == 1:
        """
        Returns the lumped mass matrix L such that
        (div * L^-1 * div) is equivalent to a TPFA method
        """
        h_perp = np.zeros(g.num_faces)
        for (face, cell) in zip(*g.cell_faces.nonzero()):
            h_perp[face] += np.linalg.norm(
                g.face_centers[:, face] - g.cell_centers[:, cell]
            )

        return sps.diags(h_perp / g.face_areas)

    else:
        raise NotImplementedError
