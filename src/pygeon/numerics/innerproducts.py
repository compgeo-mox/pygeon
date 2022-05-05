import numpy as np
import scipy.sparse as sps

import porepy as pp

# ---------------------------------- Aliases ---------------------------------- #


def P0_mass(gb, discr, **kwargs):
    return mass_matrix(gb, discr, 0, **kwargs)


def hdiv_mass(gb, discr, **kwargs):
    return mass_matrix(gb, discr, 1, **kwargs)


def hcurl_mass(gb, discr, **kwargs):
    return mass_matrix(gb, discr, 2, **kwargs)


def hgrad_mass(gb, discr, **kwargs):
    return mass_matrix(gb, discr, 3, **kwargs)


# ---------------------------------- General ---------------------------------- #


def _g_mass_matrix(g, discr, n_minus_k, data):
    if n_minus_k == 0:
        return sps.diags(g.cell_volumes)
    elif n_minus_k == 1:
        discr.discretize(g, data)
        return data[pp.DISCRETIZATION_MATRICES]["flow"]["mass"]
    else:
        raise NotImplementedError


def mass_matrix(gb, discr, n_minus_k, local_matrix=_g_mass_matrix, return_bmat=False):

    bmat_g = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_g.copy()

    # Local mass matrices
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        bmat_g[nn_g, nn_g] = local_matrix(g, discr, n_minus_k, d_g)
        bmat_mg[nn_g, nn_g] = sps.csc_matrix(bmat_g[nn_g, nn_g].shape)

    # Mortar contribution
    if n_minus_k == 1:
        for e, d_e in gb.edges():
            # Get adjacent grids and mortar_grid
            g_up = gb.nodes_of_edge(e)[1]
            mg = d_e["mortar_grid"]

            # Get indices in grid_bucket
            nn_g = gb.node_props(g_up, "node_number")

            # Local mortar mass matrix
            kn = d_e["parameters"]["flow"]["normal_diffusivity"]
            bmat_mg[nn_g, nn_g] += (
                mg.signed_mortar_to_primary
                * sps.diags(1.0 / mg.cell_volumes / kn)
                * mg.signed_mortar_to_primary.T
            )

    if return_bmat:
        return bmat_g, bmat_mg
    else:
        return sps.bmat(bmat_g, format="csc") + sps.bmat(bmat_mg, format="csc")


# ---------------------------------- Lumped ---------------------------------- #


def lumped_mass_matrix(grid, discr, n_minus_k):
    return mass_matrix(grid, discr, n_minus_k, local_matrix=_g_lumped_mass)


def _g_lumped_mass(g, discr, n_minus_k, data):
    if n_minus_k == 0:
        return _g_mass_matrix(g, None, n_minus_k, None)

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

    elif n_minus_k == 2 and g.dim == 3:
        tangents = g.nodes * g.edge_nodes
        h = np.linalg.norm(tangents, axis=0)

        cell_edges = np.abs(g.face_edges) * np.abs(g.cell_faces)
        cell_edges.data[:] = 1.0

        volumes = cell_edges * g.cell_volumes

        return sps.diags(volumes / (h * h))

    elif n_minus_k == 2 and g.dim == 2:
        volumes = g.cell_nodes() * g.cell_volumes / (g.dim + 1)

        return sps.diags(volumes)

    elif n_minus_k == 2 and g.dim < 2:
        return sps.csc_matrix((g.num_edges, g.num_edges))

    elif n_minus_k == 3:
        cell_nodes = np.abs(g.edge_nodes) * np.abs(g.face_edges) * np.abs(g.cell_faces)
        cell_nodes.data[:] = 1.0

        volumes = cell_nodes * g.cell_volumes / (g.dim + 1)

        return sps.diags(volumes)

    else:
        raise NotImplementedError
