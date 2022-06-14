import numpy as np
import scipy.sparse as sps

import porepy as pp

# ---------------------------------- Aliases ---------------------------------- #


def cell_mass(gb, discr=None, **kwargs):
    """
    Compute the mass matrix for the piecewise constants on a grid bucket

    Parameters:
        gb (pp.GridBucket).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_cells x num_cells
    """

    return mass_matrix(gb, 0, discr, **kwargs)


def face_mass(gb, discr=pp.RT0, **kwargs):
    """
    Compute the mass matrix for discretization defined on the faces of a grid bucket

    Parameters:
        gb (pp.GridBucket).
        discr (pp.RT0 or pp.MVEM).

    Returns:
        sps.csc_matrix, num_faces x num_faces
    """

    return mass_matrix(gb, 1, discr, **kwargs)


def ridge_mass(gb, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the ridges of a grid bucket

    Parameters:
        gb (pp.GridBucket).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_ridges x num_ridges
    """

    return mass_matrix(gb, 2, discr, **kwargs)


def peak_mass(gb, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the peaks of a grid bucket

    Parameters:
        gb (pp.GridBucket).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_peaks x num_peaks
    """

    return mass_matrix(gb, 3, discr, **kwargs)


# ---------------------------------- General ---------------------------------- #


def _g_mass_matrix(g, n_minus_k, discr=None, data=None):
    """
    Compute the mass matrix on a single grid

    Parameters:
        g (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    if n_minus_k == 0:
        return sps.diags(g.cell_volumes)
    elif n_minus_k == 1:
        discr.discretize(g, data)
        return data[pp.DISCRETIZATION_MATRICES]["flow"]["mass"]
    else:
        raise NotImplementedError


def mass_matrix(gb, n_minus_k, discr, local_matrix=_g_mass_matrix, return_bmat=False):
    """
    Compute the mass matrix on a grid bucket

    Parameters:
        gb (pp.GridBucket).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.
        local_matrix (function): function that generates the local mass matrix on a grid
        return_bmat (bool): Set to True to return the unassembled block matrix

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
        (if return_bmat) np.array of sps.spmatrices
    """

    bmat_g = np.empty(
        shape=(gb.num_graph_nodes(), gb.num_graph_nodes()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_g.copy()

    # Local mass matrices
    for g, d_g in gb:
        nn_g = d_g["node_number"]
        bmat_g[nn_g, nn_g] = local_matrix(g, n_minus_k, discr, d_g)
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


def lumped_mass_matrix(gb, n_minus_k, discr):
    """
    Compute the mass-lumped mass matrix on a grid bucket

    Parameters:
        gb (pp.GridBucket).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    return mass_matrix(gb, n_minus_k, discr, local_matrix=_g_lumped_mass)


def _g_lumped_mass(g, n_minus_k, discr, data):
    """
    Compute the mass-lumped mass matrix on a single grid.
    For k = 1, this is the matrix that leads to a TPFA discretization.

    Parameters:
        gb (pp.GridBucket).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

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
        tangents = g.nodes * g.ridge_peaks
        h = np.linalg.norm(tangents, axis=0)

        cell_ridges = np.abs(g.face_ridges) * np.abs(g.cell_faces)
        cell_ridges.data[:] = 1.0

        volumes = cell_ridges * g.cell_volumes

        return sps.diags(volumes / (h * h))

    elif n_minus_k == 2 and g.dim == 2:
        volumes = g.cell_nodes() * g.cell_volumes / (g.dim + 1)

        return sps.diags(volumes)

    elif n_minus_k == 2 and g.dim < 2:
        return sps.csc_matrix((g.num_ridges, g.num_ridges))

    elif n_minus_k == 3:
        cell_nodes = (
            np.abs(g.ridge_peaks) * np.abs(g.face_ridges) * np.abs(g.cell_faces)
        )
        cell_nodes.data[:] = 1.0

        volumes = cell_nodes * g.cell_volumes / (g.dim + 1)

        return sps.diags(volumes)

    else:
        raise NotImplementedError