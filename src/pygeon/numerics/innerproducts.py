import numpy as np
import scipy.sparse as sps

import porepy as pp

# ---------------------------------- Aliases ---------------------------------- #


def cell_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for the piecewise constants on a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_cells x num_cells
    """

    return mass_matrix(mdg, 0, discr, **kwargs)


def face_mass(mdg, discr=pp.RT0, **kwargs):
    """
    Compute the mass matrix for discretization defined on the faces of a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp.RT0 or pp.MVEM).

    Returns:
        sps.csc_matrix, num_faces x num_faces
    """

    return mass_matrix(mdg, 1, discr, **kwargs)


def ridge_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the ridges of a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_ridges x num_ridges
    """

    return mass_matrix(mdg, 2, discr, **kwargs)


def peak_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the peaks of a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_peaks x num_peaks
    """

    return mass_matrix(mdg, 3, discr, **kwargs)


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


def mass_matrix(mdg, n_minus_k, discr, local_matrix=_g_mass_matrix, return_bmat=False):
    """
    Compute the mass matrix on a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
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
        shape=(mdg.num_graph_nodes(), mdg.num_graph_nodes()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_g.copy()

    # Local mass matrices
    for sd, d_sd in mdg.subdomains(return_data=True):
        nn_g = d_sd["node_number"]
        bmat_g[nn_g, nn_g] = local_matrix(sd, n_minus_k, discr, d_sd)
        bmat_mg[nn_g, nn_g] = sps.csc_matrix(bmat_g[nn_g, nn_g].shape)

    # Mortar contribution
    if n_minus_k == 1:
        #for e, d_e in mdg.interfaces(return_data=True):
        for intf, d_e in mdg.interfaces(return_data=True):
            # Get adjacent grids and mortar_grid
            pair = mdg.interface_to_subdomain_pair(intf)

            # Get indices in grid_bucket
            nn_g = mdg.subdomain_data(pair[0])["node_number"]

            # Local mortar mass matrix
            kn = d_intf["parameters"]["flow"]["normal_diffusivity"]
            bmat_mg[nn_g, nn_g] += (
                intf.signed_mortar_to_primary
                * sps.diags(1.0 / intf.cell_volumes / kn)
                * intf.signed_mortar_to_primary.T
            )

    if return_bmat:
        return bmat_g, bmat_mg
    else:
        return sps.bmat(bmat_g, format="csc") + sps.bmat(bmat_mg, format="csc")


# ---------------------------------- Lumped ---------------------------------- #


def lumped_mass_matrix(mdg, n_minus_k, discr):
    """
    Compute the mass-lumped mass matrix on a grid bucket

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    return mass_matrix(mdg, n_minus_k, discr, local_matrix=_g_lumped_mass)


def _g_lumped_mass(g, n_minus_k, discr, data):
    """
    Compute the mass-lumped mass matrix on a single grid.
    For k = 1, this is the matrix that leads to a TPFA discretization.

    Parameters:
        mdg (pp.MixedDimensionalGrid).
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
