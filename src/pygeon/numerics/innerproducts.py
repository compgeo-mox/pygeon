import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

# ---------------------------------- Aliases ---------------------------------- #


def cell_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for the piecewise constants on a (MD-)grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_cells x num_cells
    """

    return mass_matrix(mdg, 0, discr, **kwargs)


def face_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the faces of a (MD-)grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp.RT0 or pp.MVEM).

    Returns:
        sps.csc_matrix, num_faces x num_faces
    """

    return mass_matrix(mdg, 1, discr, **kwargs)


def ridge_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the ridges of a (MD-)grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_ridges x num_ridges
    """

    return mass_matrix(mdg, 2, discr, **kwargs)


def peak_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the peaks of a (MD-)grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_peaks x num_peaks
    """

    return mass_matrix(mdg, 3, discr, **kwargs)


# ---------------------------------- General ---------------------------------- #
def default_discr(g, n_minus_k, keyword="flow"):
    """
    Construct the default discretization operator depending on n_minus_k.
    These correspond to the Whitney forms.
    """

    if n_minus_k == 0:
        return pg.PwConstants(keyword)
    elif n_minus_k == 1:
        return pp.RT0(keyword)
    elif n_minus_k == g.dim:
        return pg.Lagrange(keyword)
    elif n_minus_k == 2:
        return pg.Nedelec0(keyword)
    else:
        raise ValueError


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
    if n_minus_k > g.dim:
        return sps.csc_matrix((0, 0))

    if discr is None:
        discr = default_discr(g, n_minus_k)

    discr.discretize(g, data)
    return data[pp.DISCRETIZATION_MATRICES][discr.keyword]["mass"]


def local_matrix(sd, n_minus_k, discr, d_sd):
    if isinstance(sd, pg.Graph):
        return _g_lumped_mass(sd, n_minus_k, discr, d_sd)
    elif isinstance(sd, pp.Grid):
        return _g_mass_matrix(sd, n_minus_k, discr, d_sd)


def mass_matrix(mdg, n_minus_k, discr, local_matrix=local_matrix):
    """
    Compute the mass matrix on a mixed-dimensional grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.
        local_matrix (function): function that generates the local mass matrix on a grid

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    bmats = mass_matrix_bmats(mdg, n_minus_k, discr, local_matrix)

    return np.sum([sps.bmat(bmat, format="csc") for bmat in bmats])


def mass_matrix_bmats(mdg, n_minus_k, discr, local_matrix=local_matrix):
    """
    Computes the block matrices of the mass matrix
    """
    bmat_g = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_g.copy()

    # Local mass matrices
    for sd, d_sd in mdg.subdomains(return_data=True):
        nn_sd = d_sd["node_number"]
        bmat_g[nn_sd, nn_sd] = local_matrix(sd, n_minus_k, discr, d_sd)
        bmat_mg[nn_sd, nn_sd] = sps.csc_matrix(bmat_g[nn_sd, nn_sd].shape)

    # Mortar contribution
    if n_minus_k == 1:
        for intf, d_intf in mdg.interfaces(return_data=True):
            # Get the node number of the upper-dimensional neighbor
            sd_pair = mdg.interface_to_subdomain_pair(intf)
            nn_sd = mdg.node_number(sd_pair[0])

            # Local mortar mass matrix
            kn = d_intf["parameters"][discr.keyword]["normal_diffusivity"]
            bmat_mg[nn_sd, nn_sd] += (
                intf.signed_mortar_to_primary
                * sps.diags(1.0 / intf.cell_volumes / kn)
                * intf.signed_mortar_to_primary.T
            )

    return bmat_g, bmat_mg


# ---------------------------------- Lumped ---------------------------------- #


def lumped_mass_matrix(mdg, n_minus_k, discr):
    """
    Compute the mass-lumped mass matrix on a mixed-dimensional grid

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    return mass_matrix(mdg, n_minus_k, discr, _g_lumped_mass)


def _g_lumped_mass(g, n_minus_k, discr=None, data=None):
    """
    Compute the mass-lumped mass matrix on a single grid.
    For k = 1, this is the matrix that leads to a TPFA discretization.

    Parameters:
        g (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    if n_minus_k == 0:
        return _g_mass_matrix(g, n_minus_k, None, None)

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
