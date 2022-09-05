import numpy as np
import porepy as pp
import scipy.sparse as sps

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


def default_discr(sd, n_minus_k, keyword="flow"):
    """
    Construct the default discretization operator depending on n_minus_k.
    These correspond to the Whitney forms.
    """
    if n_minus_k == 0:
        return pg.PwConstants(keyword)
    elif n_minus_k == 1:
        return pp.RT0(keyword)
    elif n_minus_k == sd.dim:
        return pg.Lagrange(keyword)
    elif n_minus_k == 2:
        return pg.Nedelec0(keyword)
    else:
        raise ValueError


def _sd_mass_matrix(sd, n_minus_k, discr=None, data=None, **kwargs):
    """
    Compute the mass matrix on a single grid

    Parameters:
        sd (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    if n_minus_k > sd.dim:
        return sps.csc_matrix((0, 0))

    if discr is None:
        discr = default_discr(sd, n_minus_k, **kwargs)

    discr.discretize(sd, data)
    return data[pp.DISCRETIZATION_MATRICES][discr.keyword]["mass"]


def local_matrix(sd, n_minus_k, discr, d_sd, **kwargs):
    if isinstance(sd, pg.Graph):
        return _sd_lumped_mass(sd, n_minus_k, discr, d_sd, **kwargs)
    elif isinstance(sd, pp.Grid):
        return _sd_mass_matrix(sd, n_minus_k, discr, d_sd, **kwargs)


def mass_matrix(mdg, n_minus_k, discr, local_matrix=local_matrix, **kwargs):
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


def mass_matrix_bmats(mdg, n_minus_k, discr, local_matrix=local_matrix, **kwargs):
    """
    Computes the block matrices of the mass matrix
    """
    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_sd.copy()

    # Local mass matrices
    for sd, d_sd in mdg.subdomains(return_data=True):
        nn_sd = d_sd["node_number"]
        bmat_sd[nn_sd, nn_sd] = local_matrix(sd, n_minus_k, discr, d_sd, **kwargs)
        bmat_mg[nn_sd, nn_sd] = sps.csc_matrix(bmat_sd[nn_sd, nn_sd].shape)

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

    return bmat_sd, bmat_mg


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

    return mass_matrix(mdg, n_minus_k, discr, _sd_lumped_mass)


def _sd_lumped_mass(sd, n_minus_k, discr=None, data=None, **kwargs):
    """
    Compute the mass-lumped mass matrix on a single grid.
    For k = 1, this is the matrix that leads to a TPFA discretization.

    Parameters:
        sd (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    if n_minus_k > sd.dim:
        return sps.csc_matrix((0, 0))

    if discr is None:
        discr = default_discr(sd, n_minus_k)

    if n_minus_k == 1:
        """
        Returns the lumped mass matrix L such that
        (div * L^-1 * div.T) is equivalent to a TPFA method
        TODO: Move this to RT0
        """
        h_perp = np.zeros(sd.num_faces)
        for (face, cell) in zip(*sd.cell_faces.nonzero()):
            h_perp[face] += np.linalg.norm(
                sd.face_centers[:, face] - sd.cell_centers[:, cell]
            )

        return sps.diags(h_perp / sd.face_areas)

    else:
        return discr.assemble_lumped_matrix(sd, data)
