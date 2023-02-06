import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg

# ---------------------------------- Aliases ---------------------------------- #


def cell_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for the piecewise constants on a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_cells x num_cells
    """

    return mass_matrix(mdg, 0, discr, **kwargs)


def face_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the faces of a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp.RT0 or pp.MVEM).

    Returns:
        sps.csc_matrix, num_faces x num_faces
    """

    return mass_matrix(mdg, 1, discr, **kwargs)


def ridge_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the ridges of a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_ridges x num_ridges
    """

    return mass_matrix(mdg, 2, discr, **kwargs)


def peak_mass(mdg, discr=None, **kwargs):
    """
    Compute the mass matrix for discretization defined on the peaks of a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_matrix, num_peaks x num_peaks
    """

    return mass_matrix(mdg, 3, discr, **kwargs)


# ---------------------------------- General ---------------------------------- #


def default_discr(sd, n_minus_k, **kwargs):
    """
    Construct the default discretization operator depending on n_minus_k.
    These correspond to the Whitney forms.
    """
    keyword = kwargs.get("keyword", "flow")
    if n_minus_k == 0:
        return pg.PwConstants(keyword)
    elif n_minus_k == 1:
        return pg.RT0(keyword)
    elif n_minus_k == sd.dim:
        return pg.Lagrange1(keyword)
    elif n_minus_k == 2:  # The only remaining case is (k, sd.dim) = (1, 3)
        return pg.Nedelec0(keyword)
    else:
        raise ValueError


def _sd_mass_matrix(sd, n_minus_k, discr=None, data=None, **kwargs):
    """
    Compute the mass matrix on a single grid

    Args:
        sd (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of
            the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    if n_minus_k > sd.dim:
        return sps.csc_matrix((0, 0))

    if discr is None:
        discr = default_discr(sd, n_minus_k, **kwargs)

    return discr.assemble_mass_matrix(sd, data)


def local_matrix(sd, n_minus_k, discr, d_sd, **kwargs):
    if isinstance(sd, pg.Graph):
        return _sd_lumped_mass(sd, n_minus_k, discr, d_sd, **kwargs)
    elif isinstance(sd, pp.Grid):
        return _sd_mass_matrix(sd, n_minus_k, discr, d_sd, **kwargs)


def mass_matrix(mdg, n_minus_k, discr, local_matrix=local_matrix, **kwargs):
    """
    Compute the mass matrix on a mixed-dimensional grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of
            the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.
        local_matrix (function): function that generates the local mass matrix on a grid
        kwargs: Optional parameters
            to_sparse: In case of mixed-dimensional, convert to a sparse matrix from sparse sub-blocks. Default True.
            sps_format: Format of the sparse matrix when converted. Delfault csc.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    to_sparse = kwargs.get("to_sparse", True)
    sps_format = kwargs.get("sps_format", "csc")

    bmat_sd = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )
    bmat_mg = bmat_sd.copy()

    # Local mass matrices
    for (nn_sd, (sd, d_sd)) in enumerate(mdg.subdomains(return_data=True)):
        bmat_sd[nn_sd, nn_sd] = local_matrix(sd, n_minus_k, discr, d_sd, **kwargs)
        bmat_mg[nn_sd, nn_sd] = sps.csc_matrix(bmat_sd[nn_sd, nn_sd].shape)

    # Mortar contribution
    if n_minus_k == 1:
        for intf, d_intf in mdg.interfaces(return_data=True):
            # Get the node number of the upper-dimensional neighbor
            sd = mdg.interface_to_subdomain_pair(intf)[0]
            nn_sd = mdg.subdomains().index(sd)

            # Local mortar mass matrix
            try:
                kn = d_intf["parameters"][discr.keyword]["normal_diffusivity"]
            except KeyError:
                kn = 1

            bmat_mg[nn_sd, nn_sd] += (
                intf.signed_mortar_to_primary
                * sps.diags(1.0 / intf.cell_volumes / kn)
                * intf.signed_mortar_to_primary.T
            )

    pg.bmat.replace_nones_with_zeros(bmat_sd)
    pg.bmat.replace_nones_with_zeros(bmat_mg)

    # create the full block matrix
    bmat = bmat_sd + bmat_mg

    return sps.bmat(bmat, format=sps_format) if to_sparse else bmat

# ---------------------------------- Lumped ---------------------------------- #


def lumped_mass_matrix(mdg, n_minus_k, discr=None, **kwargs):
    """
    Compute the mass-lumped mass matrix on a mixed-dimensional grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of
            the differential.
        discr (pp discretization object).
        kwargs: Optional parameters
            to_sparse: In case of mixed-dimensional, convert to a sparse matrix from sparse sub-blocks. Default True.
            sps_format: Format of the sparse matrix when converted. Delfault csc.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """

    return mass_matrix(mdg, n_minus_k, discr, _sd_lumped_mass, **kwargs)


def _sd_lumped_mass(sd, n_minus_k, discr=None, data=None, **kwargs):
    """
    Compute the mass-lumped mass matrix on a single grid.

    Args:
        sd (pp.Grid).
        n_minus_k (int): The difference between the dimension and the order of
            the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.

    Returns:
        sps.csc_matrix, num_dofs x num_dofs
    """
    if n_minus_k > sd.dim:
        return sps.csc_matrix((0, 0))

    if discr is None:
        discr = default_discr(sd, n_minus_k, **kwargs)

    return discr.assemble_lumped_matrix(sd, data)
