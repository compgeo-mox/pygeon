import numpy as np
import scipy.sparse as sps
import pygeon as pg


def zero_tip_dofs(mdg, n_minus_k, **kwargs):
    """
    Compute the operator that maps the tip degrees of freedom to zero.

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the
            differential form.
        kwargs: Optional parameters
            to_sparse: Convert to a sparse matrix from sparse sub-blocks. Default True.

    Returns:
        sps.dia_matrix
    """
    to_sparse = kwargs.get("to_sparse", True)

    if n_minus_k == 0:
        return sps.diags(np.ones(mdg.num_subdomain_cells()), dtype=int)

    s = "tip_" + get_codim_str(n_minus_k)

    # Pre-allocation of the block-matrix
    is_tip_dof = np.empty(
        shape=(mdg.num_subdomains(), mdg.num_subdomains()), dtype=sps.spmatrix
    )
    for sd in mdg.subdomains():
        if sd.dim >= n_minus_k:
            # Get indice (node_numbers) in grid_bucket
            node_nr = mdg.subdomains().index(sd)
            # Add the sparse matrix
            is_tip_dof[node_nr, node_nr] = sps.diags(
                np.logical_not(sd.tags[s]), dtype=int
            )

    pg.bmat.replace_nones_with_zeros(is_tip_dof)
    return sps.bmat(is_tip_dof) if to_sparse else is_tip_dof


def remove_tip_dofs(mdg, n_minus_k, **kwargs):
    """
    Compute the operator that removes the tip degrees of freedom.

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the
            differential form

    Returns:
        sps.csr_matrix
    """
    R = zero_tip_dofs(mdg, n_minus_k, **kwargs).tocsr()
    return R[R.indices, :]


def get_codim_str(n_minus_k):
    """
    Helper function that returns the name of the mesh entity

    Parameters:
        n_minus_k (int): The codimension of the mesh entity

    Returns:
        str
    """
    return ["cells", "faces", "ridges", "peaks"][n_minus_k]
