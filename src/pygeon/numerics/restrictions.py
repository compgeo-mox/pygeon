import numpy as np
import scipy.sparse as sps


def zero_tip_dofs(mdg, n_minus_k):
    """
    Compute the operator that maps the tip degrees of freedom to zero.

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the
            differential form

    Returns:
        sps.dia_matrix
    """

    if n_minus_k == 0:
        return sps.diags(np.ones(mdg.num_subdomain_cells()), dtype=int)

    str = "tip_" + get_codim_str(n_minus_k)

    is_tip_dof = []
    for sd in mdg.subdomains():
        if sd.dim >= n_minus_k:
            is_tip_dof.append(sd.tags[str])

    if len(is_tip_dof) > 0:
        is_tip_dof = np.concatenate(is_tip_dof)

    return sps.diags(np.logical_not(is_tip_dof), dtype=int)


def remove_tip_dofs(mdg, n_minus_k):
    """
    Compute the operator that removes the tip degrees of freedom.

    Parameters:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of the
            differential form

    Returns:
        sps.csr_matrix
    """
    R = zero_tip_dofs(mdg, n_minus_k).tocsr()
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
