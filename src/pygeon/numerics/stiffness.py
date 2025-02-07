""" This module contains functions for computing the stiffness operators. """

from typing import Union, Optional

import pygeon as pg
import scipy.sparse as sps

# ---------------------------------- Aliases ---------------------------------- #


def cell_stiff(
    mdg: pg.MixedDimensionalGrid, discr: Optional[pg.Discretization] = None, **kwargs
) -> sps.csc_array:
    """
    Compute the stiffness matrix for the piecewise constants on a (MD-)grid

    Args:
        mdg (pg.MixedDimensionalGrid): The mixed-dimensional grid.
        discr (pp discretization object): The discretization object.

    Returns:
        sps.csc_array, num_cells x num_cells
    """

    return stiff_matrix(mdg, 0, discr, **kwargs)


def face_stiff(
    mdg: pg.MixedDimensionalGrid, discr: Optional[pg.Discretization] = None, **kwargs
) -> sps.csc_array:
    """
    Compute the stiffness matrix for discretization defined on the faces of a (MD-)grid

    Args:
        mdg (pg.MixedDimensionalGrid).
        discr (pp.RT0 or pp.MVEM).

    Returns:
        sps.csc_array, num_faces x num_faces
    """

    return stiff_matrix(mdg, 1, discr, **kwargs)


def ridge_stiff(
    mdg: Union[pg.Grid, pg.MixedDimensionalGrid],
    discr: Optional[pg.Discretization] = None,
    **kwargs
) -> sps.csc_array:
    """
    Compute the stiffness matrix for discretization defined on the ridges of a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_array, num_ridges x num_ridges
    """

    return stiff_matrix(mdg, 2, discr, **kwargs)


def peak_stiff(
    mdg: pg.MixedDimensionalGrid, discr: Optional[pg.Discretization] = None, **kwargs
) -> sps.csc_array:
    """
    Compute the stiffness matrix for discretization defined on the peaks of a (MD-)grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        discr (pp discretization object).

    Returns:
        sps.csc_array, num_peaks x num_peaks
    """

    return stiff_matrix(mdg, 3, discr, **kwargs)


# ---------------------------------- General ---------------------------------- #


def stiff_matrix(
    mdg: pg.MixedDimensionalGrid,
    n_minus_k: int,
    discr: Union[pg.Discretization, None],
    **kwargs
) -> sps.csc_array:
    """
    Compute the stiffness matrix on a mixed-dimensional grid

    Args:
        mdg (pp.MixedDimensionalGrid).
        n_minus_k (int): The difference between the dimension and the order of
            the differential.
        discr (pp discretization object).
        data (dict): the data object associated to the grid.
        local_matrix (function): function that generates the local mass matrix on a grid

    Returns:
        sps.csc_array, num_dofs x num_dofs
    """

    if n_minus_k == 0:
        return sps.csc_array((mdg.num_subdomain_cells(), mdg.num_subdomain_cells()))

    diff = pg.numerics.differentials.exterior_derivative(mdg, n_minus_k)
    mass_plus_1 = pg.numerics.innerproducts.mass_matrix(
        mdg, n_minus_k - 1, discr, **kwargs
    )

    return (diff.T @ mass_plus_1 @ diff).tocsc()
