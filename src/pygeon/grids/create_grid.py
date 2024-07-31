""" Create grids from various sources. """

from typing import Union

import numpy as np
import porepy as pp

import pygeon as pg


def grid_from_domain(
    domain: pp.Domain, mesh_size: float, **kwargs
) -> Union[pg.Grid, pg.MixedDimensionalGrid]:
    """
    Create a grid from a domain with a specified mesh size.

    Args:
        domain (pp.Domain): The domain of the grid.
        mesh_size (float): The desired mesh size for the grid.
        **kwargs: Additional options for creating the grid.
            mesh_size_min (float): The minimum mesh size. Default is mesh_size / 10.
            as_mdg (bool): If True, return the grid as a pg.MixedDimensionalGrid.
                           If False, return the grid as a pg.Grid.

    Returns:
        Either a pg.MixedDimensionalGrid or a pg.Grid, depending on the value of as_mdg.
    """
    as_mdg = kwargs.pop("as_mdg", True)
    mesh_size_min = kwargs.pop("mesh_size_min", mesh_size / 10)

    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size_min}
    mdg = pp.create_fracture_network(domain=domain).mesh(mesh_kwargs, **kwargs)

    pg.convert_from_pp(mdg)
    if as_mdg:
        return mdg
    else:
        return mdg.subdomains(dim=mdg.dim_max())[0]


def grid_from_boundary_pts(
    pts: np.ndarray, mesh_size: float, **kwargs
) -> Union[pg.Grid, pg.MixedDimensionalGrid]:
    """
    Create a 2D grid from a set of nodes, where portions of the boundary of the grid
    are constructed from subsequent nodes, with a given mesh size.

    Parameters:
        pts (np.ndarray): The ordered points representing the boundary.
        mesh_size (float): The desired mesh size.
        **kwargs: Additional options. The following options are available:
            - mesh_size_min (float): The minimum mesh size. Default is mesh_size.
            - as_mdg (bool): Return the grid as a mixed-dimensional grid.

    Returns:
        Either a pg.MixedDimensionalGrid or a pg.Grid.
    """
    # Create the lines representing the boundary of the domain
    idx = np.arange(pts.shape[1])
    order = np.vstack((idx, np.roll(idx, -1))).flatten(order="F")
    lines = np.split(pts[:, order], pts.shape[1], axis=1)

    # Create the domain
    domain = pp.Domain(polytope=lines)
    return grid_from_domain(domain, mesh_size, **kwargs)


def unit_grid(
    dim: int, mesh_size: float, **kwargs
) -> Union[pg.Grid, pg.MixedDimensionalGrid]:
    """
    Create a unit square or cube grid with a given mesh size.

    Parameters:
        dim (int): The dimension of the grid. Must be 2 or 3.
        mesh_size (float): The desired mesh size.
        kwargs: Additional options. The following options are available:
            - mesh_size_min (float): The minimum mesh size. Default is the same as mesh_size.
            - as_mdg (bool): If True, return the grid as a mixed-dimensional grid.

    Returns:
        Either a pg.MixedDimensionalGrid or a pg.Grid.
    """

    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    if dim == 3:
        bbox.update({"zmin": 0, "zmax": 1})

    domain = pp.Domain(bounding_box=bbox)
    return grid_from_domain(domain, mesh_size, **kwargs)
