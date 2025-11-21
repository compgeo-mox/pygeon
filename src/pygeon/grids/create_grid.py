"""Create grids from various sources."""

import inspect
from typing import Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


def grid_from_domain(
    domain: pp.Domain, mesh_size: float, **kwargs
) -> Union[pg.Grid, pg.MixedDimensionalGrid]:
    """
    Create a grid from a domain with a specified mesh size.

    Args:
        domain (pp.Domain): The domain of the grid.
        mesh_size (float): The desired mesh size for the grid.
        **kwargs: Additional options for creating the grid:

            - mesh_size_min (float): The minimum mesh size. Default is mesh_size / 10.
            - as_mdg (bool): If True, return the grid as a pg.MixedDimensionalGrid.
              If False, return the grid as a pg.Grid.

    Returns:
        Either a pg.MixedDimensionalGrid or a pg.Grid, depending on the value of as_mdg.
    """
    as_mdg = kwargs.get("as_mdg", True)
    mesh_size_min = kwargs.get("mesh_size_min", mesh_size / 10)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size_min}

    # Inspect the signature of the function to get the valid parameters
    sig = inspect.signature(pp.create_fracture_network)
    sub_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    # Create the fracture network
    frac_net = pp.create_fracture_network(domain=domain, **sub_kwargs)

    # Inspect the signature of the function to get the valid parameters
    sig = inspect.signature(frac_net.mesh)
    sub_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    # Create the mesh
    mdg = frac_net.mesh(mesh_kwargs, **sub_kwargs)

    pg.convert_from_pp(mdg)
    if as_mdg:
        return mdg  # type: ignore[return-value]
    else:
        return mdg.subdomains(dim=mdg.dim_max())[0]  # type: ignore[return-value]


def grid_from_boundary_pts(
    pts: np.ndarray, mesh_size: float, **kwargs
) -> Union[pg.Grid, pg.MixedDimensionalGrid]:
    """
    Create a 2D grid from a set of nodes, where portions of the boundary of the grid
    are constructed from subsequent nodes, with a given mesh size.

    Args:
        pts (np.ndarray): The ordered points representing the boundary.
        mesh_size (float): The desired mesh size.
        **kwargs: Additional options:

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

    Args:
        dim (int): The dimension of the grid.
        mesh_size (float): The desired mesh size.
        kwargs: Additional options:

            - mesh_size_min (float): The minimum mesh size. Default is the same as
              mesh_size.
            - as_mdg (bool): If True, return the grid as a mixed-dimensional grid.
            - structured (bool): If True, create a structured grid.

    Returns:
        Either a pg.MixedDimensionalGrid or a pg.Grid.
    """
    if dim == 1 or kwargs.get("structured", False):
        num = np.array([1 / mesh_size] * dim, dtype=int)
        if dim == 1:
            sd = pp.CartGrid(num, np.ones(1))
        elif dim == 2:
            sd = pp.StructuredTriangleGrid(num, np.ones(dim))  # type: ignore[assignment]
        else:
            sd = pp.StructuredTetrahedralGrid(num, np.ones(dim))  # type: ignore[assignment]
        pg.convert_from_pp(sd)

        if kwargs.get("as_mdg", True):
            return pp.meshing.subdomains_to_mdg([[sd]])  # type: ignore[return-value]
        return sd  # type: ignore[return-value]

    bbox = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
    if dim == 3:
        bbox.update({"zmin": 0.0, "zmax": 1.0})

    domain = pp.Domain(bounding_box=bbox)
    return grid_from_domain(domain, mesh_size, **kwargs)


def reference_element(dim: int) -> pg.Grid:
    """
    Create a reference element of a given dimension.

    Args:
        dim (int): The dimension of the reference element.

    Returns:
        pg.Grid: The reference element.
    """
    if dim == 1:
        sd = unit_grid(1, 1, as_mdg=False)
        sd.name = "reference_segment"
        return sd  # type: ignore[return-value]
    elif dim == 2:
        nodes = np.eye(3, k=1)

        indices = np.array([1, 2, 0, 2, 0, 1])
        indptr = np.array([0, 2, 4, 6])
        face_nodes = sps.csc_array((np.ones(6), indices, indptr))

        cell_faces = sps.csc_array(np.array([1, -1, 1])[:, None])

        return pg.Grid(2, nodes, face_nodes, cell_faces, "reference_triangle")

    elif dim == 3:
        nodes = np.eye(3, 4, k=1)

        indices = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
        indptr = np.array([0, 3, 6, 9, 12])
        face_nodes = sps.csc_array((np.ones(12), indices, indptr))

        cell_faces = sps.csc_array(np.array([1, -1, 1, -1])[:, None])

        return pg.Grid(3, nodes, face_nodes, cell_faces, "reference_tetrahedron")
    else:
        raise ValueError("Dimension must be 1, 2, or 3.")
