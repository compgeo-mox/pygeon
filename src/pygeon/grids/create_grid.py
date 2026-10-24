"""Create grids from various sources."""

import inspect
from typing import Literal, cast, overload

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


@overload
def grid_from_domain(
    domain: pp.Domain, mesh_size: float, as_mdg: Literal[False], **kwargs
) -> pg.Grid: ...


@overload
def grid_from_domain(
    domain: pp.Domain, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.MixedDimensionalGrid: ...


def grid_from_domain(
    domain: pp.Domain, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.Grid | pg.MixedDimensionalGrid:
    """
    Create a grid from a domain with a specified mesh size.

    Args:
        domain (pp.Domain): The domain of the grid.
        mesh_size (float): The desired mesh size for the grid.
        as_mdg (bool): Return the grid as a mixed-dimensional grid. Defaults to True.
        **kwargs: Additional options for creating the grid: the ones of
            pp.create_fracture_network are passed to it, the others to pp.create_mdg.

    Returns:
        pg.Grid | pg.MixedDimensionalGrid: The mixed-dimensional grid, or its grid of
        the highest dimension if as_mdg is False.
    """
    mesh_kwargs = {"cell_size": mesh_size}

    # Inspect the signature of the function to get the valid parameters
    sig = inspect.signature(pp.create_fracture_network)
    fracnet_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    # Create the fracture network
    frac_net = pp.create_fracture_network(domain=domain, **fracnet_kwargs)

    # Create the mixed-dimensional grid
    create_mdg_kwargs = {k: v for k, v in kwargs.items() if k not in fracnet_kwargs}
    mdg = pp.create_mdg("simplex", mesh_kwargs, frac_net, **create_mdg_kwargs)

    mdg = pg.convert_from_pp(mdg)
    if as_mdg:
        return mdg
    else:
        sd = mdg.subdomains(dim=mdg.dim_max())[0]
        return cast(pg.Grid, sd)


@overload
def grid_from_boundary_pts(
    pts: np.ndarray, mesh_size: float, as_mdg: Literal[False], **kwargs
) -> pg.Grid: ...


@overload
def grid_from_boundary_pts(
    pts: np.ndarray, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.MixedDimensionalGrid: ...


def grid_from_boundary_pts(
    pts: np.ndarray, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.Grid | pg.MixedDimensionalGrid:
    """
    Create a 2D grid from a set of nodes, where portions of the boundary of the grid
    are constructed from subsequent nodes, with a given mesh size.

    Args:
        pts (np.ndarray): The ordered points representing the boundary.
        mesh_size (float): The desired mesh size.
        as_mdg (bool): Return the grid as a mixed-dimensional grid. Defaults to True.
        **kwargs: Additional options, passed to grid_from_domain.

    Returns:
        pg.Grid | pg.MixedDimensionalGrid: The grid, as in grid_from_domain.
    """
    # Create the lines representing the boundary of the domain
    idx = np.arange(pts.shape[1])
    order = np.vstack((idx, np.roll(idx, -1))).flatten(order="F")
    lines = np.split(pts[:, order], pts.shape[1], axis=1)

    # Create the domain
    domain = pp.Domain(polytope=lines)
    return grid_from_domain(domain, mesh_size, as_mdg, **kwargs)


# Overloads to make mypy understand when we return a Grid or a MixedDimensionalGrid
@overload
def unit_grid(
    dim: int, mesh_size: float, as_mdg: Literal[False], **kwargs
) -> pg.Grid: ...


@overload
def unit_grid(
    dim: int, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.MixedDimensionalGrid: ...


def unit_grid(
    dim: int, mesh_size: float, as_mdg: bool = True, **kwargs
) -> pg.Grid | pg.MixedDimensionalGrid:
    """
    Create a unit square or cube grid with a given mesh size. In dimension 0 and 1,
    and when structured is True, the grid is structured, with 1 / mesh_size cells
    per direction; otherwise it is an unstructured simplicial grid.

    Args:
        dim (int): The dimension of the grid.
        mesh_size (float): The desired mesh size.
        as_mdg (bool): Return the grid as a mixed-dimensional grid. Defaults to True.
        **kwargs: Additional options:

            - structured (bool): If True, create a structured grid.
            - the others are passed to grid_from_domain for the unstructured grids.

    Returns:
        pg.Grid | pg.MixedDimensionalGrid: The grid, as in grid_from_domain.
    """
    if dim <= 1 or kwargs.get("structured", False):
        num = np.array([1 / mesh_size] * dim, dtype=int)
        sd: pp.Grid
        if dim == 0:
            sd = pp.PointGrid(np.array([[0.0], [0.0], [0.0]]))
        elif dim == 1:
            sd = pp.CartGrid(num, np.ones(1))
        elif dim == 2:
            sd = pp.StructuredTriangleGrid(num, np.ones(dim))
        else:
            sd = pp.StructuredTetrahedralGrid(num, np.ones(dim))

        if as_mdg:
            return pg.as_mdg(sd)
        else:
            return pg.convert_from_pp(sd)

    bbox = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
    if dim == 3:
        bbox.update({"zmin": 0.0, "zmax": 1.0})

    domain = pp.Domain(bounding_box=bbox)
    return grid_from_domain(domain, mesh_size, as_mdg, **kwargs)


def reference_element(dim: int) -> pg.Grid:
    """
    Create a reference element of a given dimension.

    Args:
        dim (int): The dimension of the reference element.

    Returns:
        pg.Grid: The reference element.

    Raises:
        ValueError: If the dimension is not 1, 2 or 3.
    """
    match dim:
        case 1:
            sd = unit_grid(1, 1, as_mdg=False)
            sd.name = "reference_segment"
            return sd
        case 2:
            nodes = np.eye(pg.AMBIENT_DIM, 3, k=1)

            indices = np.array([1, 2, 0, 2, 0, 1])
            indptr = np.array([0, 2, 4, 6])
            face_nodes = sps.csc_array((np.ones(6), indices, indptr))

            cell_faces = sps.csc_array(np.array([1, -1, 1])[:, None])

            return pg.Grid(2, nodes, face_nodes, cell_faces, "reference_triangle")
        case 3:
            nodes = np.eye(pg.AMBIENT_DIM, 4, k=1)

            indices = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
            indptr = np.array([0, 3, 6, 9, 12])
            face_nodes = sps.csc_array((np.ones(12), indices, indptr))

            cell_faces = sps.csc_array(np.array([1, -1, 1, -1])[:, None])

            return pg.Grid(3, nodes, face_nodes, cell_faces, "reference_tetrahedron")
        case _:
            raise ValueError("Dimension must be 1, 2, or 3.")
