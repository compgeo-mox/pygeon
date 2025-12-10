import numpy as np
import scipy.sparse as sps

import pygeon as pg


def lloyd_regularization(sd: pg.VoronoiGrid, num_iter: int) -> pg.VoronoiGrid:
    """
    Perform Lloyd's relaxation on the Voronoi grid. The topology of the grid is not preserved.

    Args:
        sd (pg.VoronoiGrid): The Voronoi grid to relax.
        num_iter (int): The number of iterations to perform.

    Returns:
        The relaxed Voronoi grid.
    """
    # Perform Lloyd's relaxation
    for _ in np.arange(num_iter):
        sd = pg.VoronoiGrid(vrt=sd.cell_centers)
        sd.compute_geometry()
    return sd


def graph_laplace_regularization(sd: pg.Grid, sliding: bool = True) -> pg.Grid:
    """
    Perform Laplace regularization on the grid by solving a graph laplacian over the
    face-ridges in 2d and ridge-peaks in 3d. The topology of the grid is preserved.

    Args:
        sd (pg.Grid): The grid to regularize.
        sliding (bool): Whether the boundary is sliding, defaults to True.

    Returns:
        The Laplace regularized grid.
    """
    # Construct the Laplacian matrix
    if sd.dim == 2:
        tags = sd.tags["domain_boundary_faces"]
        incidence = sd.face_ridges[:, np.logical_not(tags)]
    else:
        tags = sd.tags["domain_boundary_ridges"]
        incidence = sd.ridge_peaks[:, np.logical_not(tags)]

    A = incidence @ incidence.T
    A = sps.block_diag([A] * sd.dim, format="csc")

    # Assemble right-hand side
    b = -A @ sd.nodes[: sd.dim, :].ravel()

    ess = np.tile(sd.tags["domain_boundary_nodes"], sd.dim)
    u = compute_displacement(sd, A, b, sd.nodes, None if sliding else ess)
    return update_grid(sd, u)


def graph_laplace_dual_regularization(
    sd: pg.Grid, sliding: bool = True
) -> pg.VoronoiGrid:
    """
    Perform Laplace regularization on the dual grid by solving a graph laplacian over the
    cell-faces. A Voronoi grid is constructed based on the new cell centers.
    The topology of the grid is not preserved.

    Args:
        sd (pg.Grid): The grid to regularize.
        sliding (bool): Whether the boundary is sliding, defaults to True.

    Returns:
        The regularized Voronoi grid.
    """
    # ghost cells at the boundary
    bd_faces = sd.tags["domain_boundary_faces"]

    # create the new ghost cells based on the boundary faces
    ghost_cells = sps.diags(bd_faces, dtype=int, format="csc")
    ghost_cells = ghost_cells[:, ghost_cells.nonzero()[1]]

    # consider the sign of the normal vector at the boundary and switch it
    ghost_cells.eliminate_zeros()
    ghost_cells.data = -sd.cell_faces[bd_faces].tocsr().data

    incidence = sps.hstack([sd.cell_faces, ghost_cells])

    A = incidence.T @ incidence
    A = sps.block_diag([A] * sd.dim, format="csc")

    # Assemble right-hand side
    centers = np.hstack((sd.cell_centers, sd.face_centers[:, bd_faces]))
    b = -A @ centers[: sd.dim].ravel()

    # compute the new cell centers
    ess = np.tile(
        np.hstack(
            (
                np.zeros(sd.num_cells, dtype=bool),
                np.ones(ghost_cells.shape[1], dtype=bool),
            )
        ),
        sd.dim,
    )
    u = compute_displacement(sd, A, b, centers, None if sliding else ess)
    cell_centers = centers[: sd.dim] + u

    # build a voronoi grid based on the new cell centers
    sd = pg.VoronoiGrid(vrt=cell_centers)
    sd.compute_geometry()
    return sd


def elasticity_regularization(
    sd: pg.Grid, spring_const: float = 1, key: str = "reg", sliding: bool = True
) -> pg.Grid:
    """
    Regularize the grid using the elasticity regularization. The topology of the grid is
    preserved.

    Args:
        sd (pg.Grid): The grid to regularize.
        spring_const (float): The spring constant, defaults to 1.
        key (str): The key for the discretization, defaults to "reg".
        is_sliding (bool): Whether the boundary is sliding, defaults to True.

    Returns:
        The regularized grid.
    """
    discr = pg.VecVLagrange1(key)
    A = discr.assemble_stiff_matrix(sd)
    M = discr.assemble_mass_matrix(sd)

    # Assemble right-hand side
    nodes, cells, _ = sps.find(sd.cell_nodes())
    forces = (
        spring_const
        * (sd.cell_centers[:, cells] - sd.nodes[:, nodes])
        / sd.cell_volumes[cells]
    )
    force_list = [
        np.bincount(nodes, weights=forces[ind, :]) for ind in np.arange(sd.dim)
    ]
    b = M @ np.hstack(force_list)

    ess = np.tile(sd.tags["domain_boundary_nodes"], sd.dim)
    u = compute_displacement(sd, A, b, sd.nodes, None if sliding else ess)
    return update_grid(sd, u)


def compute_displacement(
    sd: pg.Grid,
    A: sps.csc_matrix,
    b: np.ndarray,
    coords: np.ndarray,
    ess: np.ndarray,
) -> np.ndarray:
    """
    Solve the regularizing system to compute the displacement field.

    Args:
        sd (pg.Grid): The grid to regularize.
        A (sps.csc_matrix): The system matrix.
        b (np.ndarray): The right-hand side.
        coords (np.ndarray): The coordinates of the nodes in the graph.
        ess (np.ndarray): The sliding degrees of freedom.

    Returns:
        The displacement field.
    """
    # Set the essential dofs for the sliding boundary
    if ess is None:
        box_min = np.min(coords, axis=1)
        box_max = np.max(coords, axis=1)

        bdry = [
            np.isclose(coords[ind, :], box_min[ind])
            + np.isclose(coords[ind, :], box_max[ind])
            for ind in np.arange(sd.dim)
        ]

        ess = np.hstack(bdry)

    # Solve the regularizing system
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess, np.zeros_like(ess, dtype=float))
    u = ls.solve()
    return u.reshape((sd.dim, -1))


def update_grid(sd: pg.Grid, u: np.ndarray) -> pg.Grid:
    """
    Update the grid with the displacement field by modifiying the node coordinates.

    Args:
        sd (pg.Grid): The grid to update.
        u (np.ndarray): The displacement field.

    Returns:
        The updated grid.
    """
    # Update the grid
    sd = sd.copy()
    sd.nodes[: sd.dim, :] += u
    sd.compute_geometry()
    return sd
