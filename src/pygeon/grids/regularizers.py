import numpy as np
import scipy.sparse as sps
import pygeon as pg


def lloyd_regularization(sd, num_iter: int) -> pg.VoronoiGrid:
    """
    Perform Lloyd's relaxation on the Voronoi grid.

    Args:
        sd (pg.Grid): The Voronoi grid to relax.
        num_iter (int): The number of iterations to perform.

    Returns:
        The relaxed Voronoi grid.
    """
    # Perform Lloyd's relaxation
    for _ in np.arange(num_iter):
        sd = pg.VoronoiGrid(vrt=sd.cell_centers)
        sd.compute_geometry()

    return sd


def graph_laplace_regularization(sd: pg.Grid) -> pg.Grid:
    """
    Perform Laplace regularization on the grid.

    Args:
        sd (pg.Grid): The grid to regularize.

    Returns:
        The Laplace regularized grid.
    """
    # Construct the Laplacian matrix
    if sd.dim == 2:
        incidence = sd.face_ridges
    else:
        incidence = sd.ridge_peaks

    A = incidence @ incidence.T

    # Preserve boundary nodes
    ess_nodes = sd.tags["domain_boundary_nodes"]

    # Assemble right-hand side
    b = -A @ sd.nodes[: sd.dim, :].T

    # Solve the Graph-Laplace equation
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess_nodes, np.zeros_like(ess_nodes, dtype=float))
    u = ls.solve()

    # Update the grid
    sd = sd.copy()
    sd.nodes[: sd.dim, :] += u.T
    sd.compute_geometry()

    return sd


def elasticity_regularization(
    sd: pg.Grid, spring_const: float = 1, key: str = "reg", is_square: bool = True
) -> pg.Grid:

    discr = pg.VecVLagrange1(key)
    A = discr.assemble_stiff_matrix(sd)
    M = discr.assemble_mass_matrix(sd)

    # Set the essential dofs
    if is_square:
        box_min = np.min(sd.nodes, axis=1)
        box_max = np.max(sd.nodes, axis=1)

        bdry = [
            np.isclose(sd.nodes[ind, :], box_min[ind])
            + np.isclose(sd.nodes[ind, :], box_max[ind])
            for ind in np.arange(sd.dim)
        ]

        ess_nodes = np.hstack(bdry)
    else:
        ess_nodes = sd.tags["domain_boundary_nodes"]
        ess_nodes = np.tile(ess_nodes, sd.dim)

    # Assemble right-hand side
    nodes, cells, _ = sps.find(sd.cell_nodes())
    forces = spring_const * (sd.cell_centers[:, cells] - sd.nodes[:, nodes])
    force_list = [
        np.bincount(nodes, weights=forces[ind, :]) for ind in np.arange(sd.dim)
    ]
    b = M @ np.hstack(force_list)

    # Solve the elasticity equation
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess_nodes, np.zeros_like(ess_nodes, dtype=float))
    u = ls.solve()
    u = u.reshape((sd.dim, -1))

    # Update the grid
    sd = sd.copy()
    sd.nodes[: sd.dim, :] += u
    sd.compute_geometry()

    return sd
