import numpy as np
import scipy.sparse as sps
import porepy as pp
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


def graph_laplace_regularization(sd: pg.Grid, is_sliding: bool = True) -> pg.Grid:
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
    A = sps.block_diag([A] * sd.dim)

    # Assemble right-hand side
    b = -A @ sd.nodes[: sd.dim, :].ravel()

    return regularize(sd, A, b, is_sliding)


def graph_laplace_regularization_with_centers(
    sd: pg.Grid, is_sliding: bool = True
) -> pg.Grid:
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
        incidence = sps.vstack((incidence, sps.csc_array((sd.num_cells, sd.num_faces))))
    else:
        incidence = sd.ridge_peaks

    nodes, cells, _ = sps.find(sd.cell_nodes().astype(int))
    num_cn = len(nodes)

    cc_n = sps.csc_matrix((-np.ones(num_cn), (nodes, np.arange(num_cn))))
    cc_c = sps.csc_matrix((np.ones(num_cn), (cells, np.arange(num_cn))))

    cc_inc = sps.vstack((cc_n, cc_c))

    incidence = sps.hstack((incidence, cc_inc))

    A = incidence @ incidence.T

    A = sps.block_diag([A] * sd.dim)

    # Assemble right-hand side
    coords = np.hstack((sd.nodes[: sd.dim, :], sd.cell_centers[: sd.dim, :])).ravel()
    b = -A @ coords

    # Set the essential dofs
    if is_sliding:
        box_min = np.min(sd.nodes, axis=1)
        box_max = np.max(sd.nodes, axis=1)

        bdry = [
            np.hstack(
                (
                    np.isclose(sd.nodes[ind, :], box_min[ind])
                    + np.isclose(sd.nodes[ind, :], box_max[ind]),
                    np.zeros(sd.num_cells, dtype=bool),
                )
            )
            for ind in np.arange(sd.dim)
        ]

        ess_nodes = np.hstack(bdry)
    else:
        ess_nodes = sd.tags["domain_boundary_nodes"]
        ess_nodes = np.hstack((ess_nodes, np.zeros(sd.num_cells, dtype=bool)))
        ess_nodes = np.tile(ess_nodes, sd.dim)

    # Solve the regularizing system
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess_nodes, np.zeros_like(ess_nodes, dtype=float))
    u = ls.solve()
    u = u.reshape((sd.dim, -1))

    # Update the grid
    sd = sd.copy()
    sd.nodes[: sd.dim, :] += u[:, : sd.num_nodes]
    sd.compute_geometry()

    return sd


def elasticity_regularization(
    sd: pg.Grid, spring_const: float = 1, key: str = "reg", is_sliding: bool = True
) -> pg.Grid:

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

    return regularize(sd, A, b, is_sliding)


def regularize(sd, A, b, is_sliding):

    # Set the essential dofs
    if is_sliding:
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

    # Solve the regularizing system
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess_nodes, np.zeros_like(ess_nodes, dtype=float))
    u = ls.solve()
    u = u.reshape((sd.dim, -1))

    # Update the grid
    sd = sd.copy()
    sd.nodes[: sd.dim, :] += u
    sd.compute_geometry()

    return sd
