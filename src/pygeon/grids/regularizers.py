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


def laplace_regularization(sd: pg.Grid, key: str = "regularizer") -> pg.Grid:
    """
    Perform Laplace regularization on the grid.

    Args:
        sd (pg.Grid): The grid to regularize.

    Returns:
        The Laplace regularized grid.
    """
    # Construct the Laplacian matrix
    discr = pg.VLagrange1(key)

    # Assemble the stiffness matrix
    A = sps.block_diag([discr.assemble_stiff_matrix(sd, key)] * sd.dim)

    ess_nodes = np.tile(sd.tags["domain_boundary_nodes"], sd.dim)

    # Solve the Laplace equation
    ls = pg.LinearSystem(A, np.zeros(sd.num_nodes))
    ls.flag_ess_bc(ess_nodes, np.zeros(discr.ndof(sd)))
    u = ls.solve()

    # Update the grid
    sd.nodes = u

    return sd
