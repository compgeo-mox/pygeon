import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class PrimalVEM1(pg.Discretization):
    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args
            sd: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        return sd.num_nodes

    def assemble_mass_matrix(self, sd: pg.Grid, data=None):
        """
        Returns the mass matrix for the lowest order Lagrange element

        Args
            sd : grid.

        Returns
            matrix: sparse (sd.num_nodes, sd.num_nodes)
                Mass matrix obtained from the discretization.

        """
        raise NotImplementedError

    def assemble_diff_matrix(self, sd: pg.Grid):
        raise NotImplementedError

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):

        raise NotImplementedError

    def eval_at_cell_centers(self, sd: pg.Grid):
        raise NotImplementedError

    def interpolate(self, sd: pg.Grid, func):
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1
        """
        raise NotImplementedError

    def get_range_discr_class(self, dim):
        raise NotImplementedError
