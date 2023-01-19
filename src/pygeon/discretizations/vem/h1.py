import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class VLagrange1(pg.Discretization):
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
        Returns the mass matrix

        Args
            sd : grid.

        Returns
            matrix: sparse (sd.num_nodes, sd.num_nodes)
                Mass matrix obtained from the discretization.

        """

        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        for (cell, diam) in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes = cell_nodes.indices[loc]

            G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
            B = self.assemble_loc_L2proj_rhs(sd, cell, diam, nodes)
            D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)

            print("dude")
        raise NotImplementedError

    def assemble_loc_L2proj_lhs(self, sd: pg.Grid, cell, diam, nodes):
        """
        Returns the system G from the hitchhiker's (3.9)
        """

        G = sd.cell_volumes[cell] / (diam**2) * np.eye(3)
        G[0, 0] = 1
        G[0, 1:] = (
            sd.nodes[: sd.dim, nodes].mean(1) - sd.cell_centers[: sd.dim, cell]
        ) / diam

        return G

    def assemble_loc_L2proj_rhs(self, sd: pg.Grid, cell, diam, nodes):
        """
        Returns the righthand side B from the hitchhiker's (3.14)
        """

        normals = (
            sd.face_normals[: sd.dim] * sd.cell_faces[:, cell].A.ravel()
        ) @ sd.face_nodes[nodes, :].T

        B = np.empty((3, nodes.size))
        B[0, :] = 1.0 / nodes.size
        B[1:, :] = normals / diam / 2

        return B

    def assemble_loc_dofs_of_monomials(self, sd: pg.Grid, cell, diam, nodes):
        """
        Returns the matrix D from the hitchhiker's (3.17)
        """

        D = np.empty((nodes.size, 3))
        D[:, 0] = 1.0
        D[:, 1:] = (
            sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim, [cell] * nodes.size]
        ).T / diam

        return D

    def assemble_loc_monomial_mass(self, sd: pg.Grid, cell, diam):
        """
        Computes the inner products of the monomials
        {1, (x - c)/d, (y - c)/d}
        """
        H = np.zeros((3, 3))
        H[0, 0] = 1.0
        raise NotImplementedError
        return sd.cell_volumes[cell] * H

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Returns the differential mapping in the discrete cochain complex.
        """

        pg.Lagrange1.assemble_diff_matrix(self, sd)

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
