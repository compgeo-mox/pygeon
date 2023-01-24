import numpy as np
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

        # Data allocation
        size = np.sum(np.square(cell_nodes.sum(0)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        for (cell, diam) in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            M_loc = self.assemble_loc_mass_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = M_loc.ravel()
            idx += cols.size

        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def assemble_loc_mass_matrix(self, sd: pg.Grid, cell, diam, nodes):
        """
        Computes the local VEM mass matrix on a given cell
        according to the Hitchhiker's (6.5)
        """

        proj = self.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        H = self.assemble_loc_monomial_mass(sd, cell, diam)

        D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return proj.T @ H @ proj + sd.cell_volumes[cell] * I_minus_Pi.T @ I_minus_Pi

    def assemble_loc_proj_to_mon(self, sd: pg.Grid, cell, diam, nodes):
        """
        Computes the local projection onto the monomials
        Returns the coefficients {a_i} in a_0 + [a_1, a_2] \dot (x - c) / d
        for each VL1 basis function.
        """

        G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
        B = self.assemble_loc_L2proj_rhs(sd, cell, diam, nodes)

        return np.linalg.solve(G, B)

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

    def assemble_loc_monomial_mass(self, sd: pg.Grid, cell, diam):
        """
        Computes the inner products of the monomials
        {1, (x - c)/d, (y - c)/d}
        Hitchhiker's (5.3)
        """
        H = np.zeros((3, 3))
        H[0, 0] = sd.cell_volumes[cell]

        M = np.ones((2, 2)) + np.eye(2)

        for face in sd.cell_faces[:, cell].indices:
            sub_volume = (
                np.dot(
                    sd.face_centers[:, face] - sd.cell_centers[:, cell],
                    sd.face_normals[:, face] * sd.cell_faces[face, cell],
                )
                / 2
            )

            vals = (
                sd.nodes[:2, sd.face_nodes[:, face].indices]
                - sd.cell_centers[:2, [cell] * 2]
            ) / diam

            H[1:, 1:] += sub_volume * vals @ M @ vals.T / 12

        return H

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

    def assemble_stiff_matrix(self, sd):
        """
        Returns the stiffness matrix

        Args
            sd : grid.

        Returns
            matrix: sparse (sd.num_nodes, sd.num_nodes)
                Stiffness matrix obtained from the discretization.

        """

        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Data allocation
        size = np.sum(np.square(cell_nodes.sum(0)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        for (cell, diam) in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            M_loc = self.assemble_loc_stiff_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = M_loc.ravel()
            idx += cols.size

        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def assemble_loc_stiff_matrix(self, sd: pg.Grid, cell, diam, nodes):
        """
        Computes the local VEM stiffness matrix on a given cell
        according to the Hitchhiker's (3.25)
        """

        proj = self.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        G = self.assemble_loc_L2proj_lhs(sd, cell, diam, nodes)
        G[0, :] = 0.0

        D = self.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return proj.T @ G @ proj + I_minus_Pi.T @ I_minus_Pi

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Returns the differential mapping in the discrete cochain complex.
        """

        pg.Lagrange1.assemble_diff_matrix(self, sd)

    def eval_at_cell_centers(self, sd: pg.Grid):

        eval = sd.cell_nodes()
        num_nodes = sps.diags(1.0 / sd.num_cell_nodes())

        return (eval @ num_nodes).T.tocsc()

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
