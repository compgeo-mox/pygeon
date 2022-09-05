import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Lagrange1(pg.Discretization):
    def ndof(self, g: pp.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args
            g: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        return g.num_nodes

    def assemble_mass_matrix(self, g, data):
        """
        Returns the matrix for a discretization of a
        L2-mass bilinear form with P1 test and trial functions.

        The name of data in the input dictionary (data) are:
        phi: array (self.g.num_cells)
            Scalar values which represent the porosity.
            If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme.
            If not given assumed unitary.

        Args
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns
            matrix: sparse dia (g.num_cells, g_num_cells)
                Mass matrix obtained from the discretization.
            rhs: array (g_num_cells)
                Null right-hand side.

        """

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H1 local matrix
            A = self.local_mass(g.cell_volumes[c], g.dim)

            # Save values for mass-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csr_matrix((dataIJ, (I, J)))

    def local_mass(self, c_volume, dim):
        """Compute the local mass matrix.

        Args
        c_volume : scalar
            Cell volume.

        Return
        ------
        out: ndarray (num_faces_of_cell, num_faces_of_cell)
            Local mass Hdiv matrix.
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return c_volume * M / ((dim + 1) * (dim + 2))

    def assemble_stiffness_matrix(self, g, data):
        # If a 0-d grid is given then we return a zero matrix
        if g.dim == 0:
            return sps.csr_matrix((1, 1))

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(g)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if g.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.local_stiff(
                k.values[0 : g.dim, 0 : g.dim, c], g.cell_volumes[c], coord_loc, g.dim
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))

    def assemble_diff_matrix(self, sd: pg.Grid):
        if sd.dim == 3:
            return sd.ridge_peaks.T
        elif sd.dim == 2:
            return sd.face_ridges.T
        elif sd.dim == 1:
            return sd.cell_faces.T
        elif sd.dim == 0:
            return sps.csr_matrix((0, 1))
        else:
            raise ValueError

    def local_stiff(self, K, c_volume, coord, dim):
        """Compute the local stiffness matrix for P1.

        Args
            K : ndarray (g.dim, g.dim)
                Permeability of the cell.
            c_volume : scalar
                Cell volume.

        Returns
            out: ndarray (num_faces_of_cell, num_faces_of_cell)
                Local mass Hdiv matrix.
        """

        dphi = self.local_grads(coord, dim)

        return c_volume * np.dot(dphi.T, np.dot(K, dphi))

    @staticmethod
    def local_grads(coord, dim):
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def assemble_lumped_matrix(self, g, data=None):
        volumes = g.cell_nodes() * g.cell_volumes / (g.dim + 1)
        return sps.diags(volumes)

    def eval_at_cell_centers(self, g):

        # Allocation
        size = (g.dim + 1) * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            loc_idx = slice(idx, idx + nodes_loc.size)
            I[loc_idx] = c
            J[loc_idx] = nodes_loc
            dataIJ[loc_idx] = 1.0 / nodes_loc.size
            idx += nodes_loc.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))

    def interpolate(self, sd: pg.Grid, func):
        return np.array([func(x) for x in sd.nodes])

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1
        """
        vals = np.zeros(self.ndof(sd))

        for face in b_faces:
            loc = slice(sd.face_nodes.indptr[face], sd.face_nodes.indptr[face + 1])
            loc_n = sd.face_nodes.indices[loc]

            vals[loc_n] += (
                func(sd.face_centers[:, face]) * sd.face_areas[face] / loc_n.size
            )

        return vals

    def get_range_discr_class(self, dim):
        if dim == 3:
            return pg.Nedelec0
        elif dim == 2:
            return pg.RT0
        elif dim == 1:
            return pg.PwConstants
        else:
            raise NotImplementedError("There's no zero discretization in PyGeoN")
