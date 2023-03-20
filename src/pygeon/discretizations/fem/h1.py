import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Lagrange1(pg.Discretization):
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

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H1 local matrix
            A = self.local_mass(sd.cell_volumes[c], sd.dim)

            # Save values for mass-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

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

    def assemble_stiffness_matrix(self, sd: pg.Grid, data: dict):
        # If a 0-d grid is given then we return a zero matrix
        if sd.dim == 0:
            return sps.csc_matrix((1, 1))

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(sd)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if sd.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.local_stiff(
                k.values[0 : sd.dim, 0 : sd.dim, c],
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid):
        if sd.dim == 3:
            return sd.ridge_peaks.T
        elif sd.dim == 2:
            return sd.face_ridges.T
        elif sd.dim == 1:
            return sd.cell_faces.T
        elif sd.dim == 0:
            return sps.csc_matrix((0, 1))
        else:
            raise ValueError

    def local_stiff(self, K, c_volume, coord, dim):
        """
        Compute the local stiffness matrix for P1.

        Args
            K : ndarray (dim, dim)
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

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        volumes = sd.cell_nodes() * sd.cell_volumes / (sd.dim + 1)
        return sps.diags(volumes)

    def eval_at_cell_centers(self, sd: pg.Grid):
        # Allocation
        size = (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            loc_idx = slice(idx, idx + nodes_loc.size)
            rows_I[loc_idx] = c
            cols_J[loc_idx] = nodes_loc
            data_IJ[loc_idx] = 1.0 / (sd.dim + 1)
            idx += nodes_loc.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def interpolate(self, sd: pg.Grid, func):
        return np.array([func(x) for x in sd.nodes.T])

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
