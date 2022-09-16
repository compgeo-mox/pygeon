import numpy as np
import porepy as pp
import scipy.sparse as sps


class Lagrange:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # Discretization of stiffness matrix
        self.stiffness_matrix_key = "stiffness"
        self.mass_matrix_key = "mass"
        self.lumped_matrix_key = "lumped"

    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pp.Grid):
            return g.num_nodes
        else:
            raise ValueError

    def discretize(self, g: pp.Grid, data: dict) -> None:
        """Set the stiffness and mass matrices

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Returns
        ------
        matrix: sparse csr (g.num_nodes, g.num_nodes)
            Matrix obtained from the discretization.

        Raises:
        ------
        ValueError if the boundary condition is not defined node-wise.
        """
        # Allow short variable names in backend function
        # pylint: disable=invalid-name

        # Get dictionary for discretization matrix storage
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        matrix_dictionary[self.stiffness_matrix_key] = self.assemble_stiffness_matrix(
            g, data
        )
        matrix_dictionary[self.mass_matrix_key] = self.assemble_mass_matrix(g, data)
        matrix_dictionary[self.lumped_matrix_key] = self.assemble_lumped_matrix(g)

    def assemble_mass_matrix(self, g, data):
        """
        Return the matrix for a discretization of a
        L2-mass bilinear form with P1 test and trial functions.

        The name of data in the input dictionary (data) are:
        phi: array (self.g.num_cells)
            Scalar values which represent the porosity.
            If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme.
            If not given assumed unitary.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data.

        Return
        ------
        matrix: sparse dia (g.num_cells, g_num_cells)
            Mass matrix obtained from the discretization.
        rhs: array (g_num_cells)
            Null right-hand side.

        """

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(g.dim + 1, 2) * g.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
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
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csr_matrix((data_IJ, (rows_I, cols_J)))

    def local_mass(self, c_volume, dim):
        """Compute the local mass matrix.

        Parameters
        ----------
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
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
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
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csr_matrix((data_IJ, (rows_I, cols_J)))

    def local_stiff(self, K, c_volume, coord, dim):
        """Compute the local stiffness matrix for P1.

        Parameters
        ----------
        K : ndarray (g.dim, g.dim)
            Permeability of the cell.
        c_volume : scalar
            Cell volume.

        Return
        ------
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

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = (g.dim + 1) * g.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            loc_idx = slice(idx, idx + nodes_loc.size)
            rows_I[loc_idx] = c
            cols_J[loc_idx] = nodes_loc
            data_IJ[loc_idx] = 1.0 / (g.dim + 1)
            idx += nodes_loc.size

        # Construct the global matrices
        return sps.csr_matrix((data_IJ, (rows_I, cols_J)))
