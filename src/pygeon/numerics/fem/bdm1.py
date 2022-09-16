import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class BDM1:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        self.lumped_matrix_key = "lumped"
        self.div_matrix_key = "div"

    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of ridges.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pg.Grid):
            return g.num_faces * g.dim
        else:
            raise ValueError

    def discretize(self, g: pg.Grid, data: dict):
        """Compute the mass matrix for a first-order Nedelec discretization

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. See self.matrix_rhs for required contents.

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

        assert g.dim == 3

        # Get dictionary for discretization matrix storage
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary[self.matrix_key] = self.assemble_matrix(g, data)

    def assemble_matrix(self, g: pg.Grid, data: dict):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.square(g.dim * (g.dim + 1)) * g.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(g.dim)

        cell_nodes = g.cell_nodes()
        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = g.cell_faces.indices[loc]
            dof_loc = np.reshape(
                g.face_nodes[:, faces_loc].indices, (g.dim, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            indices = np.unique(dof_loc, return_inverse=True)[1].reshape((g.dim, -1))

            face_nodes_loc = g.face_nodes[:, faces_loc].toarray()
            cell_nodes_loc = cell_nodes[:, c].toarray()
            # get the opposite node id for each face
            opposite_node = np.logical_xor(face_nodes_loc, cell_nodes_loc)

            # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
            Psi = np.empty((g.dim * (g.dim + 1), g.dim + 1), np.ndarray)
            for (face, nodes) in enumerate(indices.T):
                tangents = (
                    g.nodes[:, face_nodes_loc[:, face]]
                    - g.nodes[:, opposite_node[:, face]]
                )
                normal = g.face_normals[:, faces_loc[face]]
                for (index, node) in enumerate(nodes):
                    Psi[face + index * (g.dim + 1), node] = tangents[:, index] / np.dot(
                        tangents[:, index], normal
                    )
            Psi = sps.bmat(Psi)

            # Compute the inner products
            A = Psi * M * Psi.T * g.cell_volumes[c]

            loc_ind = np.hstack([faces_loc] * g.dim)
            loc_ind += np.repeat(np.arange(g.dim), g.dim + 1) * g.num_faces

            # Save values of the local matrix in the global structure
            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.todense().ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_inner_product(self, dim):
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        M = sps.lil_matrix((3 * (dim + 1), 3 * (dim + 1)))
        for i in np.arange(3):
            mask = np.arange(i, i + 3 * (dim + 1), 3)
            M[np.ix_(mask, mask)] = M_loc

        return M.tocsc()

    def assemble_div(self, g):
        return sps.bmat([[pg.div(g)]*g.dim]) / g.dim

    def eval_at_cell_centers(self, g):
        raise NotImplemented

    def assemble_lumped_matrix(self, g: pg.Grid, data: dict):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = g.dim * g.dim * (g.dim + 1) * g.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = g.cell_nodes()
        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
            faces_loc = g.cell_faces.indices[loc]
            dof_loc = np.reshape(
                g.face_nodes[:, faces_loc].indices, (g.dim, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            indices = indices.reshape((g.dim, -1))

            face_nodes_loc = g.face_nodes[:, faces_loc].toarray()
            cell_nodes_loc = cell_nodes[:, c].toarray()
            # get the opposite node id for each face
            opposite_node = np.logical_xor(face_nodes_loc, cell_nodes_loc)

            # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
            Bdm_basis = np.zeros((3, g.dim * (g.dim + 1)))
            Bdm_indices = np.hstack([faces_loc] * g.dim)
            Bdm_indices += np.repeat(np.arange(g.dim), g.dim + 1) * g.num_faces

            for (face, nodes) in enumerate(indices.T):
                tangents = (
                    g.nodes[:, face_nodes_loc[:, face]]
                    - g.nodes[:, opposite_node[:, face]]
                )
                normal = g.face_normals[:, faces_loc[face]]
                for (index, node) in enumerate(nodes):
                    Bdm_basis[:, face + index * (g.dim + 1)] = tangents[:, index] / np.dot(
                        tangents[:, index], normal
                    )

            for node in nodes_uniq:
                bf_is_at_node = dof_loc.flatten() == node
                basis = Bdm_basis[:, bf_is_at_node]
                A = basis.T @ basis
                A *= g.cell_volumes[c] / (g.dim + 1)

                loc_ind = Bdm_indices[bf_is_at_node]

                # Save values for the local matrix in the global structure
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)
                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_IJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))
