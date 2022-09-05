import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Nedelec0:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # Discretization of mass matrix
        self.mass_matrix_key = "mass"
        self.curl_matrix_key = "curl"

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
            return g.num_ridges
        else:
            raise ValueError

    def discretize(self, g: pg.Grid, data: dict):
        """Compute the mass matrix for a lowest-order Nedelec discretization

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
        matrix_dictionary[self.mass_matrix_key] = self.assemble_mass_matrix(g, data)
        matrix_dictionary[self.curl_matrix_key] = self.assemble_curl(g)

    def assemble_mass_matrix(self, g: pg.Grid, data: dict):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 6 * 6 * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(g.dim)

        cell_ridges = g.face_ridges.astype(bool) * g.cell_faces.astype(bool)
        ridge_peaks = g.ridge_peaks

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            peaks_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(peaks_loc, return_inverse=True)
            indices = np.reshape(indices, (2, -1))
            coords = g.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = self.local_grads(coords)

            # Compute a 6 x 12 matrix Psi such that Psi[i, j] = psi_i(x_j)
            Psi = np.empty((6, 4), np.ndarray)
            for (ridge, peaks) in enumerate(indices.T):
                Psi[ridge, peaks[0]] = dphi[:, peaks[1]]
                Psi[ridge, peaks[1]] = -dphi[:, peaks[0]]
            Psi = sps.bmat(Psi)

            # Compute the inner products
            A = Psi * M * Psi.T * g.cell_volumes[c]

            # Put in the right spot
            cols = np.tile(ridges_loc, (ridges_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            I[loc_idx] = cols.T.ravel()
            J[loc_idx] = cols.ravel()
            dataIJ[loc_idx] = A.todense().ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))

    def assemble_lumped_matrix(self, sd, data):
        tangents = sd.nodes * sd.ridge_peaks
        h = np.linalg.norm(tangents, axis=0)

        cell_ridges = np.abs(sd.face_ridges) * np.abs(sd.cell_faces)
        cell_ridges.data[:] = 1.0

        volumes = cell_ridges * sd.cell_volumes

        return sps.diags(volumes / (h * h))

    def local_grads(self, coord, dim=3):
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def local_inner_product(self, dim):
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        M = sps.lil_matrix((12, 12))
        for i in np.arange(3):
            range = np.arange(i, i + 12, 3)
            M[np.ix_(range, range)] = M_loc

        return M.tocsc()

    def assemble_curl(self, g):
        return pg.curl(g)

    def eval_at_cell_centers(self, g):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 6 * 3 * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(g.dim)

        cell_ridges = g.face_ridges.astype(bool) * g.cell_faces.astype(bool)
        ridge_peaks = g.ridge_peaks

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            peaks_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(peaks_loc, return_inverse=True)
            indices = np.reshape(indices, (2, -1))
            coords = g.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = self.local_grads(coords)

            Psi = np.zeros((3, 6))
            for (ridge, peaks) in enumerate(indices.T):
                Psi[:, ridge] = dphi[:, peaks[1]] - dphi[:, peaks[0]]

            # Put in the right spot
            loc_idx = slice(idx, idx + Psi.size)
            I[loc_idx] = np.repeat(np.arange(3), ridges_loc.size) + 3 * c
            J[loc_idx] = np.concatenate(3 * [[ridges_loc]]).ravel()
            dataIJ[loc_idx] = Psi.ravel() / 4.0
            idx += Psi.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))


class Nedelec1:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        self.lumped_matrix_key = "lumped"
        self.curl_matrix_key = "curl"

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
            return 2 * g.num_ridges
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
        matrix_dictionary[self.lumped_matrix_key] = self.assemble_lumped_matrix(g, data)
        matrix_dictionary[self.curl_matrix_key] = self.assemble_curl(g)

    def assemble_lumped_matrix(self, g: pg.Grid, data: dict):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 9 * 4 * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        cell_ridges = g.face_ridges.astype(bool) * g.cell_faces.astype(bool)
        ridge_peaks = g.ridge_peaks

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            dof_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            ).flatten()

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            coords = g.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = self.local_grads(coords)

            # Compute the local Nedelec basis functions and global indices
            Ne_basis = np.roll(dphi[:, indices], 6, axis=1)
            Ne_indices = np.concatenate((ridges_loc, ridges_loc + g.num_ridges))

            # Compute the inner products around each node
            for node in nodes_uniq:
                bf_is_at_node = dof_loc == node
                grads = Ne_basis[:, bf_is_at_node]
                A = grads.T @ grads
                A *= g.cell_volumes[c] / 4

                loc_ind = Ne_indices[bf_is_at_node]

                # Save values for stiff-H1 local matrix in the global structure
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)
                I[loc_idx] = cols.T.ravel()
                J[loc_idx] = cols.ravel()
                dataIJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))

    def local_grads(self, coord, dim=3):
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def assemble_curl(self, g):
        return sps.bmat([[pg.curl(g), -pg.curl(g)]]) / 2

    def eval_at_cell_centers(self, g):

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 12 * 3 * g.num_cells
        I = np.empty(size, dtype=int)
        J = np.empty(size, dtype=int)
        dataIJ = np.empty(size)
        idx = 0

        cell_ridges = g.face_ridges.astype(bool) * g.cell_faces.astype(bool)
        ridge_peaks = g.ridge_peaks

        for c in np.arange(g.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            dof_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            ).flatten()

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            coords = g.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = self.local_grads(coords)

            # Compute the local Nedelec basis functions and global indices
            Ne_basis = np.roll(dphi[:, indices], 6, axis=1)
            Ne_indices = np.concatenate((ridges_loc, ridges_loc + g.num_ridges))

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + Ne_basis.size)
            I[loc_idx] = np.repeat(np.arange(3), Ne_indices.size) + 3 * c
            J[loc_idx] = np.concatenate(3 * [[Ne_indices]]).ravel()
            dataIJ[loc_idx] = Ne_basis.ravel() / 4.0
            idx += Ne_basis.size

        # Construct the global matrices
        return sps.csr_matrix((dataIJ, (I, J)))
