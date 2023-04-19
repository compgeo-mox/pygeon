import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Nedelec0(pg.Discretization):
    """
    Discretization class for the Nedelec of the first kind of lowest order.
    Each degree of freedom is the integral over a mesh edge in 3D.
    """

    def ndof(self, sd: pp.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of ridges.

        Parameter
        ---------
        sd: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return sd.num_ridges

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Computes the mass matrix for a lowest-order Nedelec discretization

        Parameters
        ----------
        sd: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. See self.matrix_rhs for required contents.

        Returns
        ------
        matrix: sparse csc (sd.num_ridges, sd.num_ridges)
            Matrix obtained from the discretization.

        """
        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 6 * 6 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(sd.dim)

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)
        ridge_peaks = sd.ridge_peaks

        for c in np.arange(sd.num_cells):
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
            coords = sd.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = pg.Lagrange1.local_grads(coords, sd.dim)

            # Compute a 6 x 12 matrix Psi such that Psi[i, j] = psi_i(x_j)
            Psi = np.empty((6, 4), np.ndarray)
            for ridge, peaks in enumerate(indices.T):
                Psi[ridge, peaks[0]] = dphi[:, peaks[1]]
                Psi[ridge, peaks[1]] = -dphi[:, peaks[0]]
            Psi = sps.bmat(Psi)

            # Compute the inner products
            A = Psi * M * Psi.T * sd.cell_volumes[c]

            # Put in the right spot
            cols = np.tile(ridges_loc, (ridges_loc.size, 1))
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

        M = sps.lil_matrix((12, 12))
        for i in np.arange(3):
            range = np.arange(i, i + 12, 3)
            M[np.ix_(range, range)] = M_loc

        return M.tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid):
        return sd.face_ridges.T

    def eval_at_cell_centers(self, sd):
        # Allocation
        size = 6 * 3 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)
        ridge_peaks = sd.ridge_peaks

        for c in np.arange(sd.num_cells):
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
            coords = sd.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = pg.Lagrange1.local_grads(coords, sd.dim)

            Psi = np.zeros((3, 6))
            for ridge, peaks in enumerate(indices.T):
                Psi[:, ridge] = dphi[:, peaks[1]] - dphi[:, peaks[0]]

            # Put in the right spot
            loc_idx = slice(idx, idx + Psi.size)
            rows_I[loc_idx] = np.repeat(np.arange(3), ridges_loc.size) + 3 * c
            cols_J[loc_idx] = np.concatenate(3 * [[ridges_loc]]).ravel()
            data_IJ[loc_idx] = Psi.ravel() / 4.0
            idx += Psi.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        raise NotImplementedError

    def get_range_discr_class(self, sd: pg.Grid):
        return pg.RT0

    def interpolate(self, sd: pg.Grid, func):
        tangents = sd.nodes * sd.ridge_peaks
        midpoints = sd.nodes * np.abs(sd.ridge_peaks) / 2
        vals = [
            np.inner(func(x).flatten(), t) for (x, t) in zip(midpoints.T, tangents.T)
        ]
        return np.array(vals)


class Nedelec1(pg.Discretization):
    """
    Discretization class for the Nedelec of the second kind of lowest order.
    Each degree of freedom is a first moment over a mesh edge in 3D.
    """

    def ndof(self, sd: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of ridges.

        Parameter
        ---------
        sd: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return 2 * sd.num_ridges

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        raise NotImplementedError

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        # Allocation
        size = 9 * 4 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)
        ridge_peaks = sd.ridge_peaks

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            dof_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            ).flatten()

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            coords = sd.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = pg.Lagrange1.local_grads(coords, sd.dim)

            # Compute the local Nedelec basis functions and global indices
            Ne_basis = np.roll(dphi[:, indices], 6, axis=1)
            Ne_indices = np.concatenate((ridges_loc, ridges_loc + sd.num_ridges))

            # Compute the inner products around each node
            for node in nodes_uniq:
                bf_is_at_node = dof_loc == node
                grads = Ne_basis[:, bf_is_at_node]
                A = grads.T @ grads
                A *= sd.cell_volumes[c] / 4

                loc_ind = Ne_indices[bf_is_at_node]

                # Save values for stiff-H1 local matrix in the global structure
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)
                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_IJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def proj_to_Ne0(self, sd: pg.Grid):
        return sps.hstack([sps.eye(sd.num_ridges), -sps.eye(sd.num_ridges)]) / 2

    def assemble_diff_matrix(self, sd):
        Ne0_diff = pg.Nedelec0.assemble_diff_matrix(self, sd)
        proj_to_ne0 = self.proj_to_Ne0(sd)

        return Ne0_diff * proj_to_ne0

    def interpolate(self, sd: pg.Grid, func):
        tangents = sd.nodes * sd.ridge_peaks

        vals = np.zeros(self.ndof(sd))
        for r in np.arange(sd.num_ridges):
            loc = slice(sd.ridge_peaks.indptr[r], sd.ridge_peaks.indptr[r + 1])
            peaks = sd.ridge_peaks.indices[loc]
            t = tangents[:, r]
            vals[r] = np.inner(func(sd.nodes[:, peaks[0]]).flatten(), t)
            vals[r + sd.num_ridges] = np.inner(
                func(sd.nodes[:, peaks[1]]).flatten(), -t
            )

        return vals

    def eval_at_cell_centers(self, sd):
        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 12 * 3 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)
        ridge_peaks = sd.ridge_peaks

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]
            dof_loc = np.reshape(
                ridge_peaks[:, ridges_loc].indices, (2, -1), order="F"
            ).flatten()

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            coords = sd.nodes[:, nodes_uniq]

            # Compute the gradients of the Lagrange basis functions
            dphi = pg.Lagrange1.local_grads(coords, sd.dim)

            # Compute the local Nedelec basis functions and global indices
            Ne_basis = np.roll(dphi[:, indices], 6, axis=1)
            Ne_indices = np.concatenate((ridges_loc, ridges_loc + sd.num_ridges))

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + Ne_basis.size)
            rows_I[loc_idx] = np.repeat(np.arange(3), Ne_indices.size) + 3 * c
            cols_J[loc_idx] = np.concatenate(3 * [[Ne_indices]]).ravel()
            data_IJ[loc_idx] = Ne_basis.ravel() / 4.0
            idx += Ne_basis.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        raise NotImplementedError

    def get_range_discr_class(self, dim: int):
        return pg.RT0
