""" Module for the discretizations of the H(curl) space. """

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class Nedelec0(pg.Discretization):
    """
    Discretization class for the Nedelec of the first kind of lowest order.
    Each degree of freedom is the integral over a mesh edge in 3D.

    Attributes:
        None

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Computes the mass matrix for a lowest-order Nedelec discretization.

        local_inner_product(dim: int) -> sps.csc_matrix:
            Compute the local inner product matrix for the given dimension.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the differential matrix for the given grid.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the function at the cell centers of the given grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition matrix for the given grid and function.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a given function onto the grid using the hcurl discretization.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case, it returns the number of ridges in the given grid.

        Args:
            sd (pg.Grid): The grid for which the number of degrees of
                freedom is calculated.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_ridges

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Computes the mass matrix for a lowest-order Nedelec discretization

        Args:
            sd (pg.Grid): Grid, or a subclass, with geometry fields computed.
            data (dict, optional): Dictionary to store the data. See self.matrix_rhs
                for required contents.

        Returns:
            sps.csc_matrix: Matrix obtained from the discretization.
        """
        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = 6 * 6 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        M = pg.BDM1.local_inner_product(sd.dim)
        M = pg.BDM1.local_inner_product(sd.dim)

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]

            Psi = self.eval_basis_at_node(sd, ridges_loc)

            # Compute the inner products
            A = Psi @ M @ Psi.T * sd.cell_volumes[c]

            # Put in the right spot
            cols = np.tile(ridges_loc, (ridges_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def eval_basis_at_node(self, sd: pg.Grid, ridges_loc: np.ndarray) -> np.ndarray:
        """
        Compute the local basis function for the Nedelec0 finite element space.

        Args:
            sd (pg.Grid): The grid object.
            ridges_loc (np.ndarray): The local ridges.

        Returns:
            np.ndarray: The local mass matrix.
        """

        # Extract the indices of the local peaks
        rp = sd.ridge_peaks
        peaks_loc = np.empty((6, 2), int)
        for ind, ridge in enumerate(ridges_loc):
            peaks_loc[ind] = rp.indices[rp.indptr[ridge] : rp.indptr[ridge + 1]]
        peaks_loc = peaks_loc.T

        # If they follow a conventional structure, these can be hardcoded
        nodes_uniq = peaks_loc.ravel()[[2, 1, 0, 6]]
        indices = np.array([2, 1, 0, 1, 0, 0, 3, 3, 3, 2, 2, 1])

        # Recompute using unique if the convention is not followed
        if not np.all(nodes_uniq[indices] == peaks_loc.ravel()):
            nodes_uniq, indices = np.unique(peaks_loc, return_inverse=True)

        indices = np.reshape(indices, (2, -1))
        coords = sd.nodes[:, nodes_uniq]

        # Compute the gradients of the Lagrange basis functions
        dphi = pg.Lagrange1.local_grads(coords, sd.dim)

        # Compute a 6 x 12 matrix Psi such that Psi[i, j] = psi_i(x_j)
        Psi = np.zeros((6, 12))
        for ridge, peaks in enumerate(indices.T):
            Psi[ridge, 3 * peaks[0] : 3 * (peaks[0] + 1)] = dphi[:, peaks[1]]
            Psi[ridge, 3 * peaks[1] : 3 * (peaks[1] + 1)] = -dphi[:, peaks[0]]

        return Psi

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix for the given grid.

        Parameters:
            sd (pg.Grid): The grid for which the differential matrix is assembled.

        Returns:
            sps.csc_matrix: The assembled differential matrix.
        """
        return sd.face_ridges.T.tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the function at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid object representing the discretization.

        Returns:
            sps.csc_matrix: The evaluated function values at the cell centers.
        """
        # Allocation
        size = 6 * 3 * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_ridges = sd.face_ridges.astype(bool) * sd.cell_faces.astype(bool)

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(cell_ridges.indptr[c], cell_ridges.indptr[c + 1])
            ridges_loc = cell_ridges.indices[loc]

            # Create a matrix with the evaluation of the basis functions at the nodes
            Psi = self.eval_basis_at_node(sd, ridges_loc)

            # The center-values are the mean of the four nodes
            Psi_split = np.split(Psi, 4, axis=1)
            Psi_at_centre = np.sum(Psi_split, axis=0).T / 4.0

            # Put in the right spot
            loc_idx = slice(idx, idx + Psi_at_centre.size)
            rows_I[loc_idx] = np.repeat(np.arange(3), ridges_loc.size) + 3 * c
            cols_J[loc_idx] = np.concatenate(3 * [[ridges_loc]]).ravel()
            data_IJ[loc_idx] = Psi_at_centre.ravel()
            idx += Psi_at_centre.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition matrix for the given grid and function.

        Args:
            sd (pg.Grid): The grid on which to assemble the matrix.
            func (Callable): The function defining the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition matrix.
        """
        raise NotImplementedError

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class for the given grid.
        """
        return pg.RT0

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function onto the grid using the hcurl discretization.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values on the grid.
        """
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

    Attributes:
        keyword (str): The keyword associated with the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Return the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix for the given grid and data.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped matrix for the given grid and data.

        proj_to_Ne0(sd: pg.Grid) -> sps.csc_matrix:
            Project the solution to the Nedelec of the first kind.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the differential matrix for the H(curl) finite element space.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates the given function `func` over the specified grid `sd`.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the basis functions at the cell centers and construct the global matrices.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition for the given grid, function, and boundary
            faces.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case, it returns twice the number of ridges in the given grid.

        Args:
            sd (pg.Grid): The grid or a subclass.

        Returns:
            int: The number of degrees of freedom.
        """
        return 2 * sd.num_ridges

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the mass matrix for the given grid and data.

        Parameters:
            sd (pg.Grid): The grid for which the mass matrix is to be assembled.
            data (dict, optional): Additional data required for the assembly process.

        Returns:
            sps.csc_matrix: The assembled mass matrix.
        """
        raise NotImplementedError

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped matrix for the given grid and data.

        Args:
            sd (pg.Grid): The grid object.
            data (dict, optional): Additional data. Defaults to None.

        Returns:
            sps.csc_matrix: The assembled lumped matrix.
        """
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

    def proj_to_Ne0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the solution to the Nedelec of the first kind.

        Args:
            sd (pg.Grid): The grid object representing the discretization.

        Returns:
            sps.csc_matrix: The projection matrix to the Nedelec of the first kind.
        """
        return sps.hstack([sps.eye(sd.num_ridges), -sps.eye(sd.num_ridges)]) / 2

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix for the H(curl) finite element space.

        Args:
            sd (pg.Grid): The grid on which the finite element space is defined.

        Returns:
            sps.csc_matrix: The assembled differential matrix.
        """
        n0 = pg.Nedelec0(self.keyword)
        Ne0_diff = n0.assemble_diff_matrix(sd)

        proj_to_ne0 = self.proj_to_Ne0(sd)
        return Ne0_diff @ proj_to_ne0

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates the given function `func` over the specified grid `sd`.

        Args:
            sd (pg.Grid): The grid over which to interpolate the function.
            func (Callable): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values.

        """
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

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the basis functions at the cell centers and construct the global matrices.

        Args:
            sd (pg.Grid): The grid object representing the mesh.

        Returns:
            sps.csc_matrix: The global matrices constructed from the basis functions
                evaluated at the cell centers.
        """
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

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition for the given grid, function, and
            boundary faces.

        Args:
            sd (pg.Grid): The grid on which to assemble the natural boundary condition.
            func (Callable): The function defining the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition.
        """
        raise NotImplementedError

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.RT0
