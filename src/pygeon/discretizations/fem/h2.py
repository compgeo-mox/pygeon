"""Module for the discretizations of the H2 space."""

from typing import Callable, Optional, Type

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class Hermite1(pg.Discretization):
    """
    Class representing the Hermite1 finite element discretization.

    For the 1d mass and stiffness, consider the reference
    https://doi.org/10.1016/j.compstruc.2022.106938
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of nodes.

        Args
            sd: grid, or a subclass.

        Returns
            ndof: the number of degrees of freedom.
        """
        return sd.num_nodes * (1 + sd.dim) + sd.num_cells * np.square(sd.dim - 1)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Returns the mass matrix for the lowest order Hermite element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_array: The mass matrix obtained from the discretization.
        """

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()
        local_mass = self.assemble_local_mass(sd.dim)

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H2 local matrix
            A = local_mass * sd.cell_volumes[c]

            # Save values for mass-H2 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_local_mass(self, dim: int, c_volume: float) -> np.ndarray:
        """Compute the local mass matrix on an element with measure 1.

        Args:
            dim (int): Dimension of the matrix.
            c_volume (float): Scalar cell volume.

        Returns:
            np.ndarray: Local mass matrix of shape (num_nodes_of_cell,
                num_nodes_of_cell).
        """
        if dim == 1:
            h = c_volume
            # Compute the local mass matrix
            M = np.array(
                [
                    [156, 22 * h, 54, -13 * h],
                    [22 * h, 4 * h**2, 13 * h, -3 * h**2],
                    [54, 13 * h, 156, -22 * h],
                    [-13 * h, -3 * h**2, -22 * h, 4 * h**2],
                ]
            )
            M *= h / 420
        else:
            raise NotImplementedError(
                f"Local mass matrix not implemented for dimension {dim}."
            )
        return M

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the
                matrix.

        Returns:
            sps.csc_array: The assembled stiffness matrix.
        """
        weight = np.ones(sd.num_cells)
        if data is not None:
            weight = (
                data.get(pp.PARAMETERS, {}).get(self.keyword, {}).get("weight", weight)
            )

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.square(np.square(sd.dim + 1) + np.square(sd.dim - 1)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]

            # Compute the stiff-H2 local matrix
            A = self.local_stiff(
                weight[c],
                sd.cell_volumes[c],
                sd.dim,
            )

            # Save values for stiff-H2 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The differential matrix.
        """
        if sd.dim == 3:
            return sd.ridge_peaks.T.tocsc()
        elif sd.dim == 2:
            return sd.face_ridges.T.tocsc()
        elif sd.dim == 1:
            return sps.csc_array(sd.cell_faces.T)
        elif sd.dim == 0:
            return sps.csc_array((0, 1))
        else:
            raise ValueError

    def local_stiff(self, weight: float, c_volume: np.ndarray, dim: int) -> np.ndarray:
        """
        Compute the local stiffness matrix for H1.

        Args:
            weight (float): coefficient for the local cell.
            c_volume (np.ndarray): scalar cell volume.
            dim (int): dimension of the problem.

        Returns:
            np.ndarray: local stiffness matrix of (dim+1, dim+1) shape.
        """
        if dim == 1:
            h = c_volume
            # Compute the local stiffness matrix
            M = 2 * np.array(
                [
                    [6, 3 * h, -6, 3 * h],
                    [3 * h, 2 * h**2, -3 * h, h**2],
                    [-6, -3 * h, 6, -3 * h],
                    [3 * h, h**2, -3 * h, 2 * h**2],
                ]
            )
            M *= weight / h**3
        else:
            raise NotImplementedError(
                f"Local mass matrix not implemented for dimension {dim}."
            )
        return M

    def proj_to_pwLinears(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a Lagrangian function to a piecewise linear
        function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        rows_I = np.arange(sd.num_cells * (sd.dim + 1))
        rows_I = rows_I.reshape((-1, sd.num_cells)).ravel(order="F")
        cols_J = sd.cell_nodes().indices
        data_IJ = np.ones_like(rows_I, dtype=float)

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def proj_to_pwConstants(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a Lagrangian function to a piecewise
        constant function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        node_cells = sd.cell_nodes().T.astype(float)
        node_cells *= sd.cell_volumes[:, np.newaxis] / (sd.dim + 1)

        # Return the global matrix
        return sps.csc_array(node_cells)

    def proj_to_Hermite2(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a linear Lagrangian function to a second
        order Hermite function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        if sd.dim == 1:
            edge_nodes = sd.cell_faces
        elif sd.dim == 2:
            edge_nodes = sd.face_ridges
        elif sd.dim == 3:
            edge_nodes = sd.ridge_peaks

        edge_nodes = abs(edge_nodes) / 2  # type: ignore[assignment]

        ndof = self.ndof(sd)
        return sps.vstack((sps.eye_array(ndof), edge_nodes.T)).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for evaluating a Lagrangian function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection at the cell centers.
        """
        if sd.dim == 0:
            return sps.csc_array((1, 0))
        eval = sps.csc_array(sd.cell_nodes())
        num_nodes = sps.diags_array(1.0 / sd.num_cell_nodes())

        return (eval @ num_nodes).T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the
                grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Hermite1

        Args:
            sd (pg.Grid): The grid object representing the computational domain
            func (Callable[[np.ndarray], np.ndarray]): The function used to evaluate
                the 'natural' boundary condition
            b_faces (np.ndarray): The array of boundary faces

        Returns:
            np.ndarray: The assembled 'natural' boundary condition values
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for face in b_faces:
            loc = slice(sd.face_nodes.indptr[face], sd.face_nodes.indptr[face + 1])
            loc_n = sd.face_nodes.indices[loc]

            vals[loc_n] += (
                func(sd.face_centers[:, face]) * sd.face_areas[face] / loc_n.size
            )

        return vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the appropriate range discretization class based on the dimension.

        Args:
            dim (int): The dimension of the problem.

        Returns:
            object: The range discretization class.

        Raises:
            NotImplementedError: If there's no zero discretization in PyGeoN.
        """
        if dim == 3:
            return pg.Nedelec0
        elif dim == 2:
            return pg.RT0
        elif dim == 1:
            return pg.PwConstants
        else:
            raise NotImplementedError("There's no zero discretization in PyGeoN")
