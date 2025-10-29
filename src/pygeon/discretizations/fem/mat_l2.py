"""Module for the discretizations of the matrix L2 space."""

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class MatPwConstants(pg.VecPwConstants):
    """
    A class representing the discretization using matrix piecewise constant functions.
    """

    poly_order = 0
    tensor_order = pg.MATRIX

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwConstants(keyword)  # type: ignore[assignment]


class MatPwLinears(pg.VecPwLinears):
    """
    A class representing the discretization using matrix piecewise linear functions.
    """

    poly_order = 1
    tensor_order = pg.MATRIX

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwLinears(keyword)  # type: ignore[assignment]

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the matrix-valued piecewise linears.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        num_int_points = self.ndof_per_cell(sd) // (sd.dim**2)

        size = num_int_points * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.ones(size)
        idx = 0

        if sd.dim == 2:
            mask = [0, 1, 2, 9, 10, 11]
        elif sd.dim == 3:
            mask = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35]

        range_disc = pg.PwLinears()

        for c in np.arange(sd.num_cells):
            loc_dofs = self.local_dofs_of_cell(sd, c)[mask]

            ran_dofs = range_disc.local_dofs_of_cell(sd, c)
            ran_dofs = np.tile(ran_dofs, sd.dim)

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + loc_dofs.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = loc_dofs
            idx += loc_dofs.size

        # Construct the global matrices
        shape = (range_disc.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ, (rows_I, cols_J)), shape=shape)

    def assemble_asym_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the asymmetry matrix for the matrix-valued
        piecewise linears.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The asymmetry matrix obtained from the discretization.
        """
        num_int_points = self.ndof_per_cell(sd) // (sd.dim**2)
        size = num_int_points * (sd.dim * (sd.dim - 1)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        if sd.dim == 2:
            mask = np.arange(3, 9)
            loc_data = np.repeat([-1, 1], 3)
            range_disc = pg.PwLinears()  # type: ignore[assignment]
            rearrange = np.tile(np.arange(range_disc.ndof_per_cell(sd)), 2)

        elif sd.dim == 3:
            mask = np.hstack((np.arange(4, 16), np.arange(20, 32)))
            loc_data = np.repeat([-1, 1, 1, -1, -1, 1], 4)
            range_disc = pg.VecPwLinears()  # type: ignore[assignment]

            rearrange = np.arange(12).reshape((3, 4))
            rearrange = rearrange[[2, 1, 2, 0, 1, 0]].ravel()

        for c in np.arange(sd.num_cells):
            loc_dofs = self.local_dofs_of_cell(sd, c)[mask]
            ran_dofs = range_disc.local_dofs_of_cell(sd, c)[rearrange]

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + loc_dofs.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = loc_dofs
            data_IJ[loc_idx] = loc_data
            idx += loc_dofs.size

        # Construct the global matrices
        shape = (range_disc.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ, (rows_I, cols_J)), shape=shape)

    def assemble_corotational_correction(
        self, sd: pg.Grid, rotation: np.ndarray
    ) -> sps.csc_array:
        """
        Assembles and returns the corotational correction matrix for the matrix-valued
        piecewise linears. We assume rotation to be a piecewise constant function in P0.

        Args:
            sd (pg.Grid): The grid.
            rotation (np.ndarray): The rotation in P0, either a scalar field in 2D or
                a vector field in 3D.

        Returns:
            sps.csc_array: The corotational correction matrix obtained from the
                discretization.
        """
        # Retrieve the discretization for the rotation, it depends on the dimension
        if sd.dim == 2:
            disc_rot = pg.PwConstants(self.keyword)
        else:
            disc_rot = pg.VecPwConstants(self.keyword)

        # The idea is to project the rotation from P0 to P1 to be able to
        # perform the assembly with the P1 asymmetry matrix
        proj_p1 = disc_rot.proj_to_higher_PwPolynomials(sd)

        # Assemble the asymmetry matrix in P1 space
        asym = self.assemble_asym_matrix(sd)

        # Convert the rotation to the Omega tensor, which is in matrix P0 space
        omega = -asym.T @ proj_p1 @ rotation

        # Assemble the multiplication matrices A*Omega and Omega*A used in the
        # corotational correction
        A_omega = self.assemble_mult_matrix(sd, omega, right_mult=True)
        omega_A = self.assemble_mult_matrix(sd, omega, right_mult=False)

        # Return the corotational correction matrix
        return A_omega - omega_A

    def assemble_mult_matrix(
        self, sd: pg.Grid, mult_mat: np.ndarray, right_mult: bool
    ) -> sps.csc_array:
        """
        Assembles and returns the multiplication matrix for the matrix-valued
        piecewise constants.

        Args:
            sd (pg.Grid): The grid.
            mult_mat (np.ndarray): The matrix to multiply with. It is assumed to be
                a piecewise constant matrix.
            right_mult (bool): If True, performs right multiplication. If False, left
                multiplication.

        Returns:
            sps.csc_array: The multiplication matrix obtained from the discretization.
        """
        size = sd.dim**4 * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Precompute the basis functions
        basis = np.eye(sd.dim**2).reshape((sd.dim, sd.dim, -1))

        if right_mult:
            # right multiplication: A @ mult_mat
            oper_str = "ijk,jp->ipk"
        else:
            # left multiplication: mult_mat @ A
            oper_str = "ijk,pi->pjk"

        for c in np.arange(sd.num_cells):
            # Get the local degrees of freedom for the cell ordered column wise
            loc_dofs = self.local_dofs_of_cell(sd, c).reshape((-1, sd.dim + 1))

            # Iterate over the node dofs
            for node_dofs in loc_dofs.T:
                mult_mat_loc = mult_mat[node_dofs].reshape((sd.dim, sd.dim))

                # Compute the product A @ mult_mat (if right_mult is True)
                # or mult_mat @ A (if right_mult is False)
                prod = np.einsum(oper_str, basis, mult_mat_loc)

                # Reshape the product to match the degrees of freedom
                prod_at_dofs = np.array(
                    [prod[:, :, i].ravel() for i in np.arange(sd.dim**2)]
                )

                # Save only the non-zeros values of the local matrix in the global
                # structure
                rows_loc, cols_loc = prod_at_dofs.nonzero()

                # Save values of the local matrix in the global structure
                loc_idx = slice(idx, idx + rows_loc.size)
                rows_I[loc_idx] = node_dofs[cols_loc]
                cols_J[loc_idx] = node_dofs[rows_loc]
                data_IJ[loc_idx] = prod_at_dofs[rows_loc, cols_loc]
                idx += rows_loc.size

        # Construct the global matrices
        shape = (self.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ[:idx], (rows_I[:idx], cols_J[:idx])), shape=shape)


class MatPwQuadratics(pg.VecPwQuadratics):
    """
    A class representing the discretization using matrix piecewise quadratic functions.
    """

    poly_order = 2
    tensor_order = pg.MATRIX

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the matrix discretization class.
                Default is pg.UNITARY_DATA.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwQuadratics(keyword)  # type: ignore[assignment]

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the matrix-valued
        piecewise quadratics.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        range_disc = pg.PwQuadratics()

        size = range_disc.ndof_per_cell(sd) * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.ones(size)
        idx = 0

        if sd.dim == 2:
            mask = np.hstack((np.arange(6), np.arange(18, 24)))
        elif sd.dim == 3:
            mask = np.hstack((np.arange(10), np.arange(40, 50), np.arange(80, 90)))

        for c in np.arange(sd.num_cells):
            loc_dofs = self.local_dofs_of_cell(sd, c)[mask]

            ran_dofs = range_disc.local_dofs_of_cell(sd, c)
            ran_dofs = np.tile(ran_dofs, sd.dim)

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + loc_dofs.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = loc_dofs
            idx += loc_dofs.size

        # Construct the global matrices
        shape = (range_disc.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ, (rows_I, cols_J)), shape=shape)

    def assemble_asym_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the asymmetry matrix for the matrix-valued
        piecewise quadratics.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The asymmetry matrix obtained from the discretization.
        """
        num_int_points = self.ndof_per_cell(sd) // (sd.dim**2)
        size = num_int_points * (sd.dim * (sd.dim - 1)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        if sd.dim == 2:
            mask = np.arange(6, 18)
            loc_data = np.repeat([-1, 1], num_int_points)
            range_disc = pg.PwQuadratics()  # type: ignore[assignment]
            rearrange = np.tile(np.arange(range_disc.ndof_per_cell(sd)), 2)

        elif sd.dim == 3:
            mask = np.hstack((np.arange(10, 40), np.arange(50, 80)))
            loc_data = np.repeat([-1, 1, 1, -1, -1, 1], num_int_points)
            range_disc = pg.VecPwQuadratics()  # type: ignore[assignment]

            rearrange = np.arange(3 * num_int_points).reshape((3, -1))
            rearrange = rearrange[[2, 1, 2, 0, 1, 0]].ravel()

        for c in np.arange(sd.num_cells):
            loc_dofs = self.local_dofs_of_cell(sd, c)[mask]
            ran_dofs = range_disc.local_dofs_of_cell(sd, c)[rearrange]

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + loc_dofs.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = loc_dofs
            data_IJ[loc_idx] = loc_data
            idx += loc_dofs.size

        # Construct the global matrices
        shape = (range_disc.ndof(sd), self.ndof(sd))
        return sps.csc_array((data_IJ, (rows_I, cols_J)), shape=shape)
