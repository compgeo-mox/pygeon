"""Module for the discretizations of the matrix L2 space."""

import numpy as np
import scipy.sparse as sps

import pygeon as pg


class MatPwConstants(pg.VecPwConstants):
    """
    A class representing the discretization using matrix piecewise constant functions.
    """

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
        self.base_discr: pg.VecPwConstants = pg.VecPwConstants(keyword)  # type: ignore[assignment]


class MatPwLinears(pg.VecPwLinears):
    """
    A class representing the discretization using matrix piecewise linear functions.
    """

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
        self.base_discr: pg.VecPwLinears = pg.VecPwLinears(keyword)  # type: ignore[assignment]

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


class MatPwQuadratics(pg.VecPwQuadratics):
    """
    A class representing the discretization using matrix piecewise quadratic functions.
    """

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
        self.base_discr: pg.VecPwQuadratics = pg.VecPwQuadratics(keyword)  # type: ignore[assignment]

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
            range_disc = pg.PwQuadratics()
            rearrange = np.tile(np.arange(range_disc.ndof_per_cell(sd)), 2)

        elif sd.dim == 3:
            mask = np.hstack((np.arange(10, 40), np.arange(50, 80)))
            loc_data = np.repeat([-1, 1, 1, -1, -1, 1], num_int_points)
            range_disc = pg.VecPwQuadratics()

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
