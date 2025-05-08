"""Module for the discretizations of the matrix L2 space."""

import numpy as np
import scipy.sparse as sps
import pygeon as pg


class MatPwConstants(pg.VecPwConstants):
    """
    A class representing the discretization using matrix piecewise constant functions.

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    Methods:
        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray],
            np.ndarray], relative: Optional[bool] = True, etype:
            Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a
            function.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwConstants.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwConstants(keyword)


class MatPwLinears(pg.VecPwLinears):
    """
    A class representing the discretization using matrix piecewise linear functions.

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwLinears.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwLinears(keyword)

    def assemble_trace_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles and returns the trace matrix for the matrix-valued piecewise linears.

        Args:
            sd (pg.Grid): The grid.

        Returns:
            sps.csc_array: The trace matrix obtained from the discretization.
        """
        size = (sd.dim + 1) * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.ones(size)
        idx = 0

        if sd.dim == 2:
            mask = [0, 1, 2, 9, 10, 11]
        elif sd.dim == 3:
            mask = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35]

        linears = pg.PwLinears()

        for c in np.arange(sd.num_cells):
            loc_dofs = self.local_dofs_of_cell(sd, c)[mask]

            ran_dofs = linears.local_dofs_of_cell(sd, c)
            ran_dofs = np.tile(ran_dofs, sd.dim)

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + loc_dofs.size)
            rows_I[loc_idx] = ran_dofs
            cols_J[loc_idx] = loc_dofs
            idx += loc_dofs.size

        # Construct the global matrices
        shape = (linears.ndof(sd), self.ndof(sd))
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
        size = (sd.dim + 1) * (sd.dim * (sd.dim - 1)) * sd.num_cells  # TODO
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        if sd.dim == 2:
            mask = np.arange(3, 9)
            loc_data = np.repeat([-1, 1], 3)
            range_disc = pg.PwLinears()
            rearrange = [0, 1, 2, 0, 1, 2]

        elif sd.dim == 3:
            mask = np.hstack((np.arange(4, 16), np.arange(20, 32)))
            loc_data = np.repeat([-1, 1, 1, -1, -1, 1], 4)
            range_disc = pg.VecPwLinears()

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

    Attributes:
        keyword (str): The keyword for the matrix discretization class.
        base_discr (pg.Discretization): The base discretization class.

    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the matrix discretization class.
        The base discretization class is pg.PwQuadratics.

        Args:
            keyword (str): The keyword for the matrix discretization class.

        Returns:
            None
        """
        super().__init__(keyword)
        self.base_discr = pg.VecPwQuadratics(keyword)
