import numpy as np
import scipy.sparse as sps

import pygeon as pg


class PwConstants(pg.Discretization):
    """
    Discretization class for the piecewise constants.
    NB! Each degree of freedom is the integral over the cell.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom associated to the method.
        In this case number of cells.

        Args
            sd: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.

        """
        return sd.num_cells

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Computes the mass matrix for piecewise constants

        Args
            sd: grid, or a subclass, with geometry fields computed.
            data: dictionary with possible scaling

        Returns
            matrix: sparse csr (sd.num_cells, sd.num_cells)
        """

        return sps.diags(1 / sd.cell_volumes).tocsc()

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict):
        """
        Computes the lumped mass matrix, which coincides with the mass matrix for P0.
        """

        return self.assemble_mass_matrix(sd, data)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            diff_matrix: the differential matrix.
        """

        return sps.csr_matrix((0, self.ndof(sd)))

    def assemble_stiff_matrix(self, sd: pg.Grid, data):
        """
        Returns a zero matrix.

        Args
            sd: grid, or a subclass.

        Returns
            diff_matrix: the differential matrix.
        """

        return sps.csr_matrix((self.ndof(sd), self.ndof(sd)))

    def interpolate(self, sd: pg.Grid, func):
        """
        Interpolates a function onto the finite element space

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            array: the values of the degrees of freedom
        """

        return np.array(
            [func(x) * vol for (x, vol) in zip(sd.cell_centers, sd.cell_volumes)]
        )

    def eval_at_cell_centers(self, sd: pg.Grid, data=None):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """

        return sps.diags(1 / sd.cell_volumes).tocsc()

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Returns a zero vector
        """

        return np.zeros(self.ndof(sd))

    def get_range_discr_class(self, dim: int):
        """
        Raises an error since the range of the differential is zero.
        """

        raise NotImplementedError("There's no zero discretization in PyGeoN (yet)")
