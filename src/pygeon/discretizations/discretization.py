from typing import Callable
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Discretization:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        self.mass_matrix_key = "mass"
        self.diff_matrix_key = "diff"
        self.lumped_matrix_key = "lumped"

    def ndof(self, g: pg.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.

        Args
            g: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.
        """

        raise NotImplementedError

    def assemble_mass_matrix(self, g: pg.Grid, data: dict = None):
        """
        Assemble the mass matrix

        Args
            g: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        raise NotImplementedError

    def assemble_diff_matrix(self, g: pg.Grid):
        """
        Assemble the matrix corresponding to the differential

        Args
            g: grid, or a subclass.

        Returns
            mass_matrix: the differential matrix.
        """

        raise NotImplementedError

    def interpolate(self, g: pg.Grid, func: Callable):
        raise NotImplementedError

    def eval_at_cell_centers(self, g: pg.Grid, func: Callable):
        raise NotImplementedError

    def source_term(self, g: pg.Grid, func: Callable):
        return self.assemble_mass_matrix(g) * self.interpolate(g, func)

    def assemble_nat_bc(self, g: pg.Grid, b_dofs):
        raise NotImplementedError
