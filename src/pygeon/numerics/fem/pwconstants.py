import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class PwConstants:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix dictionary
        # Discretization of mass matrix
        self.mass_matrix_key = "mass"

    def ndof(self, g: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(g, pg.Grid):
            return g.num_cells
        else:
            raise ValueError

    def discretize(self, g: pg.Grid, data: dict):

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary[self.mass_matrix_key] = self.assemble_mass_matrix(g, data)

    def assemble_mass_matrix(self, g: pg.Grid, data: dict):
        """Compute the mass matrix for piecewise constants

        Parameters
        ----------
        g: grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. See self.matrix_rhs for required contents.

        Returns
        ------
        matrix: sparse csr (g.num_cells, g.num_cells)
            Matrix obtained from the discretization.
        """

        return sps.diags(g.cell_volumes)
