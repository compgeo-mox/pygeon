"""
Discretization of the flux term of an equation.
"""

import scipy.sparse as sps

import porepy as pp
import pygeon as pg

module_sections = ["numerics", "disrcetization"]

class P1Source(pp.numerics.discretization.Discretization):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = param.get_source.keyword.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    @pp.time_logger(sections=module_sections)
    def ndof(self, g) -> int:
        return g.num_nodes

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(self, g, data):
        """Return the (null) matrix and right-hand side for a discretization of the
        integrated source term. Also discretize the necessary operators if the data
        dictionary does not contain a source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs.
            sources (array, self.ndof): Right-hand side vector.

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """

        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    @pp.time_logger(sections=module_sections)
    def assemble_matrix(self, g, data):
        """Return the (null) matrix and for a discretization of the integrated source
        term. Also discretize the necessary operators if the data dictionary does not
        contain a source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): Null system matrix of this
                discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["source"]

    @pp.time_logger(sections=module_sections)
    def assemble_rhs(self, g, data):
        """Return the rhs for a discretization of the integrated source term. Also
        discretize the necessary operators if the data dictionary does not contain a
        source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.array (self.ndof): Right hand side vector representing the
                source.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        sources = parameter_dictionary["source"]
        assert sources.size == self.ndof(
            g
        ), "There should be one source value for each cell"
        return matrix_dictionary["bound_source"]

    @pp.time_logger(sections=module_sections)
    def discretize(self, g, data):
        """Discretize an integrated source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs, stored as
                self._key() + "source".
            sources (array, self.ndof): Right-hand side vector, stored as
                self._key() + "bound_source".

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.
        """
        source = data[pp.PARAMETERS][self.keyword]["source"]

        solver = pg.P1MassMatrix(keyword=self.keyword)
        M = solver.matrix(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary["source"] = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        matrix_dictionary["bound_source"] = M.dot(source)
