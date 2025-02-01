""" Module for the discretizations of the H1 space. """

from typing import Optional

import numpy as np
import porepy as pp
import scipy.linalg as spl
import scipy.sparse as sps

import pygeon as pg


class VecVLagrange1(pg.VecDiscretization):
    """
    Vector Lagrange virtual element discretization for H1 space in 2d.

    This class represents a virtual element discretization for the H1 space using
    vector virtual Lagrange elements. It provides methods for assembling various matrices
    and operators, such as the mass matrix, divergence matrix, symmetric gradient
    matrix, and more.

    Convention for the ordering is first all the x then all the y.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4.

    We are considering the following structure of the stress tensor in 2d

    sigma = [[sigma_xx, sigma_xy],
             [sigma_yx, sigma_yy]]

    which is represented in the code unrolled row-wise as a vector of length 4

    sigma = [sigma_xx, sigma_xy,
             sigma_yx, sigma_yy]

    The strain tensor follows the same approach.

    Args:
        keyword (str): The keyword for the H1 class.

    Attributes:
        scalar_discr (pg.VLagrange1): A local virtual Lagrange1 class for performing some of
            the computations.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated with the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles and returns the mass matrix for the lowest order Lagrange element.

        assemble_div_matrix(sd: pg.Grid) -> sps.csc_array:
            Returns the divergence matrix operator for the lowest order vector Lagrange
            element.

        local_div(c_volume: float, coord: np.ndarray, dim: int) -> np.ndarray:
            Computes the local divergence matrix for P1.

        assemble_div_div_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Returns the div-div matrix operator for the lowest order vector Lagrange element.

        assemble_symgrad_matrix(sd: pg.Grid) -> sps.csc_array:
            Returns the symmetric gradient matrix operator for the lowest order vector Lagrange
            element.

        local_symgrad(c_volume: float, coord: np.ndarray, dim: int, sym: np.ndarray)
            -> np.ndarray:
            Computes the local symmetric gradient matrix for P1.

        assemble_symgrad_symgrad_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Returns the symgrad-symgrad matrix operator for the lowest order vector Lagrange
            element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.
    """

    def __init__(self, keyword: str = pg.UNITARY_DATA) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.Lagrange1.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        super().__init__(keyword, pg.VLagrange1)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The div matrix obtained from the discretization.
        """
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = cell_nodes.sum() * sd.dim
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for cell, diam in enumerate(cell_diams):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])

            nodes_loc = cell_nodes.indices[loc]

            # Compute the div local matrix
            A = self.local_div(sd, cell, diam, nodes_loc)

            # Save values for the local matrix in the global structure
            cols = nodes_loc + shift
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cell * np.ones(cols.size)
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_div(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local div matrix for vector P1.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the div matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.

        Returns:
            ndarray: Local mass Hdiv matrix.
        """
        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        return sd.cell_volumes[cell] * proj[1:] / diam

    def assemble_div_div_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Returns the div-div matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by the Lame' parameter lambda.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Additional data, the Lame' parameter lambda.
                Defaults to None.

        Returns:
            sps.csc_array: sparse (sd.num_nodes, sd.num_nodes)
                Div-div matrix obtained from the discretization.
        """
        if data is None:
            labda = 1
        else:
            parameter_dictionary = data[pp.PARAMETERS][self.keyword]
            labda = parameter_dictionary.get("lambda", 1)

        p0 = pg.PwConstants(self.keyword)

        div = self.assemble_div_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)

        return div.T @ (labda * mass) @ div

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the symmetric gradient matrix operator for the
        lowest order vector Lagrange element

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_array: The sparse symmetric gradient matrix operator.

        Raises:
            None

        Notes:
            - If a 0-dimensional grid is given, a zero matrix is returned.
            - The method maps the domain to a reference geometry.
            - The method allocates data to store matrix entries efficiently.
            - The symmetrization matrix is constructed differently for 2D and 3D cases.
            - The method computes the symgrad local matrix for each cell and saves
              the values in the global structure.
            - Finally, the method constructs the global matrices using the saved values.

        """
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = cell_nodes.sum() * np.power(sd.dim, 3)
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        dim2 = np.square(sd.dim)
        # construct the symmetrization matrix
        sym = np.eye(dim2)
        if sd.dim == 2:
            sym[np.ix_([1, 2], [1, 2])] = 0.5
        else:
            raise ValueError("Grid dimension should be 2.")

        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for cell, diam in enumerate(cell_diams):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the symgrad local matrix
            A = self.local_symgrad(sd, cell, diam, nodes_loc, sym)

            # Save values for the local matrix in the global structure
            cols = (nodes_loc + shift).ravel()
            cols = cols * np.ones((dim2, 1), dtype=int)

            rows = cell + np.arange(dim2) * sd.num_cells
            rows = np.ones(nodes_loc.size * sd.dim, dtype=int) * rows.reshape((-1, 1))

            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = rows.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_symgrad(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray, sym: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local symgrad matrix for vector virtual Lagrangian.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the div matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of node indices.
            sym (np.ndarray): Symmetric matrix.

        Returns:
            np.ndarray: Local symmetric gradient matrix.
        """

        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)
        grad = spl.block_diag(*([proj[1:]] * sd.dim))

        return sd.cell_volumes[cell] * sym @ grad / diam

    def assemble_symgrad_symgrad_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Returns the symgrad-symgrad matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by twice the Lame' parameter mu.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Additional data, the Lame' parameter mu. Defaults to None.

        Returns:
            sps.csc_array: Sparse symgrad-symgrad matrix of shape
                (sd.num_nodes, sd.num_nodes).
                The matrix obtained from the discretization.

        NOTE: Duplicate of pg.VecLagrange1.assemble_symgrad_symgrad_matrix
        """
        if data is None:
            mu = 1
        else:
            parameter_dictionary = data[pp.PARAMETERS][self.keyword]
            mu = parameter_dictionary.get("mu", 1)

        coeff = 2 * mu
        p0 = pg.PwConstants(self.keyword)

        symgrad = self.assemble_symgrad_matrix(sd)
        mass = p0.assemble_mass_matrix(sd)
        tensor_mass = sps.block_diag([coeff * mass] * np.square(sd.dim), format="csc")

        return symgrad.T @ tensor_mass @ symgrad

    def assemble_penalisation_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles and returns the penalisation matrix.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_array: The penalisation matrix obtained from the discretization.
        """
        # Precomputations
        cell_nodes = sd.cell_nodes()
        cell_diams = sd.cell_diameters(cell_nodes)

        # Data allocation
        size = np.sum(np.square(cell_nodes.sum(0)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        for cell, diam in enumerate(cell_diams):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            nodes_loc = cell_nodes.indices[loc]

            A = self.assemble_loc_penalisation_matrix(sd, cell, diam, nodes_loc)

            # Save values for local mass matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        scalar_pen = sps.csc_array((data_V, (rows_I, cols_J)))
        return sps.block_diag([scalar_pen] * sd.dim, format="csc")

    def assemble_loc_penalisation_matrix(
        self, sd: pg.Grid, cell: int, diam: float, nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the local penalisation VEM matrix on a given cell
        according to the Hitchhiker's (6.5)

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            cell (int): The index of the cell on which to compute the mass matrix.
            diam (float): The diameter of the cell.
            nodes (np.ndarray): The array of nodes associated with the cell.

        Returns:
            np.ndarray: The computed local VEM mass matrix.
        """
        proj = self.scalar_discr.assemble_loc_proj_to_mon(sd, cell, diam, nodes)

        D = self.scalar_discr.assemble_loc_dofs_of_monomials(sd, cell, diam, nodes)
        I_minus_Pi = np.eye(nodes.size) - D @ proj

        return I_minus_Pi.T @ I_minus_Pi

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.

        NOTE: Duplicate of pg.VecLagrange1.assemble_diff_matrix
        """
        div = self.assemble_div_matrix(sd)
        symgrad = self.assemble_symgrad_matrix(sd)

        return sps.block_array([[symgrad], [div]], format="csc")

    def assemble_stiff_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the global stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid on which the finite element method is defined.
            data (Optional[dict]): Additional data required for the assembly process.

        Returns:
            sps.csc_array: The assembled global stiffness matrix.
        """
        # compute the two parts of the global stiffness matrix
        sym_sym = self.assemble_symgrad_symgrad_matrix(sd, data)
        div_div = self.assemble_div_div_matrix(sd, data)

        # penalisation
        dofi_dofi = self.assemble_penalisation_matrix(sd)

        # return the global stiffness matrix
        return sym_sym + div_div + dofi_dofi

    def get_range_discr_class(self, dim: int) -> object:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of
                the differential.

        Raises:
            NotImplementedError: There is no range discretization for the vector
                Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector VLagrangian 1 in PyGeoN"
        )

    def compute_stress(
        self,
        sd: pg.Grid,
        u: np.ndarray,
        data: dict,
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field.

        Args:
            sd (pg.Grid): The spatial discretization object.
            u (ndarray): The displacement field.
            data (dict): Data for the computation including the Lame parameters accessed with
                the keys "lambda" and "mu". Both float and np.ndarray are accepted.

        Returns:
            ndarray: The stress tensor.

        NOTE: Duplicate of pg.VecLagrange1.compute_stress
        """
        # construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        p0 = pg.PwConstants(self.keyword)
        proj = p0.eval_at_cell_centers(sd)

        # retrieve Lamé parameters
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        mu = parameter_dictionary["mu"]
        labda = parameter_dictionary["lambda"]

        # compute the two terms and split on each component
        sigma = np.array(np.split(2 * mu * symgrad @ u, np.square(sd.dim)))
        sigma[:: (sd.dim + 1)] += labda * div @ u

        # compute the actual dofs
        sigma = sigma @ proj

        # create the indices to re-arrange the components for the second
        # order tensor
        idx = np.arange(np.square(sd.dim)).reshape((sd.dim, -1), order="F")

        return sigma[idx].T
