"""Module for the discretizations of the vector H1 space."""

from typing import Optional, Type

import numpy as np
import porepy as pp
import scipy.linalg as spl
import scipy.sparse as sps

import pygeon as pg


class VecLagrange1(pg.VecDiscretization):
    """
    Vector Lagrange finite element discretization for H1 space.

    This class represents a finite element discretization for the H1 space using
    vector Lagrange elements. It provides methods for assembling various matrices
    and operators, such as the mass matrix, divergence matrix, symmetric gradient
    matrix, and more.

    Convention for the ordering is first all the x, then all the y, and (if dim = 3)
    all the z.

    The stress tensor and strain tensor are represented as vectors unrolled row-wise.
    In 2D, the stress tensor has a length of 4, and in 3D, it has a length of 9.

    We are considering the following structure of the stress tensor in 2d

    sigma = [[sigma_xx, sigma_xy],
             [sigma_yx, sigma_yy]]

    which is represented in the code unrolled row-wise as a vector of length 4

    sigma = [sigma_xx, sigma_xy,
             sigma_yx, sigma_yy]

    While in 3d the stress tensor can be written as

    sigma = [[sigma_xx, sigma_xy, sigma_xz],
             [sigma_yx, sigma_yy, sigma_yz],
             [sigma_zx, sigma_zy, sigma_zz]]

    where its vectorized structure of lenght 9 is given by

    sigma = [sigma_xx, sigma_xy, sigma_xz,
             sigma_yx, sigma_yy, sigma_yz,
             sigma_zx, sigma_zy, sigma_zz]


    The strain tensor follows the same approach.

    Args:
        keyword (str): The keyword for the H1 class.

    Attributes:
        scalar_discr (pg.Lagrange1): A local Lagrange1 class for performing some of the
            computations.

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

        assemble_div_div_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Returns the div-div matrix operator for the lowest order vector Lagrange
            element.

        assemble_symgrad_matrix(sd: pg.Grid) -> sps.csc_array:
            Returns the symmetric gradient matrix operator for the lowest order vector
            Lagrange element.

        local_symgrad(c_volume: float, coord: np.ndarray, dim: int, sym: np.ndarray)
            -> np.ndarray:
            Computes the local symmetric gradient matrix for P1.

        assemble_symgrad_symgrad_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Returns the symgrad-symgrad matrix operator for the lowest order vector
            Lagrange element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
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
        self.scalar_discr: pg.Lagrange1
        super().__init__(keyword, pg.Lagrange1)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Returns the div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_array: The div matrix obtained from the discretization.
        """
        # If a 0-d grid is given then we return a zero matrix
        if sd.dim == 0:
            return sps.csc_array((1, 1))

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(sd)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = sd.dim * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()
        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the div local matrix
            A = self.local_div(
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for the local matrix in the global structure
            cols = nodes_loc + shift
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = c * np.ones(cols.size)
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_div(self, c_volume: float, coord: np.ndarray, dim: int) -> np.ndarray:
        """
        Compute the local div matrix for vector P1.

        Args:
            c_volume (float): Cell volume.
            coord (ndarray): Coordinates of the cell.
            dim (int): Dimension of the cell.

        Returns:
            ndarray: Local mass Hdiv matrix.
                Shape: (num_faces_of_cell, num_faces_of_cell)
        """

        dphi = self.scalar_discr.local_grads(coord, dim)

        return c_volume * dphi

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
            csc_array: sparse (sd.num_nodes, sd.num_nodes)
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
        # If a 0-d grid is given then we return a zero matrix
        if sd.dim == 0:
            return sps.csc_array((1, 1))

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, _, _, node_coords = pp.map_geometry.map_grid(sd)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(sd.dim, 3) * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        dim2 = np.square(sd.dim)
        # construct the symmetrization matrix, which is different in
        # 2d and in 3d
        sym = np.eye(dim2)
        if sd.dim == 2:
            sym[np.ix_([1, 2], [1, 2])] = 0.5
        elif sd.dim == 3:
            sym[np.ix_([1, 3], [1, 3])] = 0.5
            sym[np.ix_([2, 6], [2, 6])] = 0.5
            sym[np.ix_([5, 7], [5, 7])] = 0.5

        cell_nodes = sd.cell_nodes()
        # shift to comply with the ordering convention of (x, y, z) components
        shift = np.atleast_2d(np.arange(sd.dim)).T * sd.num_nodes
        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the symgrad local matrix
            A = self.local_symgrad(sd.cell_volumes[c], coord_loc, sd.dim, sym)

            # Save values for the local matrix in the global structure
            cols = (nodes_loc + shift).ravel()
            cols = cols * np.ones((dim2, 1), dtype=int)

            rows = c + np.arange(dim2) * sd.num_cells
            rows = np.ones(dim2 + sd.dim, dtype=int) * rows.reshape((-1, 1))

            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = rows.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_symgrad(
        self, c_volume: float, coord: np.ndarray, dim: int, sym: np.ndarray
    ) -> np.ndarray:
        """
        Compute the local symmetric gradient matrix for P1.

        Args:
            c_volume (float): Cell volume.
            coord (np.ndarray): Coordinates of the cell.
            dim (int): Dimension of the cell.
            sym (np.ndarray): Symmetric matrix.

        Returns:
            np.ndarray: Local symmetric gradient matrix of shape
                (num_faces_of_cell, num_faces_of_cell).
        """
        dphi = self.scalar_discr.local_grads(coord, dim)
        grad = spl.block_diag(*([dphi] * dim))
        return c_volume * sym @ grad

    def assemble_symgrad_symgrad_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Returns the symgrad-symgrad matrix operator for the lowest order
        vector Lagrange element. The matrix is multiplied by twice the Lame' parameter
        mu.

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Additional data, the Lame' parameter mu. Defaults to
                None.

        Returns:
            sps.csc_array: Sparse symgrad-symgrad matrix of shape
                (sd.num_nodes, sd.num_nodes).
                The matrix obtained from the discretization.
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
        tensor_mass = sps.block_diag([coeff * mass] * np.square(sd.dim)).tocsc()

        return symgrad.T @ tensor_mass @ symgrad

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        div = self.assemble_div_matrix(sd)
        symgrad = self.assemble_symgrad_matrix(sd)

        return sps.block_array([[symgrad], [div]]).tocsc()

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

        # return the global stiffness matrix
        return sym_sym + div_div

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the discretization class that contains the range of the differential.

        Args:
            dim (int): The dimension of the range.

        Returns:
            Discretization: The discretization class that contains the range of
                the differential.

        Raises:
            NotImplementedError: If there is no range discretization for the vector
                Lagrangian 1 in PyGeoN.
        """
        raise NotImplementedError(
            "There's no range discr for the vector Lagrangian 1 in PyGeoN"
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
            data (dict): Data for the computation including the Lame parameters accessed
                with the keys "lambda" and "mu". Both float and np.ndarray are accepted.

        Returns:
            ndarray: The stress tensor.
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

    def proj_to_pwLinears(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a vector Lagrangian function to a piecewise
        vector linear function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        proj = self.scalar_discr.proj_to_pwLinears(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()

    def proj_to_pwConstants(self, sd: pg.Grid) -> sps.csc_array:
        """
        Construct the matrix for projecting a vector Lagrangian function to a piecewise
        vector constant function.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_array: The matrix representing the projection.
        """
        proj = self.scalar_discr.proj_to_pwConstants(sd)
        return sps.block_diag([proj] * sd.dim).tocsc()
