from typing import Callable, Optional, Union

import numpy as np
import porepy as pp
import scipy.linalg as spl
import scipy.sparse as sps

import pygeon as pg


class Lagrange1(pg.Discretization):
    """
    Class representing the Lagrange1 finite element discretization.

    Attributes:
        None

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix for the lowest order Lagrange element.

        local_mass(c_volume: np.ndarray, dim: int) -> np.ndarray:
            Computes the local mass matrix.

        assemble_stiffness_matrix(sd: pg.Grid, data: dict) -> sps.csc_matrix:
            Assembles the stiffness matrix for the finite element method.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the differential matrix based on the dimension of the grid.

        local_stiff(K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int)
            -> np.ndarray:
            Computes the local stiffness matrix for P1.

        local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
            Calculates the local gradients of the finite element basis functions.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix for the finite element method.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Constructs the matrix for evaluating a Lagrangian function at the cell centers of
            the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a given function over the nodes of a grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the 'natural' boundary condition.

        get_range_discr_class(dim: int) -> object:
            Returns the appropriate range discretization class based on the dimension.
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
        return sd.num_nodes

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Returns the mass matrix for the lowest order Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Optional data for the assembly process.

        Returns:
            sps.csc_matrix: The mass matrix obtained from the discretization.
        """

        # Data allocation
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            # Compute the mass-H1 local matrix
            A = self.local_mass(sd.cell_volumes[c], sd.dim)

            # Save values for mass-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_mass(self, c_volume: np.ndarray, dim: int) -> np.ndarray:
        """Compute the local mass matrix.

        Args:
            c_volume (np.ndarray): Cell volume.
            dim (int): Dimension of the matrix.

        Returns:
            np.ndarray: Local mass matrix of shape (num_nodes_of_cell, num_nodes_of_cell).
        """

        M = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        return c_volume * M / ((dim + 1) * (dim + 2))

    def assemble_stiffness_matrix(self, sd: pg.Grid, data: dict) -> sps.csc_matrix:
        """
        Assembles the stiffness matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (dict): A dictionary containing the necessary data for assembling the matrix.

        Returns:
            sps.csc_matrix: The assembled stiffness matrix.
        """
        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability, boundary conditions
        k = parameter_dictionary["second_order_tensor"]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, node_coords = pp.map_geometry.map_grid(sd)

        if not data.get("is_tangential", False):
            # Rotate the permeability tensor and delete last dimension
            if sd.dim < 3:
                k = k.copy()
                k.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                k.values = np.delete(k.values, (remove_dim), axis=0)
                k.values = np.delete(k.values, (remove_dim), axis=1)

        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = np.power(sd.dim + 1, 2) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]
            coord_loc = node_coords[:, nodes_loc]

            # Compute the stiff-H1 local matrix
            A = self.local_stiff(
                k.values[0 : sd.dim, 0 : sd.dim, c],
                sd.cell_volumes[c],
                coord_loc,
                sd.dim,
            )

            # Save values for stiff-H1 local matrix in the global structure
            cols = np.tile(nodes_loc, (nodes_loc.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the differential matrix based on the dimension of the grid.

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        if sd.dim == 3:
            return sd.ridge_peaks.T.tocsc()
        elif sd.dim == 2:
            return sd.face_ridges.T.tocsc()
        elif sd.dim == 1:
            return sd.cell_faces.T.tocsc()
        elif sd.dim == 0:
            return sps.csc_matrix((0, 1))
        else:
            raise ValueError

    def local_stiff(
        self, K: np.ndarray, c_volume: np.ndarray, coord: np.ndarray, dim: int
    ) -> np.ndarray:
        """
        Compute the local stiffness matrix for P1.

        Args:
            K (np.ndarray): permeability of the cell of (dim, dim) shape.
            c_volume (np.ndarray): scalar cell volume.
            coord (np.ndarray): coordinates of the cell vertices of (dim+1, dim) shape.
            dim (int): dimension of the problem.

        Returns:
            np.ndarray: local stiffness matrix of (dim+1, dim+1) shape.
        """

        dphi = self.local_grads(coord, dim)

        return c_volume * np.dot(dphi.T, np.dot(K, dphi))

    @staticmethod
    def local_grads(coord: np.ndarray, dim: int) -> np.ndarray:
        """
        Calculate the local gradients of the finite element basis functions.

        Args:
            coord (np.ndarray): The coordinates of the nodes in the element.
            dim (int): The dimension of the element.

        Returns:
            np.ndarray: The local gradients of the finite element basis functions.
        """
        Q = np.hstack((np.ones((dim + 1, 1)), coord.T))
        invQ = np.linalg.inv(Q)
        return invQ[1:, :]

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix for the finite element method.

        Args:
            sd (pg.Grid): The grid object representing the discretization.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled lumped mass matrix.
        """
        volumes = sd.cell_nodes() * sd.cell_volumes / (sd.dim + 1)
        return sps.diags(volumes).tocsc()

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Construct the matrix for evaluating a Lagrangian function at the
        cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to construct the matrix.

        Returns:
            sps.csc_matrix: The matrix representing the projection at the cell centers.
        """
        if sd.dim == 0:
            return sd.cell_nodes().T.tocsc()

        # Allocation
        size = (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its nodes
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])

            nodes_loc = cell_nodes.indices[loc]

            loc_idx = slice(idx, idx + nodes_loc.size)
            rows_I[loc_idx] = c
            cols_J[loc_idx] = nodes_loc
            data_IJ[loc_idx] = 1.0 / (sd.dim + 1)
            idx += nodes_loc.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function over the nodes of a grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: An array containing the interpolated values at each node of the grid.
        """
        return np.array([func(x) for x in sd.nodes.T])

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the 'natural' boundary condition
        (u, func)_Gamma with u a test function in Lagrange1

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

    def get_range_discr_class(self, dim: int) -> object:
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
        lagrange1 (pg.Lagrange1): A local Lagrange1 class for performing some of the
            computations.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom associated with the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles and returns the mass matrix for the lowest order Lagrange element.

        assemble_div_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the divergence matrix operator for the lowest order vector Lagrange
            element.

        local_div(c_volume: float, coord: np.ndarray, dim: int) -> np.ndarray:
            Computes the local divergence matrix for P1.

        assemble_div_div_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Returns the div-div matrix operator for the lowest order vector Lagrange element.

        assemble_symgrad_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Returns the symmetric gradient matrix operator for the lowest order vector Lagrange
            element.

        local_symgrad(c_volume: float, coord: np.ndarray, dim: int, sym: np.ndarray)
            -> np.ndarray:
            Computes the local symmetric gradient matrix for P1.

        assemble_symgrad_symgrad_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_matrix:
            Returns the symgrad-symgrad matrix operator for the lowest order vector Lagrange
            element.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space.
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the vector discretization class.
        The scalar discretization class is pg.Lagrange1.

        Args:
            keyword (str): The keyword for the vector discretization class.

        Returns:
            None
        """
        super().__init__(keyword, pg.Lagrange1)

    def assemble_div_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.

        Returns:
            sps.csc_matrix: The div matrix obtained from the discretization.
        """
        # If a 0-d grid is given then we return a zero matrix
        if sd.dim == 0:
            return sps.csc_matrix((1, 1))

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
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

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
    ) -> sps.csc_matrix:
        """
        Returns the div-div matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Additional data. Defaults to None.

        Returns:
            matrix: sparse (sd.num_nodes, sd.num_nodes)
                Div-div matrix obtained from the discretization.
        """
        div = self.assemble_div_matrix(sd)
        # TODO add the Lame' parameter in the computation of the P0 mass
        p0 = pg.PwConstants(self.keyword)
        mass = p0.assemble_mass_matrix(sd, data)

        return div.T @ mass @ div

    def assemble_symgrad_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Returns the symmetric gradient matrix operator for the
        lowest order vector Lagrange element

        Args:
            sd (pg.Grid): The grid object representing the domain.

        Returns:
            sps.csc_matrix: The sparse symmetric gradient matrix operator.

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
            return sps.csc_matrix((1, 1))

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
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

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
    ) -> sps.csc_matrix:
        """
        Returns the symgrad-symgrad matrix operator for the lowest order
        vector Lagrange element

        Args:
            sd (pg.Grid): The grid.
            data (Optional[dict]): Additional data. Defaults to None.

        Returns:
            sps.csc_matrix: Sparse symgrad-symgrad matrix of shape
                (sd.num_nodes, sd.num_nodes).
                The matrix obtained from the discretization.
        """

        symgrad = self.assemble_symgrad_matrix(sd)
        # TODO add the Lame' parameter in the computation of the P0 mass
        p0 = pg.PwConstants(self.keyword)
        mass = p0.assemble_mass_matrix(sd, data)
        tensor_mass = sps.block_diag([mass] * np.square(sd.dim), format="csc")

        return symgrad.T @ tensor_mass @ symgrad

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        div = self.assemble_div_matrix(sd)
        symgrad = self.assemble_symgrad_matrix(sd)

        return sps.bmat([[symgrad], [div]], format="csc")

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): grid, or a subclass.
                The grid onto which the function will be interpolated.
            func (function): a function that returns the function values at coordinates
                The function to be interpolated.

        Returns:
            array: the values of the degrees of freedom
                The interpolated values of the function on the finite element space.

        NOTE: We are assuming the sd grid in the (x,y) coordinates
        """
        return self.scalar_discr.interpolate(sd, func).ravel(order="F")

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix by evaluating the Lagrange basis functions at the cell
        centers of the given grid.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The evaluation matrix.
        """
        proj = self.scalar_discr.eval_at_cell_centers(sd)
        return sps.block_diag([proj] * sd.dim, format="csc")

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (Tr q, p)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines the
                natural boundary condition.
            b_faces (np.ndarray): List of boundary faces where the natural boundary
                condition is applied.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        bc_val = []
        for d in np.arange(sd.dim):
            f = lambda x: func(x)[d]
            bc_val.append(self.scalar_discr.assemble_nat_bc(sd, f, b_faces))
        return np.hstack(bc_val)

    def get_range_discr_class(self, dim: int) -> object:
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
        labda: Union[float, np.ndarray],
        mu: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Compute the stress tensor for a given displacement field.

        Args:
            sd (pg.Grid): The spatial discretization object.
            u (ndarray): The displacement field.
            labda (float or ndarray): The first Lamé parameter.
            mu (float or ndarray): The second Lamé parameter.

        Returns:
            ndarray: The stress tensor.
        """
        # construct the differentials
        symgrad = self.assemble_symgrad_matrix(sd)
        div = self.assemble_div_matrix(sd)

        p0 = pg.PwConstants(self.keyword)
        proj = p0.eval_at_cell_centers(sd)

        # compute the two terms and split on each component
        sigma = np.array(np.split(2 * mu * symgrad @ u, np.square(sd.dim)))
        sigma[:: (sd.dim + 1)] += labda * div @ u

        # compute the actual dofs
        sigma = sigma @ proj

        # create the indices to re-arrange the components for the second
        # order tensor
        idx = np.arange(np.square(sd.dim)).reshape((sd.dim, -1), order="F")

        return sigma[idx].T
