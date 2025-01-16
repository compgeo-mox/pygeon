""" Module for the discretizations of the H(div) space. """

from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class RT0(pg.Discretization):
    """
    Discretization class for Raviart-Thomas of lowest order.
    Each degree of freedom is the integral over a mesh face.

    The implementation of this class is inspired by the RT0 class in PorePy.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of faces.

        create_unitary_data(sd: pg.Grid, data: Optional[dict] = None) -> dict:
            Updates data such that it has all the necessary components for pp.RT0

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped mass matrix L such that B^T L^{-1} B is a TPFA method.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a function onto the finite element space

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix for evaluating the solution at the cell centers.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term (n dot q, func)_Gamma

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray], np.ndarray],
            relative: Optional[bool] = True, etype: Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a function.
    """

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args:
            sd (pg.Grid): grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.
        """
        return sd.num_faces

    def create_unitary_data(self, sd: pg.Grid, data: Optional[dict] = None) -> dict:
        """
        Updates data such that it has all the necessary components for pp.RT0, if the
        second order tensor is not present, it is set to the identity. It represents
        the inverse of the diffusion tensor (permeability for porous media).

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (dict): Dictionary object or None.

        Returns:
            dict: Dictionary with required attributes.
        """
        if data is None:
            data = {
                pp.PARAMETERS: {self.keyword: {}},
                pp.DISCRETIZATION_MATRICES: {self.keyword: {}},
            }

        try:
            data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        except KeyError:
            perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
            data[pp.PARAMETERS].update({self.keyword: {"second_order_tensor": perm}})

        try:
            data[pp.DISCRETIZATION_MATRICES][self.keyword]
        except KeyError:
            data.update({pp.DISCRETIZATION_MATRICES: {self.keyword: {}}})

        return data

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling,
                in particular the second_order_tensor that is the inverse of the diffusion
                tensor (permeability for porous media).

        Returns:
            sps.csc_matrix: The mass matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_matrix((sd.num_faces, sd.num_faces))

        # create unitary data, unitary permeability, in case not present
        data = self.create_unitary_data(sd, data)

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the inverse of permeability
        inv_K = parameter_dictionary["second_order_tensor"]

        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        _, _, _, R, dim, nodes = pp.map_geometry.map_grid(sd)
        nodes = nodes[: sd.dim, :]

        if not data.get("is_tangential", False):
            # Rotate the inverse of the permeability tensor and delete last dimension
            if sd.dim < 3:
                inv_K = inv_K.copy()
                inv_K.rotate(R)
                remove_dim = np.where(np.logical_not(dim))[0]
                inv_K.values = np.delete(inv_K.values, (remove_dim), axis=0)
                inv_K.values = np.delete(inv_K.values, (remove_dim), axis=1)

        # Allocate the data to store matrix A entries
        size = np.square(sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the local inner product matrix
        M = self.local_inner_product(sd)

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]
            sign_loc = sd.cell_faces.data[loc]

            # get the opposite node id for each face
            coord_loc = nodes[:, opposites_loc]

            Psi = self.eval_basis(coord_loc, sign_loc, sd.dim)

            weight = np.kron(np.eye(sd.dim + 1), inv_K.values[:, :, c])

            # Compute the H_div-mass local matrix
            A = Psi @ M @ weight @ Psi.T / sd.cell_volumes[c]

            # Save values for local matrix in the global structure
            cols = np.concatenate(faces_loc.size * [[faces_loc]])
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    @staticmethod
    def local_inner_product(sd: pg.Grid) -> np.ndarray:
        """
        Compute the local inner product matrix for a given grid.

        Args:
            sd (pg.Grid): The grid object containing the discretization information.

        Returns:
            np.ndarray: local inner product matrix.
        """
        size = sd.dim * (sd.dim + 1)
        M = np.zeros((size, size))

        for it in np.arange(0, size, sd.dim):
            M += np.diagflat(np.ones(size - it), it)

        M += M.T
        M /= sd.dim * sd.dim * (sd.dim + 1) * (sd.dim + 2)
        return M

    @staticmethod
    def eval_basis(coord: np.ndarray, sign: np.ndarray, dim: int) -> np.ndarray:
        """
        Evaluate the basis functions.

        Args:
            coord (np.ndarray): the coordinates of the opposite node for each face.
            sign (np.ndarray): The sign associated to each of the face of the degree of freedom
            dim (int): The dimension of the grid.

        Return:
            np.ndarray: The value of the basis functions.
        """
        N = coord.flatten("F").reshape((-1, 1)) * np.ones(
            (1, dim + 1)
        ) - np.concatenate((dim + 1) * [coord])

        return (N * sign).T

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_matrix: The finite element solution evaluated at the cell centers.
        """
        # Map the domain to a reference geometry (i.e. equivalent to compute
        # surface coordinates in 1d and 2d)
        c_centers, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
            sd
        )

        # Allocate the data to store matrix P entries
        size = 3 * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            # get the opposite node id for each face
            coord_loc = node_coords[:, opposites_loc]

            # Compute the flux reconstruction matrix
            P = pp.RT0.faces_to_cell(
                c_centers[:, c],
                coord_loc,
                f_centers[:, faces_loc],
                f_normals[:, faces_loc],
                dim,
                R,
            )

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + P.size)
            rows_I[loc_idx] = np.repeat(c + np.arange(3) * sd.num_cells, sd.dim + 1)
            cols_J[loc_idx] = np.tile(faces_loc, 3)
            data_IJ[loc_idx] = P.ravel()
            idx += P.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix L such that B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling.
                In particular the second_order_tensor that is the inverse of the diffusion
                tensor (permeability for porous media).

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        if data is None:
            data = self.create_unitary_data(sd, data)

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the inverse of the permeability
        inv_K = parameter_dictionary["second_order_tensor"]

        h_perp = np.zeros(sd.num_faces)
        for face, cell in zip(*sd.cell_faces.nonzero()):
            dist = sd.face_centers[:, face] - sd.cell_centers[:, cell]
            h_perp_loc = dist.T @ inv_K.values[:, :, cell] @ dist
            norm_dist = np.linalg.norm(dist)
            h_perp[face] += h_perp_loc / norm_dist if norm_dist else 0

        return sps.diags(h_perp / sd.face_areas).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator, the divergence in
        this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        return sd.cell_faces.T.tocsc()

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the function
                values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """
        vals = [
            np.inner(func(x).flatten(), normal)
            for (x, normal) in zip(sd.face_centers.T, sd.face_normals.T)
        ]
        return np.array(vals)

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for dof in b_faces:
            vals[dof] = func(sd.face_centers[:, dof]) * np.sum(
                sd.cell_faces.tocsr()[dof, :]
            )

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants

    def error_l2(
        self,
        sd: pg.Grid,
        num_sol: np.ndarray,
        ana_sol: Callable[[np.ndarray], np.ndarray],
        relative: Optional[bool] = True,
        etype: Optional[str] = "specific",
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to "specific".

        Returns:
            float: The computed error.
        """
        if etype == "standard":
            return super().error_l2(sd, num_sol, ana_sol, relative, etype)

        proj = self.eval_at_cell_centers(sd)
        int_sol = np.vstack([ana_sol(x).T for x in sd.cell_centers.T]).T
        num_sol = (proj * num_sol).reshape((3, -1))

        D = sps.diags(sd.cell_volumes)
        norm = np.trace(int_sol @ D @ int_sol.T) if relative else 1

        diff = num_sol - int_sol
        return np.sqrt(np.trace(diff @ D @ diff.T) / norm)


class BDM1(pg.Discretization):
    """
    BDM1 is a class that represents the BDM1 (Brezzi-Douglas-Marini) finite element method.
    It provides methods for assembling matrices, projecting to and from the RT0 space,
    evaluating the solution at cell centers, interpolating a given function onto the grid,
    assembling the natural boundary condition term, and more.

    Attributes:
        keyword (str): The keyword associated with the BDM1 method.

    Methods:
        ndof(sd: pp.Grid) -> int:
            Return the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the mass matrix for the given grid.

        local_inner_product(dim: int) -> sps.csc_matrix:
            Compute the local inner product matrix for the given dimension.

        proj_to_RT0(sd: pg.Grid) -> sps.csc_matrix:
            Project the function space to the lowest order Raviart-Thomas (RT0) space.

        proj_from_RT0(sd: pg.Grid) -> sps.csc_matrix:
            Project the RT0 finite element space onto the faces of the given grid.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_matrix:
            Assembles the matrix corresponding to the differential operator.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_matrix:
            Evaluate the finite element solution at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
            Interpolates a given function onto the grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_matrix:
            Assembles the lumped matrix for the given grid.
    """

    def ndof(self, sd: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case the number of faces times the dimension.

        Args:
            sd (pp.Grid): Grid object or a subclass.

        Returns:
            int: The number of degrees of freedom.

        Raises:
            ValueError: If the input grid is not an instance of pp.Grid.

        """
        if isinstance(sd, pg.Grid):
            return sd.face_nodes.nnz
        else:
            raise ValueError

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the mass matrix for the given grid.

        Args:
            sd (pg.Grid): The grid for which the mass matrix is assembled.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_matrix: The assembled mass matrix.
        """
        size = np.square(sd.dim * (sd.dim + 1)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(sd.dim)

        try:
            inv_K = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        except Exception:
            inv_K = pp.SecondOrderTensor(np.ones(sd.num_cells))

        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.eval_basis_at_node(sd, opposites_loc, faces_loc)

            weight = np.kron(np.eye(sd.dim + 1), inv_K.values[:, :, c])

            # Compute the inner products
            A = Psi @ M @ weight @ Psi.T * sd.cell_volumes[c]

            loc_ind = np.hstack([faces_loc] * sd.dim)
            loc_ind += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            # Save values of the local matrix in the global structure
            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def eval_basis_at_node(
        self,
        sd: pg.Grid,
        opposites: np.ndarray,
        faces_loc: np.ndarray,
        return_node_ind: bool = False,
    ) -> np.ndarray:
        """
        Compute the local basis function for the BDM1 finite element space.

        Args:
            sd (pg.Grid): The grid object.
            opposites (np.ndarray): The local degrees of freedom.
            cell_nodes_loc (np.ndarray): The local nodes of the cell.
            faces_loc (np.ndarray): The local faces.
            return_node_ind (bool): Whether to return the local indexing of the nodes,
                                    used in assemble_lumped_matrix

        Returns:
            np.ndarray: The local mass matrix.
        """
        fn = sd.face_nodes
        nodes = np.empty((sd.dim + 1, sd.dim), int)
        for ind, face in enumerate(faces_loc):
            nodes[ind] = fn.indices[fn.indptr[face] : fn.indptr[face + 1]]
        nodes = nodes.ravel(order="F")

        node_ind = np.repeat(np.arange(sd.dim + 1), sd.dim)

        if not np.all(nodes[:: sd.dim][node_ind] == nodes):
            node_ind = np.unique(nodes, return_inverse=True)[1]

        face_ind = np.tile(np.arange(sd.dim + 1), sd.dim)

        # get the opposite node id for each face
        opposite_nodes = opposites[face_ind]

        # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
        tangents = sd.nodes[:, nodes] - sd.nodes[:, opposite_nodes]
        normals = sd.face_normals[:, faces_loc[face_ind]]
        vals = tangents / np.sum(tangents * normals, axis=0)

        # Create a (i, j, v) triplet
        dof_id = np.tile(np.arange(sd.dim * (sd.dim + 1)), 3)
        nod_id = 3 * np.tile(node_ind, (3, 1)) + np.arange(3)[:, None]

        result = np.zeros((sd.dim * (sd.dim + 1), 3 * (sd.dim + 1)))
        result[dof_id, nod_id.ravel()] = vals.ravel()

        if return_node_ind:
            return result, node_ind
        else:
            return result

    @staticmethod
    def local_inner_product(dim: int) -> np.ndarray:
        """
        Compute the local inner product matrix for the given dimension.

        Args:
            dim (int): The dimension of the matrix.

        Returns:
            np.ndarray: The computed local inner product matrix.
        """
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        return np.kron(M_loc, np.eye(3))

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_matrix: The projection matrix to the RT0 space.
        """
        return sps.hstack([sps.eye(sd.num_faces)] * sd.dim, format="csc") / sd.dim

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_matrix: The projection matrix.
        """
        return sps.vstack([sps.eye(sd.num_faces)] * sd.dim, format="csc")

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        rt0 = pg.RT0(self.keyword)
        RT0_diff = rt0.assemble_diff_matrix(sd)

        proj_to_rt0 = self.proj_to_RT0(sd)
        return RT0_diff @ proj_to_rt0

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_matrix: The finite element solution evaluated at the cell centers.
        """
        size = 3 * sd.dim * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            Psi = self.eval_basis_at_node(sd, opposites_loc, faces_loc)
            basis_at_center = np.sum(np.split(Psi, sd.dim + 1, axis=1), axis=0) / (
                sd.dim + 1
            )

            loc_ind = np.hstack([faces_loc] * sd.dim)
            loc_ind += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            # Save values of the local matrix in the global structure
            row = np.repeat(c + np.arange(3) * sd.num_cells, basis_at_center.shape[0])
            loc_idx = slice(idx, idx + row.size)
            rows_I[loc_idx] = row
            cols_J[loc_idx] = np.tile(loc_ind, 3)
            data_IJ[loc_idx] = basis_at_center.ravel(order="F")
            idx += row.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a given function onto the grid.

        Args:
            sd (pg.Grid): The grid on which to interpolate the function.
            func (Callable[[np.ndarray], np.ndarray]): The function to be interpolated.

        Returns:
            np.ndarray: The interpolated values on the grid.
        """
        vals = np.zeros(self.ndof(sd))

        for face in np.arange(sd.num_faces):
            func_loc = np.array(
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, face].indices]
            ).T
            vals_loc = sd.face_normals[:, face] @ func_loc
            vals[face + np.arange(sd.dim) * sd.num_faces] = vals_loc

        return vals

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        p1 = pg.Lagrange1(self.keyword)
        local_mass = p1.local_mass(sd.dim - 1)

        vals = np.zeros(self.ndof(sd))
        signs = sd.cell_faces @ np.ones(sd.num_cells)
        fn = sd.face_nodes

        for face in b_faces:
            loc_vals = np.array(
                [
                    func(sd.nodes[:, node])
                    for node in fn.indices[fn.indptr[face] : fn.indptr[face + 1]]
                ]
            ).ravel()

            vals[face + np.arange(sd.dim) * sd.num_faces] = (
                signs[face] * local_mass @ loc_vals
            )

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwConstants

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_matrix: The assembled lumped matrix.
        """
        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = sd.dim * sd.dim * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        try:
            inv_K = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        except Exception:
            inv_K = pp.SecondOrderTensor(np.ones(sd.num_cells))

        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            opposites_loc = opposite_nodes.data[loc]

            # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
            Psi, nod_ind = self.eval_basis_at_node(sd, opposites_loc, faces_loc, True)

            Bdm_indices = np.hstack([faces_loc] * sd.dim)
            Bdm_indices += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            for node in np.arange(sd.dim + 1):
                bf_is_at_node = nod_ind == node
                basis = Psi[bf_is_at_node, 3 * node : 3 * (node + 1)]
                A = basis @ inv_K.values[:, :, c] @ basis.T
                A *= sd.cell_volumes[c] / (sd.dim + 1)

                loc_ind = Bdm_indices[bf_is_at_node]

                # Save values for the local matrix in the global structure
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)
                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_IJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))


class RT1(pg.Discretization):

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of degrees of freedom.

        Args:
            sd (pg.Grid): grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.
        """
        return sd.dim * (sd.num_faces + sd.num_cells)

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling,
                in particular the second_order_tensor that is the inverse of the diffusion
                tensor (permeability for porous media).

        Returns:
            sps.csc_matrix: The mass matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_matrix((0, 0))

        # create unitary data, unitary permeability, in case not present
        data = RT0.create_unitary_data(self, sd, data)

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the inverse of permeability
        inv_K = parameter_dictionary["second_order_tensor"]

        # Allocate the data to store matrix A entries
        size = np.square(sd.dim * (sd.dim + 2)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the local inner product matrix
        M = self.local_inner_product(sd.dim)

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            nodes_loc, faces_loc, signs_loc = self.reorder_faces(sd, opposite_nodes, c)

            Psi = self.eval_basis_functions(
                sd, nodes_loc, sd.cell_volumes[c], signs_loc
            )

            # Compute the H_div-mass local matrix
            A = Psi @ M @ Psi.T * sd.cell_volumes[c]

            loc_face = np.hstack([faces_loc] * sd.dim)
            loc_face += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces
            loc_cell = sd.dim * sd.num_faces + sd.num_cells * np.arange(sd.dim) + c
            loc_ind = np.hstack((loc_face, loc_cell))

            # Save values of the local matrix in the global structure
            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_inner_product(self, dim: int):

        lagrange2 = pg.Lagrange2()
        alphas_nodes = np.eye(dim + 1)

        e_nodes = lagrange2.get_local_edge_nodes(dim)
        alphas_edges = np.zeros((dim + 1, e_nodes.shape[0]))
        for ind, edge in enumerate(e_nodes):
            alphas_edges[edge, ind] = 1

        alphas = np.hstack((alphas_nodes, alphas_edges))
        M = lagrange2.assemble_barycentric_mass(alphas)

        return np.kron(M, np.eye(3))

    def reorder_faces(self, sd, opposite_nodes, c):
        # For the current cell retrieve its faces
        loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
        faces_loc = sd.cell_faces.indices[loc]
        signs_loc = sd.cell_faces.data[loc]
        opposites_loc = opposite_nodes.data[loc]

        # Sort the nodes
        nodes_loc = np.sort(opposites_loc)

        # Reorder the faces so that
        # face_0 is (0, 1) and opposite node 2
        # face_1 is (0, 2) and opposite node 1
        # face_2 is (1, 2) and opposite node 0
        sorter = np.argsort(opposites_loc)[::-1]
        faces_loc = faces_loc[sorter]
        signs_loc = signs_loc[sorter]

        return nodes_loc, faces_loc, signs_loc

    def eval_basis_functions(self, sd, nodes_loc, volume, signs_loc):

        dim = sd.dim
        n_edges = dim * (dim + 1) // 2

        opp_node = np.tile(np.arange(dim + 1), dim)[::-1]
        loc_node = np.repeat(np.arange(dim + 1), dim)

        tangent = lambda i, j: sd.nodes[:, nodes_loc[j]] - sd.nodes[:, nodes_loc[i]]

        signs = np.tile(signs_loc, dim)

        if dim == 2:
            # helper functions
            psi_0 = np.zeros((dim + 1 + n_edges, 3))
            psi_0[dim + 1] = tangent(0, 1)
            psi_0[dim + 2] = tangent(0, 2)

            psi_1 = np.zeros_like(psi_0)
            psi_1[dim + 1] = tangent(1, 0)
            psi_1[dim + 3] = tangent(1, 2)

            psi_2 = np.zeros_like(psi_0)
            psi_2[dim + 2] = tangent(2, 0)
            psi_2[dim + 3] = tangent(2, 1)

            psi = np.vstack([psi.ravel() for psi in [psi_0, psi_1, psi_2]])
        else:
            raise NotImplementedError

        # Let's go
        Psi = np.zeros((dim * (dim + 2), 3 * (dim + 1 + n_edges)))

        for ind, (i, j) in enumerate(zip(loc_node, opp_node)):
            Psi[ind, 3 * i : 3 * (i + 1)] = tangent(j, i)
            Psi[ind] -= psi[j] - psi[i]
            Psi[ind] *= signs[ind]

        Psi[-dim:] = (dim + 1) ** 2 * psi[:dim]

        return Psi / (dim * volume)

    def eval_basis_at_center(self, sd, nodes_loc, volume):
        basis = np.zeros((3, sd.dim))

        for ind in np.arange(sd.dim):
            basis[:, ind] = np.sum(
                sd.nodes[:, nodes_loc]
                - sd.nodes[:, np.repeat(nodes_loc[ind], sd.dim + 1)],
                axis=1,
            )
        return basis / (sd.dim * volume)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_matrix: The finite element solution evaluated at the cell centers.
        """

        # Allocate the data to store matrix P entries
        size = 3 * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            opposites_loc = opposite_nodes.data[loc]

            P = self.eval_basis_at_center(sd, opposites_loc[::-1], sd.cell_volumes[c])

            cell_dofs = sd.num_faces * sd.dim + sd.num_cells * np.arange(sd.dim) + c
            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + P.size)
            rows_I[loc_idx] = np.repeat(c + np.arange(3) * sd.num_cells, sd.dim)
            cols_J[loc_idx] = np.tile(cell_dofs, 3)
            data_IJ[loc_idx] = P.ravel()
            idx += P.size

        # Construct the global matrix
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator, the divergence in
        this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        # Allocate the data to store matrix A entries
        size = (sd.dim * (sd.dim + 2)) * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Local indices related to dof
        loc_div = self.compute_local_div_matrix(sd)
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            _, faces_loc, signs_loc = self.reorder_faces(sd, opposite_nodes, c)

            signs = np.ones(loc_div.shape[1])
            signs[: -sd.dim] = np.tile(signs_loc, sd.dim)

            div = loc_div * signs / (sd.dim * sd.cell_volumes[c])

            loc_face = np.hstack([faces_loc] * sd.dim)
            loc_face += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces
            loc_cell = sd.dim * sd.num_faces + np.arange(sd.dim) * sd.num_cells + c
            loc_ind = np.hstack((loc_face, loc_cell))
            loc_ind = np.tile(loc_ind, sd.dim + 1)

            pwlinear_ind = np.arange((sd.dim + 1) * c, (sd.dim + 1) * (c + 1))
            pwlinear_ind = np.repeat(pwlinear_ind, div.shape[1])

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + div.size)
            rows_I[loc_idx] = pwlinear_ind
            cols_J[loc_idx] = loc_ind
            data_IJ[loc_idx] = div.ravel()
            idx += div.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def compute_local_div_matrix(self, sd):
        opp_node = np.tile(np.arange(sd.dim + 1), sd.dim)[::-1]
        loc_node = np.repeat(np.arange(sd.dim + 1), sd.dim)

        face_div = np.ones((sd.dim + 1, sd.dim * (sd.dim + 1)))
        face_div[loc_node, np.arange(loc_node.size)] += sd.dim + 1
        face_div[opp_node, np.arange(loc_node.size)] -= sd.dim + 1

        cell_div = 3 * np.eye(sd.dim + 1, sd.dim)
        cell_div -= 1
        cell_div *= (sd.dim + 1) ** 2

        loc_div = np.hstack((face_div, cell_div))
        return loc_div

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the function
                values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """

        bdm1 = pg.BDM1()
        interp_faces = bdm1.interpolate(sd, func)
        interp_cells = np.zeros(sd.dim * sd.num_cells)
        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            nodes_loc = cell_nodes.indices[
                cell_nodes.indptr[c] : cell_nodes.indptr[c + 1]
            ]
            basis_at_center = self.eval_basis_at_center(
                sd, nodes_loc, sd.cell_volumes[c]
            )[: sd.dim, :]

            coefficients = np.linalg.solve(
                basis_at_center, func(sd.cell_centers[:, c])[: sd.dim]
            )

            interp_cells[sd.num_cells * np.arange(sd.dim) + c] = coefficients

        return np.hstack((interp_faces, interp_cells))

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_Gamma

        Args:
            sd (pg.Grid): The grid object representing the computational domain.
            func (Callable[[np.ndarray], np.ndarray]): The function that defines
                the natural boundary condition.
            b_faces (np.ndarray): The array of boundary faces.

        Returns:
            np.ndarray: The assembled natural boundary condition term.
        """
        vals = np.zeros(self.ndof(sd))
        bdm1 = pg.BDM1()
        vals[: bdm1.ndof(sd)] = bdm1.assemble_nat_bc(sd, func, b_faces)

        return vals

    def get_range_discr_class(self, dim: int) -> pg.Discretization:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwLinears
