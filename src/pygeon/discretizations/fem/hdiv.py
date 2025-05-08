"""Module for the discretizations of the H(div) space."""

from typing import Callable, Optional, Tuple, Type, Union

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

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles the mass matrix

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
            Assembles the lumped mass matrix L such that B^T L^{-1} B is a TPFA method.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a function onto the finite element space

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix for evaluating the solution at the cell centers.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term (n dot q, func)_Gamma

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray],
            np.ndarray], relative: Optional[bool] = True,
            etype: Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a
            function.
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

    @staticmethod
    def create_unitary_data(
        keyword: str, sd: pg.Grid, data: Optional[dict] = None
    ) -> dict:
        """
        Updates data such that it has all the necessary components for pp.RT0, if the
        second order tensor is not present, it is set to the identity. It represents
        the inverse of the diffusion tensor (permeability for porous media).

        Args:
            keyword (str): The keyword for the discretization.
            sd (pg.Grid): Grid object or a subclass.
            data (dict): Dictionary object or None.

        Returns:
            dict: Dictionary with required attributes.
        """
        if data is None:
            data = {
                pp.PARAMETERS: {keyword: {}},
                pp.DISCRETIZATION_MATRICES: {keyword: {}},
            }

        try:
            data[pp.PARAMETERS][keyword]["second_order_tensor"]
        except KeyError:
            perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
            data[pp.PARAMETERS].update({keyword: {"second_order_tensor": perm}})

        try:
            data[pp.DISCRETIZATION_MATRICES][keyword]
        except KeyError:
            data.update({pp.DISCRETIZATION_MATRICES: {keyword: {}}})

        return data

    def assemble_mass_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for
                scaling, in particular the second_order_tensor that is the inverse of
                the diffusion tensor (permeability for porous media).

        Returns:
            sps.csc_array: The mass matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((sd.num_faces, sd.num_faces))

        # create unitary data, unitary permeability, in case not present
        data = RT0.create_unitary_data(self.keyword, sd, data)

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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

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
            sign (np.ndarray): The sign associated to each of the face of the degree of
                freedom
            dim (int): The dimension of the grid.

        Return:
            np.ndarray: The value of the basis functions.
        """
        N = coord.flatten("F").reshape((-1, 1)) * np.ones(
            (1, dim + 1)
        ) - np.concatenate((dim + 1) * [coord])

        return (N * sign).T

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_array: The finite element solution evaluated at the cell centers.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((3 * sd.num_faces, sd.num_faces))

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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the lumped mass matrix L such that B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for
                scaling. In particular the second_order_tensor that is the inverse of
                the diffusion tensor (permeability for porous media).

        Returns:
            sps.csc_array: The lumped mass matrix.
        """
        if data is None:
            data = RT0.create_unitary_data(self.keyword, sd, data)

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

        return sps.diags_array(h_perp / sd.face_areas).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator, the divergence
        in this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        return sps.csc_array(sd.cell_faces.T)

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

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
            vals[dof] = (
                func(sd.face_centers[:, dof]) * sd.cell_faces.tocsr()[dof, :].sum()
            )

        return vals

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
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
        relative: bool = True,
        etype: str = "specific",
        data: Optional[dict] = None,
    ) -> float:
        """
        Returns the l2 error computed against an analytical solution given as a
        function.

        Args:
            sd (pg.Grid): Grid, or a subclass.
            num_sol (np.ndarray): Vector of the numerical solution.
            ana_sol (Callable[[np.ndarray], np.ndarray]): Function that represents the
                analytical solution.
            relative (Optional[bool], optional): Compute the relative error or not.
                Defaults to True.
            etype (Optional[str], optional): Type of error computed. Defaults to
                "specific".

        Returns:
            float: The computed error.
        """
        if etype == "standard":
            return super().error_l2(sd, num_sol, ana_sol, relative, etype)

        proj = self.eval_at_cell_centers(sd)
        int_sol = np.vstack([ana_sol(x).T for x in sd.cell_centers.T]).T
        num_sol = (proj @ num_sol).reshape((3, -1))

        D = sps.diags_array(sd.cell_volumes)
        norm = np.trace(int_sol @ D @ int_sol.T) if relative else 1

        diff = num_sol - int_sol
        return np.sqrt(np.trace(diff @ D @ diff.T) / norm)


class BDM1(pg.Discretization):
    """
    BDM1 is a class that represents the BDM1 (Brezzi-Douglas-Marini) finite element
    method. It provides methods for assembling matrices, projecting to and from the RT0
    space, evaluating the solution at cell centers, interpolating a given function onto
    the grid, assembling the natural boundary condition term, and more.

    Attributes:
        keyword (str): The keyword associated with the BDM1 method.

    Methods:
        ndof(sd: pp.Grid) -> int:
            Return the number of degrees of freedom associated to the method.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles the mass matrix for the given grid.

        local_inner_product(dim: int) -> sps.csc_array:
            Compute the local inner product matrix for the given dimension.

        proj_to_RT0(sd: pg.Grid) -> sps.csc_array:
            Project the function space to the lowest order Raviart-Thomas (RT0) space.

        proj_from_RT0(sd: pg.Grid) -> sps.csc_array:
            Project the RT0 finite element space onto the faces of the given grid.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator.

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Evaluate the finite element solution at the cell centers of the given grid.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a given function onto the grid.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray],
            b_faces: np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term.

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        assemble_lumped_matrix(sd: pg.Grid, data: Optional[dict] = None)
            -> sps.csc_array:
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
    ) -> sps.csc_array:
        """
        Assembles the mass matrix for the given grid.

        Args:
            sd (pg.Grid): The grid for which the mass matrix is assembled.
            data (Optional[dict]): Additional data for the assembly process.

        Returns:
            sps.csc_array: The assembled mass matrix.
        """
        size = np.square(sd.dim * (sd.dim + 1)) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        M = self.local_inner_product(sd.dim)

        inv_K = pp.SecondOrderTensor(np.ones(sd.num_cells))
        if data is not None:
            inv_K = (
                data.get(pp.PARAMETERS, {})
                .get(self.keyword, {})
                .get("second_order_tensor", inv_K)
            )

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
            A = Psi @ M @ weight @ Psi.T * sd.cell_volumes[c]  # type: ignore[union-attr]

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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def eval_basis_at_node(
        self,
        sd: pg.Grid,
        opposites: np.ndarray,
        faces_loc: np.ndarray,
        return_node_ind: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_array: The projection matrix to the RT0 space.
        """
        proj = sps.hstack([sps.eye_array(sd.num_faces)] * sd.dim) / sd.dim
        return proj.tocsc()

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_array:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_array: The projection matrix.
        """
        return sps.vstack([sps.eye_array(sd.num_faces)] * sd.dim).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        rt0 = pg.RT0(self.keyword)
        RT0_diff = rt0.assemble_diff_matrix(sd)

        proj_to_rt0 = self.proj_to_RT0(sd)
        return RT0_diff @ proj_to_rt0

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_array: The finite element solution evaluated at the cell centers.
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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

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
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, [face]].indices]
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

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
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
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid.

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """
        # Allocate the data to store matrix entries, that's the most efficient
        # way to create a sparse matrix.
        size = sd.dim * sd.dim * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        inv_K = pp.SecondOrderTensor(np.ones(sd.num_cells))
        if data is not None:
            inv_K = (
                data.get(pp.PARAMETERS, {})
                .get(self.keyword, {})
                .get("second_order_tensor", inv_K)
            )

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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))


class RT1(pg.Discretization):
    """
    RT1 Discretization class for H(div) finite element method.

    This class implements the Raviart-Thomas elements of order 1 (RT1) for
    discretizing vector fields in H(div) space. It provides methods for
    assembling mass matrices, differential matrices, evaluating basis functions,
    and interpolating functions onto the finite element space.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of degrees of freedom for the given grid.

        assemble_mass_matrix(sd: pg.Grid, data: Optional[dict] = None) -> sps.csc_array:
            Assembles the mass matrix for the given grid and optional physical
            parameters.

        local_inner_product(dim: int) -> np.ndarray:
            Assembles the local inner product matrix based on the Lagrange2 element.

        reorder_faces(cell_faces: sps.csc_array, opposite_nodes: sps.csc_array,
            cell: int) ->  Tuple[np.ndarray]:
            Reorders the local nodes, faces, and corresponding cell-face orientations.

        eval_basis_functions(sd: pg.Grid, nodes_loc: np.ndarray, signs_loc: np.ndarray,
            volume: float) -> np.ndarray:

        eval_basis_functions_at_center(sd: pg.Grid, nodes_loc: np.ndarray,
            volume: float) ->  np.ndarray:

        eval_at_cell_centers(sd: pg.Grid) -> sps.csc_array:
            Evaluates the finite element solution at the cell centers of the given grid.

        assemble_diff_matrix(sd: pg.Grid) -> sps.csc_array:
            Assembles the matrix corresponding to the differential operator
            (divergence).

        compute_local_div_matrix(dim: int) -> np.ndarray:
            Assembles the local divergence matrix using local node and face ordering.

        interpolate(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray])
            -> np.ndarray:
            Interpolates a function onto the finite element space.

        assemble_nat_bc(sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces:
            np.ndarray) -> np.ndarray:
            Assembles the natural boundary condition term (n dot q, func)_Gamma.

        get_range_discr_class(dim: int) -> pg.Discretization:
    """

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
    ) -> sps.csc_array:
        """
        Assembles the mass matrix

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for
                scaling, in particular the second_order_tensor that is the inverse of
                the diffusion tensor (permeability for porous media).

        Returns:
            sps.csc_array: The mass matrix.
        """
        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((0, 0))

        # create unitary data, unitary permeability, in case not present
        data = RT0.create_unitary_data(self.keyword, sd, data)

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

        # Precompute the local inner product matrix
        M = self.local_inner_product(sd.dim)

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            nodes_loc, faces_loc, signs_loc = self.reorder_faces(
                sd.cell_faces, opposite_nodes, c
            )

            Psi = self.eval_basis_functions(
                sd, nodes_loc, signs_loc, sd.cell_volumes[c]
            )

            weight = np.kron(np.eye(M.shape[0] // 3), inv_K.values[:, :, c])

            # Compute the H_div-mass local matrix
            A = Psi @ M @ weight @ Psi.T * sd.cell_volumes[c]

            # Get the indices for the local face and cell degrees of freedom
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
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def local_inner_product(self, dim: int) -> np.ndarray:
        """
        Assembles the local inner products based on the Lagrange2 element

        Args:
            dim (int): Dimension of the grid

        Returns:
            np.ndarray: The local mass matrix.
        """
        lagrange2 = pg.Lagrange2()
        # We first need the barycentric coordinates lambda_i for each i
        expnts_nodes = np.eye(dim + 1)

        # For each edge (i, j) also consider the products \lambda_i \lambda_j
        # by setting the appropriate exponents to one
        edge_nodes = lagrange2.get_local_edge_nodes(dim)
        expnts_edges = np.zeros((dim + 1, edge_nodes.shape[0]))
        for ind, edge in enumerate(edge_nodes):
            expnts_edges[edge, ind] = 1

        expnts = np.hstack((expnts_nodes, expnts_edges))
        M = lagrange2.assemble_barycentric_mass(expnts)

        return np.kron(M, np.eye(3))

    def reorder_faces(
        self, cell_faces: sps.csc_array, opposite_nodes: sps.csc_array, cell: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reorders the local nodes, faces, and corresponding cell-face orientations

        Args:
            cell_faces (sps.csc_array): cell_face connectivity of the grid
            opposite_nodes (sps.csc_array): opposite nodes for each face
            cell (int): cell index

        Returns:
            np.ndarray: The reordered local node indices
            np.ndarray: The reordered local face indices
            np.ndarray: The reordered cell-face orientation signs
        """
        # For the current cell retrieve its faces
        loc = slice(cell_faces.indptr[cell], cell_faces.indptr[cell + 1])
        faces_loc = cell_faces.indices[loc]
        signs_loc = cell_faces.data[loc]
        opposites_loc = opposite_nodes.data[loc]

        # Sort the nodes in ascending order
        nodes_loc = np.sort(opposites_loc)

        # Reorder the faces so that
        # - face_0 is (0, 1) and opposite node 2
        # - face_1 is (0, 2) and opposite node 1
        # - face_2 is (1, 2) and opposite node 0
        # I.e. the faces are reordered so that
        # the opposite node indices are descending
        sorter = np.argsort(opposites_loc)[::-1]
        faces_loc = faces_loc[sorter]
        signs_loc = signs_loc[sorter]

        return nodes_loc, faces_loc, signs_loc

    def eval_basis_functions(
        self, sd: pg.Grid, nodes_loc: np.ndarray, signs_loc: np.ndarray, volume: float
    ) -> np.ndarray:
        """
        Evaluates the basis functions at the nodes and edges of a cell.

        Args:
            sd (pg.Grid): the grid
            nodes_loc (np.ndarray): nodes of the cell
            signs_loc (np.ndarray): cell-face orientation signs
            volume (float): cell volume

        Returns:
            np.ndarray: An array Psi in which [i, 3j : 3(j + 1)] contains
                the values of basis function phi_i at evaluation point j
        """
        dim = sd.dim

        # We assign each basis function to a node opposite a face (opp_node)
        # and the node at which the dof is located (loc_node)
        opp_nodes = np.tile(np.arange(dim + 1), dim)[::-1]
        loc_nodes = np.repeat(np.arange(dim + 1), dim)
        signs = np.tile(signs_loc, dim)

        # Compute the tangent in physical space by taking local indices
        tangent = lambda i, j: sd.nodes[:, nodes_loc[j]] - sd.nodes[:, nodes_loc[i]]

        # Helper functions psi_k as outlined in the notes in docs/RT1.md
        edge_nodes = pg.Lagrange2().get_local_edge_nodes(dim)
        n_edges = edge_nodes.shape[0]

        psi_nodes = np.zeros((dim + 1, 3 * (dim + 1)))
        psi_edges = np.zeros((dim + 1, 3 * n_edges))

        for edge, (i, j) in enumerate(edge_nodes):
            psi_edges[i, 3 * edge : 3 * (edge + 1)] = tangent(i, j)
            psi_edges[j, 3 * edge : 3 * (edge + 1)] = tangent(j, i)

        psi_k = np.hstack((psi_nodes, psi_edges))

        # Preallocation
        Psi = np.zeros((dim * (dim + 2), 3 * (dim + 1 + n_edges)))

        # Evaluate the basis functions of the face-dofs
        for dof, (i, j) in enumerate(zip(loc_nodes, opp_nodes)):
            Psi[dof, 3 * i : 3 * (i + 1)] = tangent(j, i)
            Psi[dof] -= psi_k[j] - psi_k[i]
            Psi[dof] *= signs[dof]

        # Evaluate the basis functions of the cell-dofs
        Psi[-dim:] = psi_k[:dim]

        return Psi / (dim * volume)

    def eval_basis_functions_at_center(
        self, sd: pg.Grid, nodes_loc: np.ndarray, volume: float
    ) -> np.ndarray:
        """
        Evaluates the basis functions at the center of a cell.

        Args:
            sd (pg.Grid): the grid
            nodes_loc (np.ndarray): nodes of the cell
            volume (float): cell volume

        Returns:
            np.ndarray: A (3 x dim) array with the values of the
                cell-based basis functions at the cell center.
        """
        # Preallocation
        basis = np.empty((3, sd.dim))

        # As outlined in docs/RT1, the only nonzero basis function
        # at the center are the cell-based ones, given by
        # phi_k = sum_i lambda_k lambda_i tau_ki
        for ind in np.arange(sd.dim):
            basis[:, ind] = np.sum(
                sd.nodes[:, nodes_loc] - sd.nodes[:, nodes_loc[ind]][:, None],
                axis=1,
            ) / ((sd.dim + 1) ** 2)

        return basis / (sd.dim * volume)

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_array:
        """
        Evaluate the finite element solution at the cell centers of the given grid.

        Args:
            sd (pg.Grid): The grid on which to evaluate the solution.

        Returns:
            sps.csc_array: The finite element solution evaluated at the cell centers.
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
            nodes_loc = np.sort(opposite_nodes.data[loc])

            P = self.eval_basis_functions_at_center(sd, nodes_loc, sd.cell_volumes[c])

            cell_dofs = sd.num_faces * sd.dim + sd.num_cells * np.arange(sd.dim) + c

            # Save values for projection P local matrix in the global structure
            loc_idx = slice(idx, idx + P.size)
            rows_I[loc_idx] = np.repeat(c + np.arange(3) * sd.num_cells, sd.dim)
            cols_J[loc_idx] = np.tile(cell_dofs, 3)
            data_IJ[loc_idx] = P.ravel()
            idx += P.size

        # Construct the global matrix
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_array:
        """
        Assembles the matrix corresponding to the differential operator, the divergence
        in this case.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_array: The differential matrix.
        """
        # Allocate the data to store matrix A entries
        size = (sd.dim * (sd.dim + 2)) * (sd.dim + 1) * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Precompute the local divergence matrix
        loc_div = self.compute_local_div_matrix(sd.dim)
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            _, faces_loc, signs_loc = self.reorder_faces(
                sd.cell_faces, opposite_nodes, c
            )

            # Change the sign of the face-dofs according to the cell-face orientation
            signs = np.ones(loc_div.shape[1])
            signs[: -sd.dim] = np.tile(signs_loc, sd.dim)

            div = loc_div * signs / (sd.dim * sd.cell_volumes[c])

            # Indices of the local degrees of freedom
            loc_face = np.hstack([faces_loc] * sd.dim)
            loc_face += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces
            loc_cell = sd.dim * sd.num_faces + np.arange(sd.dim) * sd.num_cells + c
            loc_ind = np.hstack((loc_face, loc_cell))
            loc_ind = np.tile(loc_ind, sd.dim + 1)

            # Indices of the range degrees of freedom
            pwlinear_ind = sd.num_cells * np.arange(sd.dim + 1) + c
            pwlinear_ind = np.repeat(pwlinear_ind, div.shape[1])

            # Save values of the local matrix in the global structure
            loc_idx = slice(idx, idx + div.size)
            rows_I[loc_idx] = pwlinear_ind
            cols_J[loc_idx] = loc_ind
            data_IJ[loc_idx] = div.ravel()
            idx += div.size

        # Construct the global matrices
        return sps.csc_array((data_IJ, (rows_I, cols_J)))

    def compute_local_div_matrix(self, dim: int) -> np.ndarray:
        """
        Assembles the local divergence matrix using local node and face ordering

        Args:
            dim (int): dimension of the grid

        Returns:
            np.ndarray: The local divergence matrix
        """
        opp_node = np.tile(np.arange(dim + 1), dim)[::-1]
        loc_node = np.repeat(np.arange(dim + 1), dim)

        # The face basis function phi_i^j has divergence
        # 1 + (dim + 1) (lambda_i - lambda_j)
        face_div = np.ones((dim + 1, dim * (dim + 1)))
        face_div[loc_node, np.arange(loc_node.size)] += dim + 1
        face_div[opp_node, np.arange(loc_node.size)] -= dim + 1

        # The cell basis function phi_k has divergence
        # (dim + 1) lambda_k - 1
        cell_div = (dim + 1) * np.eye(dim + 1, dim)
        cell_div -= 1

        return np.hstack((face_div, cell_div))

    def interpolate(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolates a function onto the finite element space

        Args:
            sd (pg.Grid): Grid, or a subclass.
            func (Callable[[np.ndarray], np.ndarray]): A function that returns the
                function values at coordinates.

        Returns:
            np.ndarray: The values of the degrees of freedom.
        """

        # The face dofs are determined as in BDM1
        interp_faces = pg.BDM1().interpolate(sd, func)

        # The cell dofs are determined by solving (d x d) linear systems
        interp_cells = np.zeros(sd.dim * sd.num_cells)
        cell_nodes = sd.cell_nodes()

        for c in np.arange(sd.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
            nodes_loc = cell_nodes.indices[loc]

            basis_at_center = self.eval_basis_functions_at_center(
                sd, nodes_loc, sd.cell_volumes[c]
            )
            func_at_center = func(sd.cell_centers[:, c])

            # Compute the coefficients c_i such
            # that sum_i c_i phi_i = f at the cell center
            coefficients = np.linalg.solve(
                basis_at_center[: sd.dim, :], func_at_center[: sd.dim]
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

    def get_range_discr_class(self, dim: int) -> Type[pg.Discretization]:
        """
        Returns the range discretization class for the given dimension.

        Args:
            dim (int): The dimension of the range space.

        Returns:
            pg.Discretization: The range discretization class.
        """
        return pg.PwLinears

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_array:
        """
        Assembles the lumped matrix for the given grid,
        using the integration rule from Egger & Radu (2020)

        Args:
            sd (pg.Grid): The grid object.
            data (Optional[dict]): Optional data dictionary.

        Returns:
            sps.csc_array: The assembled lumped matrix.
        """

        # If a 0-d grid is given then we return an empty matrix
        if sd.dim == 0:
            return sps.csc_array((0, 0))

        bdm1 = pg.BDM1(self.keyword)
        bdm1_lumped = bdm1.assemble_lumped_matrix(sd, data) / (sd.dim + 2)

        # create unitary data, unitary permeability, in case not present
        data = RT0.create_unitary_data(self.keyword, sd, data)

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the inverse of permeability
        inv_K = parameter_dictionary["second_order_tensor"]

        # Allocate the data to store matrix P entries
        size = sd.dim * sd.dim * sd.num_cells
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
        idx = 0

        # Compute the opposite nodes for each face
        opposite_nodes = sd.compute_opposite_nodes()

        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            nodes_loc = np.sort(opposite_nodes.data[loc])

            P = self.eval_basis_functions_at_center(sd, nodes_loc, sd.cell_volumes[c])
            weight = inv_K.values[:, :, c]

            A = P.T @ weight @ P * sd.cell_volumes[c] * (sd.dim + 1) / (sd.dim + 2)

            loc_ind = sd.num_cells * np.arange(sd.dim) + c

            # Save values for projection P local matrix in the global structure
            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + A.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.ravel()
            idx += A.size

        # Construct the global matrix
        cell_dof_lumped = sps.csc_array((data_IJ, (rows_I, cols_J)))

        return sps.block_diag((bdm1_lumped, cell_dof_lumped), "csc")
