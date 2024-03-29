from typing import Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class RT0(pg.Discretization, pp.RT0):
    """
    Discretization class for Raviart-Thomas of lowest order.
    Each degree of freedom is the integral over a mesh face.

    Args:
        keyword (str): The keyword for the discretization.

    Attributes:
        keyword (str): The keyword for the discretization.

    Methods:
        ndof(sd: pg.Grid) -> int:
            Returns the number of faces.

        create_dummy_data(sd: pg.Grid, data: Optional[dict] = None) -> dict:
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
            Assembles the natural boundary condition term (n dot q, func)_\Gamma

        get_range_discr_class(dim: int) -> pg.Discretization:
            Returns the range discretization class for the given dimension.

        error_l2(sd: pg.Grid, num_sol: np.ndarray, ana_sol: Callable[[np.ndarray], np.ndarray],
            relative: Optional[bool] = True, etype: Optional[str] = "specific") -> float:
            Returns the l2 error computed against an analytical solution given as a function.
    """

    def __init__(self, keyword: str) -> None:
        """
        Initialize the HDiv class.

        Args:
            keyword (str): The keyword for the discretization.

        Returns:
            None
        """
        pg.Discretization.__init__(self, keyword)
        pp.RT0.__init__(self, keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args:
            sd (pg.Grid): grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.
        """

        return sd.num_faces

    def create_dummy_data(self, sd: pg.Grid, data: Optional[dict] = None) -> dict:
        """
        Updates data such that it has all the necessary components for pp.RT0

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
            data (Optional[dict]): Optional dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The mass matrix.
        """

        data = self.create_dummy_data(sd, data)
        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.mass_matrix_key
        ].tocsc()

    def assemble_lumped_matrix(
        self, sd: pg.Grid, data: Optional[dict] = None
    ) -> sps.csc_matrix:
        """
        Assembles the lumped mass matrix L such that
        B^T L^{-1} B is a TPFA method.

        Args:
            sd (pg.Grid): Grid object or a subclass.
            data (Optional[dict]): Optional dictionary with physical parameters for scaling.

        Returns:
            sps.csc_matrix: The lumped mass matrix.
        """
        if data is None:
            data = self.create_dummy_data(sd, data)

        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability
        k = parameter_dictionary["second_order_tensor"]

        h_perp = np.zeros(sd.num_faces)
        for face, cell in zip(*sd.cell_faces.nonzero()):
            inv_k = np.linalg.inv(k.values[:, :, cell])
            dist = sd.face_centers[:, face] - sd.cell_centers[:, cell]
            h_perp_loc = dist.T @ inv_k @ dist
            norm_dist = np.linalg.norm(dist)
            h_perp[face] += h_perp_loc / norm_dist if norm_dist else 0

        return sps.diags(h_perp / sd.face_areas).tocsc()

    def assemble_diff_matrix(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix corresponding to the differential operator.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The differential matrix.
        """
        return sd.cell_faces.T

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

    def eval_at_cell_centers(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Assembles the matrix for evaluating the solution at the cell centers.

        Args:
            sd (pg.Grid): Grid object or a subclass.

        Returns:
            sps.csc_matrix: The evaluation matrix.
        """
        data = self.create_dummy_data(sd, None)
        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.vector_proj_key]

    def assemble_nat_bc(
        self, sd: pg.Grid, func: Callable[[np.ndarray], np.ndarray], b_faces: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma

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
        num_sol = (proj * num_sol).reshape((3, -1), order="F")

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
            return sd.num_faces * sd.dim
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

        cell_nodes = sd.cell_nodes()
        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its faces and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            dof_loc = np.reshape(
                sd.face_nodes[:, faces_loc].indices, (sd.dim, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            indices = np.unique(dof_loc, return_inverse=True)[1].reshape((sd.dim, -1))

            face_nodes_loc = sd.face_nodes[:, faces_loc].toarray()
            cell_nodes_loc = cell_nodes[:, c].toarray()
            # get the opposite node id for each face
            opposite_node = np.logical_xor(face_nodes_loc, cell_nodes_loc)

            # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
            Psi = np.empty((sd.dim * (sd.dim + 1), sd.dim + 1), np.ndarray)
            for face, nodes in enumerate(indices.T):
                tangents = (
                    sd.nodes[:, face_nodes_loc[:, face]]
                    - sd.nodes[:, opposite_node[:, face]]
                )
                normal = sd.face_normals[:, faces_loc[face]]
                for index, node in enumerate(nodes):
                    Psi[face + index * (sd.dim + 1), node] = tangents[
                        :, index
                    ] / np.dot(tangents[:, index], normal)
            Psi = sps.bmat(Psi)

            # Compute the inner products
            A = Psi * M * Psi.T * sd.cell_volumes[c]

            loc_ind = np.hstack([faces_loc] * sd.dim)
            loc_ind += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            # Save values of the local matrix in the global structure
            cols = np.tile(loc_ind, (loc_ind.size, 1))
            loc_idx = slice(idx, idx + cols.size)
            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_IJ[loc_idx] = A.todense().ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))

    def local_inner_product(self, dim: int) -> sps.csc_matrix:
        """
        Compute the local inner product matrix for the given dimension.

        Args:
            dim (int): The dimension of the matrix.

        Returns:
            sps.csc_matrix: The computed local inner product matrix.
        """
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        M = sps.lil_matrix((3 * (dim + 1), 3 * (dim + 1)))
        for i in np.arange(3):
            mask = np.arange(i, i + 3 * (dim + 1), 3)
            M[np.ix_(mask, mask)] = M_loc

        return M.tocsc()

    def proj_to_RT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the function space to the lowest order Raviart-Thomas (RT0) space.

        Args:
            sd (pg.Grid): The grid object representing the computational domain.

        Returns:
            sps.csc_matrix: The projection matrix to the RT0 space.
        """
        return sps.hstack([sps.eye(sd.num_faces)] * sd.dim) / sd.dim

    def proj_from_RT0(self, sd: pg.Grid) -> sps.csc_matrix:
        """
        Project the RT0 finite element space onto the faces of the given grid.

        Args:
            sd (pg.Grid): The grid on which the projection is performed.

        Returns:
            sps.csc_matrix: The projection matrix.
        """
        return sps.vstack([sps.eye(sd.num_faces)] * sd.dim)

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
        rt0 = pg.RT0(self.keyword)
        eval_rt0 = rt0.eval_at_cell_centers(sd)

        proj_to_rt0 = self.proj_to_RT0(sd)
        return eval_rt0 @ proj_to_rt0

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
        (n dot q, func)_\Gamma

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
        local_mass = p1.local_mass(np.ones(1), sd.dim - 1)

        vals = np.zeros(self.ndof(sd))
        for face in b_faces:
            sign = np.sum(sd.cell_faces.tocsr()[face, :])
            loc_vals = np.array(
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, face].indices]
            )

            vals[face + np.arange(sd.dim) * sd.num_faces] = sign * local_mass @ loc_vals

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

        cell_nodes = sd.cell_nodes()
        for c in np.arange(sd.num_cells):
            # For the current cell retrieve its ridges and
            # determine the location of the dof
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]
            dof_loc = np.reshape(
                sd.face_nodes[:, faces_loc].indices, (sd.dim, -1), order="F"
            )

            # Find the nodes of the cell and their coordinates
            nodes_uniq, indices = np.unique(dof_loc, return_inverse=True)
            indices = indices.reshape((sd.dim, -1))

            face_nodes_loc = sd.face_nodes[:, faces_loc].toarray()
            cell_nodes_loc = cell_nodes[:, c].toarray()
            # get the opposite node id for each face
            opposite_node = np.logical_xor(face_nodes_loc, cell_nodes_loc)

            # Compute a matrix Psi such that Psi[i, j] = psi_i(x_j)
            Bdm_basis = np.zeros((3, sd.dim * (sd.dim + 1)))
            Bdm_indices = np.hstack([faces_loc] * sd.dim)
            Bdm_indices += np.repeat(np.arange(sd.dim), sd.dim + 1) * sd.num_faces

            for face, nodes in enumerate(indices.T):
                tangents = (
                    sd.nodes[:, face_nodes_loc[:, face]]
                    - sd.nodes[:, opposite_node[:, face]]
                )
                normal = sd.face_normals[:, faces_loc[face]]
                for index, node in enumerate(nodes):
                    Bdm_basis[:, face + index * (sd.dim + 1)] = tangents[
                        :, index
                    ] / np.dot(tangents[:, index], normal)

            for node in nodes_uniq:
                bf_is_at_node = dof_loc.flatten() == node
                basis = Bdm_basis[:, bf_is_at_node]
                A = basis.T @ basis  # PUT INV PERM HERE
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
