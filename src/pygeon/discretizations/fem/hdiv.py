import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class RT0(pg.Discretization, pp.RT0):
    """
    Discretization class for Raviart-Thomas of lowest order.
    Each degree of freedom is the integral over a mesh face.
    """

    def __init__(self, keyword: str) -> None:
        pg.Discretization.__init__(self, keyword)
        pp.RT0.__init__(self, keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args
            sd: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.
        """

        return sd.num_faces

    def create_dummy_data(self, sd: pg.Grid, data):
        """
        Updates data such that it has all the necessary components for pp.RT0

        Args
            sd: grid, or a subclass.
            data: dict or None.

        Returns
            data: dictionary with required attributes
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
            data.update({pp.PARAMETERS: {self.keyword: {"second_order_tensor": perm}}})

        try:
            data[pp.DISCRETIZATION_MATRICES][self.keyword]
        except KeyError:
            data.update({pp.DISCRETIZATION_MATRICES: {self.keyword: {}}})

        return data

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the mass matrix

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        data = self.create_dummy_data(sd, data)
        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.mass_matrix_key]

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the lumped mass matrix L such that
        B^T L^{-1} B is a TPFA method.

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            lumped_matrix: the lumped mass matrix.
        """
        # Get dictionary for parameter storage
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Retrieve the permeability
        k = parameter_dictionary["second_order_tensor"]

        h_perp = np.zeros(sd.num_faces)
        for (face, cell) in zip(*sd.cell_faces.nonzero()):
            inv_k = np.linalg.inv(k.values[:, :, cell])
            dist = sd.face_centers[:, face] - sd.cell_centers[:, cell]
            h_perp[face] += dist.T @ inv_k @ dist / np.linalg.norm(dist)

        return sps.diags(h_perp / sd.face_areas)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csc_matrix: the differential matrix.
        """
        return sd.cell_faces.T

    def interpolate(self, sd: pg.Grid, func):
        """
        Interpolates a function onto the finite element space

        Args
            sd: grid, or a subclass.
            func: a function that returns the function values at coordinates

        Returns
            array: the values of the degrees of freedom
        """
        vals = [
            np.inner(func(x).flatten(), normal)
            for (x, normal) in zip(sd.face_centers.T, sd.face_normals.T)
        ]
        return np.array(vals)

    def eval_at_cell_centers(self, sd: pg.Grid):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """

        data = self.create_dummy_data(sd, None)
        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.vector_proj_key]

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))

        for dof in b_faces:
            vals[dof] = func(sd.face_centers[:, dof]) * np.sum(
                sd.cell_faces.tocsr()[dof, :]
            )

        return vals

    def get_range_discr_class(self, dim: int):
        return pg.PwConstants


class BDM1(pg.Discretization):
    def ndof(self, sd: pp.Grid) -> int:
        """
        Return the number of degrees of freedom associated to the method.
        In this case the number of faces times the dimension.

        Parameter
        ---------
        sd: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        if isinstance(sd, pg.Grid):
            return sd.num_faces * sd.dim
        else:
            raise ValueError

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):

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
            for (face, nodes) in enumerate(indices.T):
                tangents = (
                    sd.nodes[:, face_nodes_loc[:, face]]
                    - sd.nodes[:, opposite_node[:, face]]
                )
                normal = sd.face_normals[:, faces_loc[face]]
                for (index, node) in enumerate(nodes):
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

    def local_inner_product(self, dim):
        M_loc = np.ones((dim + 1, dim + 1)) + np.identity(dim + 1)
        M_loc /= (dim + 1) * (dim + 2)

        M = sps.lil_matrix((3 * (dim + 1), 3 * (dim + 1)))
        for i in np.arange(3):
            mask = np.arange(i, i + 3 * (dim + 1), 3)
            M[np.ix_(mask, mask)] = M_loc

        return M.tocsc()

    def proj_to_RT0(self, sd: pg.Grid):
        return sps.hstack([sps.eye(sd.num_faces)] * sd.dim) / sd.dim

    def proj_from_RT0(self, sd: pg.Grid):
        return sps.vstack([sps.eye(sd.num_faces)] * sd.dim)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csc_matrix: the differential matrix.
        """
        RT0_diff = pg.RT0.assemble_diff_matrix(self, sd)
        proj_to_rt0 = self.proj_to_RT0(sd)

        return RT0_diff * proj_to_rt0

    def eval_at_cell_centers(self, sd):

        eval_rt0 = pg.RT0(self.keyword).eval_at_cell_centers(sd)
        proj_to_rt0 = self.proj_to_RT0(sd)

        return eval_rt0 * proj_to_rt0

    def interpolate(self, sd: pg.Grid, func):
        vals = np.zeros(self.ndof(sd))

        for face in np.arange(sd.num_faces):
            func_loc = np.array(
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, face].indices]
            ).T
            vals_loc = sd.face_normals[:, face] @ func_loc
            vals[face + np.arange(sd.dim) * sd.num_faces] = vals_loc

        return vals

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))
        local_mass = pg.Lagrange1.local_mass(None, 1, sd.dim - 1)

        for face in b_faces:
            sign = np.sum(sd.cell_faces.tocsr()[face, :])
            loc_vals = np.array(
                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, face].indices]
            )

            vals[face + np.arange(sd.dim) * sd.num_faces] = sign * local_mass @ loc_vals

        return vals

    def get_range_discr_class(self, dim: int):
        return pg.PwConstants

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):

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

            for (face, nodes) in enumerate(indices.T):
                tangents = (
                    sd.nodes[:, face_nodes_loc[:, face]]
                    - sd.nodes[:, opposite_node[:, face]]
                )
                normal = sd.face_normals[:, faces_loc[face]]
                for (index, node) in enumerate(nodes):
                    Bdm_basis[:, face + index * (sd.dim + 1)] = tangents[
                        :, index
                    ] / np.dot(tangents[:, index], normal)

            for node in nodes_uniq:
                bf_is_at_node = dof_loc.flatten() == node
                basis = Bdm_basis[:, bf_is_at_node]
                A = basis.T @ basis
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
