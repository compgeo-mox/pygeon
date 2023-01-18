import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class MVEM(pg.Discretization, pp.MVEM):
    """
    Each degree of freedom is the integral over a mesh face.
    """

    def __init__(self, keyword: str) -> None:
        pg.Discretization.__init__(self, keyword)
        pp.MVEM.__init__(self, keyword)

    def ndof(self, sd: pg.Grid) -> int:
        """
        Returns the number of faces.

        Args
            sd: grid, or a subclass.

        Returns
            dof: the number of degrees of freedom.
        """

        return sd.num_faces

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the mass matrix

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        data = pg.RT0.create_dummy_data(self, sd, data)
        pp.MVEM.discretize(self, sd, data)
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

        return pg.RT0.assemble_lumped_matrix(self, sd, data)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
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

    def eval_at_cell_centers(self, sd: pg.Grid, data=None):
        """
        Assembles the matrix

        Args
            sd: grid, or a subclass.

        Returns
            matrix: the evaluation matrix.
        """

        data = pg.RT0.create_dummy_data(self, sd, data)
        pp.MVEM.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.vector_proj_key]

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma
        """
        return pg.RT0.assemble_nat_bc(self, sd, func, b_faces)

    def get_range_discr_class(self, dim: int):
        return pg.PwConstants


class VBDM1(pg.Discretization):
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
            return sd.face_nodes.nnz
        else:
            raise ValueError

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        raise NotImplementedError

    def proj_to_VRT0(self, sd: pg.Grid):
        dof = self.get_dof_enumeration(sd).tocoo()
        return sps.csc_matrix((np.ones(self.ndof(sd)), (dof.col, dof.data))) / 2

    #
    #    def proj_from_RT0(self, sd: pg.Grid):
    #        return sps.vstack([sps.eye(sd.num_faces)] * sd.dim)
    #
    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """
        VRT0_diff = pg.MVEM.assemble_diff_matrix(self, sd)
        proj_to_vrt0 = self.proj_to_VRT0(sd)

        return VRT0_diff * proj_to_vrt0

    def eval_at_cell_centers(self, sd):
        raise NotImplementedError

    def interpolate(self, sd: pg.Grid, func):
        raise NotImplementedError

    #        vals = np.zeros(self.ndof(sd))
    #
    #        for face in np.arange(sd.num_faces):
    #            func_loc = np.array(
    #                [func(sd.nodes[:, node]) for node in sd.face_nodes[:, face].indices]
    #            ).T
    #            vals_loc = sd.face_normals[:, face] @ func_loc
    #            vals[face + np.arange(sd.dim) * sd.num_faces] = vals_loc
    #
    #        return vals

    def assemble_nat_bc(self, sd: pg.Grid, func, b_faces):
        """
        Assembles the natural boundary condition term
        (n dot q, func)_\Gamma
        """
        if b_faces.dtype == "bool":
            b_faces = np.where(b_faces)[0]

        vals = np.zeros(self.ndof(sd))
        local_mass = pg.Lagrange1.local_mass(None, 1, sd.dim - 1)
        dof = self.get_dof_enumeration(sd)

        for face in b_faces:
            sign = np.sum(sd.cell_faces.tocsr()[face, :])
            nodes_loc = sd.face_nodes[:, face].indices
            loc_vals = np.array([func(sd.nodes[:, node]) for node in nodes_loc])
            dof_loc = dof[nodes_loc, face].data

            vals[dof_loc] = sign * local_mass @ loc_vals

        return vals

    def get_range_discr_class(self, dim: int):
        return pg.PwConstants

    def get_dof_enumeration(self, sd):
        dof = sd.face_nodes.copy()
        dof.data = np.arange(sd.face_nodes.nnz)
        return dof

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Quadrature that is in the Overleaf.
        Assumes uv is linear and uses three-point quadrature per subsimplex
        """

        # Allocate the data to store matrix entries
        cell_node_pairs = np.abs(sd.face_nodes) * np.abs(sd.cell_faces)
        size = int(np.sum(np.square(cell_node_pairs.data)))

        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        dof = self.get_dof_enumeration(sd)
        subvolumes = sd.compute_subvolumes()
        face_nodes = sd.face_nodes.tocsr()
        tangents = sd.nodes * sd.face_ridges

        for c in np.arange(sd.num_cells):
            loc = slice(subvolumes.indptr[c], subvolumes.indptr[c + 1])
            nodes_loc = subvolumes.indices[loc]
            subvolumes_loc = subvolumes.data[loc]

            faces_of_cell = sd.cell_faces[:, c]

            for node, subvolume in zip(nodes_loc, subvolumes_loc):
                faces_of_node = face_nodes[node, :].T
                faces_loc = faces_of_node.multiply(faces_of_cell).indices

                tangents_loc = tangents[:, faces_loc[::-1]]
                normals_loc = sd.face_normals[:, faces_loc]

                Bdm_basis = tangents_loc / np.sum(tangents_loc * normals_loc, axis=0)
                A = subvolume * Bdm_basis.T @ Bdm_basis

                # Save values for the local matrix in the global structure
                loc_ind = dof[node, faces_loc].data
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)

                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_V[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrix
        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def assemble_lumped_matrix_quad(self, sd: pg.Grid, data: dict = None):
        """
        Uses Simpson's rule on the faces and assumes that uv is linear in the interior
        """

        # Allocate the data to store matrix entries
        size = int(16 * np.sum(np.abs(sd.cell_faces)))

        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        dof = self.get_dof_enumeration(sd)
        subsimplices = sd.compute_subvolumes(True)[1]

        tangents = sd.nodes * sd.face_ridges
        M_scaling = np.kron(np.ones((2, 2)) + np.eye(2), np.ones((2, 2)))

        for c in np.arange(sd.num_cells):
            loc = slice(subsimplices.indptr[c], subsimplices.indptr[c + 1])
            faces_loc = subsimplices.indices[loc]
            subsimplices_loc = subsimplices.data[loc]

            # Obtain local indices of dofs, oredered by associated node number
            local_dof = dof[:, faces_loc].tocsr().tocoo()
            dof_indx = local_dof.data
            dof_node = local_dof.row
            dof_face = faces_loc[local_dof.col]

            # Compute the values of the basis functions
            swapper = np.arange(dof_face.size)
            swapper[::2] += 1
            swapper[1::2] -= 1
            swapped_tangents = tangents[:, dof_face[swapper]]

            BDM_basis = swapped_tangents / np.sum(
                swapped_tangents * sd.face_normals[:, dof_face], axis=0
            )

            for face, subvolume in zip(faces_loc, subsimplices_loc):
                nodes_of_face = sd.face_nodes[:, face].indices
                loc_ind = np.logical_or(
                    dof_node == nodes_of_face[0], dof_node == nodes_of_face[1]
                )

                loc_basis = BDM_basis[:, loc_ind]

                A = subvolume * loc_basis.T @ loc_basis / 6
                A *= M_scaling

                # Save values for the local matrix in the global structure
                cols = np.tile(dof_indx[loc_ind], (loc_ind.sum(), 1))
                loc_idx = slice(idx, idx + cols.size)

                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_V[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def assemble_lumped_matrix_midpoint(self, sd: pg.Grid, data: dict = None):
        """
        Assumes that u and v are piecewise linear in the interior and applies a
        three-point quadrature on the subsimplices
        """

        # Allocate the data to store matrix entries
        size = int(np.sum(np.square(2 * np.sum(np.abs(sd.cell_faces), 0))))

        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        dof = self.get_dof_enumeration(sd)
        subvolumes = sd.compute_subvolumes()

        tangents = sd.nodes * sd.face_ridges

        for c in np.arange(sd.num_cells):
            loc = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
            faces_loc = sd.cell_faces.indices[loc]

            # Obtain local indices of dofs, oredered by associated node number
            local_dof = dof[:, faces_loc].tocsr().tocoo()
            dof_indx = local_dof.data
            dof_node = local_dof.row
            dof_face = faces_loc[local_dof.col]

            dof_subvolume = subvolumes[dof_node, c].data

            # Compute the values of the basis functions
            swapper = np.arange(dof_face.size)
            swapper[::2] += 1
            swapper[1::2] -= 1
            swapped_tangents = tangents[:, dof_face[swapper]]

            BDM_basis = swapped_tangents / np.sum(
                swapped_tangents * sd.face_normals[:, dof_face], axis=0
            )

            A = (dof_subvolume * BDM_basis).T @ (dof_subvolume * BDM_basis)
            A /= sd.cell_volumes[c]

            # Save values for the local matrix in the global structure
            cols = np.tile(dof_indx, (dof_indx.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_V, (rows_I, cols_J)))
