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
            loc_vals = np.array(
                [func(sd.nodes[:, node]) for node in nodes_loc]
            )
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

        # Allocate the data to store matrix entries
        cell_node_pairs = np.abs(sd.face_nodes) * np.abs(sd.cell_faces)
        size = int(np.sum(np.square(cell_node_pairs.data)))
        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_IJ = np.empty(size)
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

                tangents_loc = tangents[:, faces_loc]
                normals_loc = sd.face_normals[:, faces_loc]

                Bdm_basis = tangents_loc[:, ::-1] / np.sum(
                    tangents_loc[:, ::-1] * normals_loc, axis=0
                )
                A = subvolume * Bdm_basis.T @ Bdm_basis

                # Save values for the local matrix in the global structure
                loc_ind = dof[node, faces_loc].data
                cols = np.tile(loc_ind, (loc_ind.size, 1))
                loc_idx = slice(idx, idx + cols.size)

                rows_I[loc_idx] = cols.T.ravel()
                cols_J[loc_idx] = cols.ravel()
                data_IJ[loc_idx] = A.ravel()
                idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_IJ, (rows_I, cols_J)))
