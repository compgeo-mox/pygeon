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
        """
        Computes the mass matrix
        """

        # Allocate the data to store matrix entries
        cell_nodes = sd.cell_nodes()
        size = int(np.sum(np.square(2 * np.sum(cell_nodes, 0))))

        rows_I = np.empty(size, dtype=int)
        cols_J = np.empty(size, dtype=int)
        data_V = np.empty(size)
        idx = 0

        dof = self.get_dof_enumeration(sd)
        disc_VL1 = pg.VLagrange1("dummy")

        tangents = sd.nodes * sd.face_ridges
        cell_diams = sd.cell_diameters(cell_nodes)

        for cell, diam in enumerate(cell_diams):
            faces_loc = sd.cell_faces[:, cell].indices

            # Obtain local indices of dofs, ordered by associated node number
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

            vals = BDM_basis.T @ BDM_basis
            VL_mass = disc_VL1.assemble_loc_mass_matrix(sd, cell, diam, dof_node[::2])
            VL_mass = np.kron(VL_mass, np.ones((2, 2)))

            A = np.multiply(vals, VL_mass)

            # Save values for the local matrix in the global structure
            cols = np.tile(dof_indx, (dof_indx.size, 1))
            loc_idx = slice(idx, idx + cols.size)

            rows_I[loc_idx] = cols.T.ravel()
            cols_J[loc_idx] = cols.ravel()
            data_V[loc_idx] = A.ravel()
            idx += cols.size

        # Construct the global matrices
        return sps.csc_matrix((data_V, (rows_I, cols_J)))

    def proj_to_VRT0(self, sd: pg.Grid):
        dof = self.get_dof_enumeration(sd).tocoo()
        return sps.csc_matrix((np.ones(self.ndof(sd)), (dof.col, dof.data))) / 2

    def proj_from_RT0(self, sd: pg.Grid):
        raise NotImplementedError

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
        raise NotImplementedError
