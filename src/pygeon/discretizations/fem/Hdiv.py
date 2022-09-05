import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class RT0(pg.Discretization, pp.RT0):
    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)
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

    def assemble_mass_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the mass matrix

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            mass_matrix: the mass matrix.
        """

        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.mass_matrix_key]

    def assemble_lumped_matrix(self, sd: pg.Grid, data: dict = None):
        """
        Assembles the lumped mass matrix such that a TPFA method is obtained.

        Args
            sd: grid, or a subclass.
            data: optional dictionary with physical parameters for scaling.

        Returns
            lumped_matrix: the lumped mass matrix.
        """

        h_perp = np.zeros(sd.num_faces)
        for (face, cell) in zip(*sd.cell_faces.nonzero()):
            h_perp[face] += np.linalg.norm(
                sd.face_centers[:, face] - sd.cell_centers[:, cell]
            )

        return sps.diags(h_perp / sd.face_areas)

    def assemble_diff_matrix(self, sd: pg.Grid):
        """
        Assembles the matrix corresponding to the differential

        Args
            sd: grid, or a subclass.

        Returns
            csr_matrix: the differential matrix.
        """
        P0mass = pg.PwConstants(self.keyword).assemble_mass_matrix(sd)
        P0mass.data = 1.0 / P0mass.data

        return P0mass * sd.cell_faces.T

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
            np.inner(func(x), normal)
            for (x, normal) in zip(sd.face_centers, sd.face_normals)
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

        # Create dummy data to pass to porepy.
        data = {}
        data[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
        data[pp.PARAMETERS] = {"flow": {}}
        data[pp.PARAMETERS]["flow"]["second_order_tensor"] = pp.SecondOrderTensor(
            np.ones(sd.num_cells)
        )

        pp.RT0.discretize(self, sd, data)
        return data[pp.DISCRETIZATION_MATRICES][self.keyword][self.vector_proj_key]

    def assemble_nat_bc(self, sd: pg.Grid, b_dofs):
        """
        Assembles the natural boundary condition term

        """
        raise NotImplementedError

    def get_range_discr_class(self):
        return pg.PwConstants
