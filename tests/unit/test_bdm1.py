""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class BDM1Test(unittest.TestCase):
    def test0(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_matrix(sd, None)

        discr_rt0 = pp.RT0("flow")
        data = pp.initialize_default_data(sd, {}, "flow", {})
        discr_rt0.discretize(sd, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_rt0.mass_matrix_key]

        E = sps.bmat([[sps.eye(sd.num_faces)] * 2])

        check = E * mass_bdm1 * E.T - mass_rt0

        self.assertEqual(check.nnz, 0)

    def test1(self):
        N, dim = 2, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_matrix(sd, None)

        discr_rt0 = pp.RT0("flow")
        data = pp.initialize_default_data(sd, {}, "flow", {})
        discr_rt0.discretize(sd, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES]["flow"][discr_rt0.mass_matrix_key]

        E = sps.bmat([[sps.eye(sd.num_faces)] * 3])

        check = E * mass_bdm1 * E.T - mass_rt0
        self.assertEqual(check.nnz, 0)

    def test3(self):
        N, dim = 20, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_lumped_matrix(sd, None)

        div = discr_bdm1.assemble_diff_matrix(sd)

        # assemble the saddle point problem
        spp = sps.bmat([[mass_bdm1, -div.T], [div, None]], format="csc")

        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

        b_face_centers = sd.face_centers[:, b_faces]

        faces, _, sign = sps.find(sd.cell_faces)
        sign = sign[np.unique(faces, return_index=True)[1]]

        bc_val = np.zeros(sd.num_faces * 2)
        bc_val[b_faces] = -sign[b_faces] * b_face_centers[0, :] / 2
        bc_val[b_faces + sd.num_faces] = -sign[b_faces] * b_face_centers[0, :] / 2

        rhs = np.zeros(spp.shape[0])
        rhs[: bc_val.size] += bc_val

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-sd.num_cells :]

        save = pp.Exporter(sd, "sol")
        save.write_vtu([("p", p)])

    def test4(self):
        N, dim = 3, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr_bdm1 = pg.BDM1("flow")
        mass_bdm1 = discr_bdm1.assemble_lumped_matrix(sd, None)

        div = discr_bdm1.assemble_diff_matrix(sd)

        # assemble the saddle point problem
        spp = sps.bmat([[mass_bdm1, -div.T], [div, None]], format="csc")

        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

        b_face_centers = sd.face_centers[:, b_faces]

        faces, _, sign = sps.find(sd.cell_faces)
        sign = sign[np.unique(faces, return_index=True)[1]]

        bc_val = np.zeros(sd.num_faces * sd.dim)
        bc_val[b_faces] = -sign[b_faces] * b_face_centers[0, :] / sd.dim
        bc_val[b_faces + sd.num_faces] = -sign[b_faces] * b_face_centers[0, :] / sd.dim
        bc_val[b_faces + 2 * sd.num_faces] = (
            -sign[b_faces] * b_face_centers[0, :] / sd.dim
        )

        rhs = np.zeros(spp.shape[0])
        rhs[: bc_val.size] += bc_val

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-sd.num_cells :]

        save = pp.Exporter(sd, "sol")
        save.write_vtu([("p", p)])


if __name__ == "__main__":
    BDM1Test().test0()
    # unittest.main()
