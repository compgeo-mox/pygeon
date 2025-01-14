import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class RT1Test(unittest.TestCase):

    def test_mass(self):
        N, dim = 1, 2
        # sd = pp.CartGrid([N] * dim, [1] * dim)
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        # sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.RT1()
        # self.assertEqual(discr.ndof(sd), sd.num_faces)

        face_mass = discr.assemble_mass_matrix(sd)

        discr_p1 = pg.PwLinears()
        cell_mass = discr_p1.assemble_mass_matrix(sd, None)

        div = cell_mass @ discr.assemble_diff_matrix(sd)

        def q_0(x):
            return x**2

        q_interp = discr.interpolate(sd, q_0)

        # assemble the saddle point problem
        spp = sps.bmat([[face_mass, -div.T], [div, None]], format="csc")

        b_faces = sd.tags["domain_boundary_faces"]

        def p_0(x):
            return x[0]

        bc_val = -discr.assemble_nat_bc(sd, p_0, b_faces)

        rhs = np.zeros(spp.shape[0])
        rhs[: bc_val.size] += bc_val

        # solve the problem
        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()

        q = x[: bc_val.size]
        p = x[-discr_p1.ndof(sd) :]

        # face_proj = discr.eval_at_cell_centers(sd)
        cell_proj = discr_p1.eval_at_cell_centers(sd)

        # cell_q = (face_proj * q).reshape((3, -1))
        cell_p = cell_proj @ p

        # known_q = np.zeros(cell_q.shape)
        # known_q[0, :] = -1.0
        known_p = sd.cell_centers[0, :]

        # self.assertAlmostEqual(np.linalg.norm(cell_q - known_q), 0)
        # self.assertAlmostEqual(np.linalg.norm(cell_p - known_p), 0)

        # self.assertTrue(discr.get_range_discr_class(sd.dim) is pg.PwConstants)

    #     M = discr.assemble_lumped_matrix(sd)

    #     # fmt: off
    #     M_known_data = np.array(
    #     [0.372678  , 0.372678  , 0.33333333, 0.372678  , 0.74535599,
    #     0.33333333, 0.372678  , 0.74535599, 0.372678  , 0.33333333,
    #     0.74535599, 0.74535599, 0.33333333, 0.372678  , 0.372678  ,
    #     0.372678  ]
    #     )

    #     M_known_indices = np.array(
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
    #     )

    #     M_known_indptr = np.array(
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
    #     )
    #     # fmt: on

    #     self.assertTrue(np.allclose(M.data, M_known_data))
    #     self.assertTrue(np.allclose(M.indptr, M_known_indptr))
    #     self.assertTrue(np.allclose(M.indices, M_known_indices))

    #     fun = lambda x: x[0] + x[1]
    #     faces = sd.tags["domain_boundary_faces"]

    #     vals = discr.assemble_nat_bc(sd, fun, faces.nonzero()[0])
    #     vals_from_bool = discr.assemble_nat_bc(sd, fun, faces)

    #     # fmt: off
    #     vals_known = np.array(
    #     [ 0.25, -0.25,  0.  ,  0.75,  0.  ,  0.  ,  1.25,  0.  , -0.75,
    #     0.  ,  0.  ,  0.  ,  0.  ,  1.75, -1.25, -1.75]
    #     )
    #     # fmt: on

    #     self.assertTrue(np.allclose(vals, vals_known))
    #     self.assertTrue(np.allclose(vals_from_bool, vals_known))

    # def test_mass_matrix(self):
    #     discr = pg.RT0("flow")
    #     discr_pp = pp.RT0("flow")

    #     for dim in np.arange(1, 4):
    #         sd = pg.unit_grid(dim, 0.5, as_mdg=False)
    #         sd.compute_geometry()

    #         data = discr.create_unitary_data(sd)
    #         discr_pp.discretize(sd, data)

    #         M_pp = data[pp.DISCRETIZATION_MATRICES]["flow"][
    #             discr_pp.mass_matrix_key
    #         ].tocsc()

    #         self.assertTrue(np.allclose(M.data, M_pp.data))
    #         self.assertTrue(np.allclose(M.indptr, M_pp.indptr))
    #         self.assertTrue(np.allclose(M.indices, M_pp.indices))

    # def test_range_discr_class(self):
    #     discr = pg.RT0()
    #     self.assertTrue(discr.get_range_discr_class(2) is pg.PwConstants)


if __name__ == "__main__":
    RT1Test().test_mass()
    # unittest.main()
