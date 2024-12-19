import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class RT0Test(unittest.TestCase):

    def test_0d(self):
        sd = pp.PointGrid(np.zeros(3))

        discr = pg.RT0("flow")
        M = discr.assemble_mass_matrix(sd)
        self.assertEqual(M.shape, (0, 0))

    def test0(self):
        N, dim = 2, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.RT0("flow")
        self.assertEqual(discr.ndof(sd), sd.num_faces)

        M = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.array(
        [0.372678  , 0.372678  , 0.33333333, 0.372678  , 0.74535599,
        0.33333333, 0.372678  , 0.74535599, 0.372678  , 0.33333333,
        0.74535599, 0.74535599, 0.33333333, 0.372678  , 0.372678  ,
        0.372678  ]
        )

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

        fun = lambda x: x[0] + x[1]
        faces = sd.tags["domain_boundary_faces"]

        vals = discr.assemble_nat_bc(sd, fun, faces.nonzero()[0])
        vals_from_bool = discr.assemble_nat_bc(sd, fun, faces)

        # fmt: off
        vals_known = np.array(
        [ 0.25, -0.25,  0.  ,  0.75,  0.  ,  0.  ,  1.25,  0.  , -0.75,
        0.  ,  0.  ,  0.  ,  0.  ,  1.75, -1.25, -1.75]
        )
        # fmt: on

        self.assertTrue(np.allclose(vals, vals_known))
        self.assertTrue(np.allclose(vals_from_bool, vals_known))

    def test_mass_matrix(self):
        discr = pg.RT0("flow")
        discr_pp = pp.RT0("flow")

        for dim in np.arange(1, 4):
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            M = discr.assemble_mass_matrix(sd)

            data = discr.create_dummy_data(sd)
            discr_pp.discretize(sd, data)

            M_pp = data[pp.DISCRETIZATION_MATRICES]["flow"][
                discr_pp.mass_matrix_key
            ].tocsc()

            self.assertTrue(np.allclose(M.data, M_pp.data))
            self.assertTrue(np.allclose(M.indptr, M_pp.indptr))
            self.assertTrue(np.allclose(M.indices, M_pp.indices))

    def test_range_discr_class(self):
        discr = pg.RT0("flow")
        self.assertTrue(discr.get_range_discr_class(2) is pg.PwConstants)


if __name__ == "__main__":
    RT0Test().test_mass_matrix()
    # unittest.main()
