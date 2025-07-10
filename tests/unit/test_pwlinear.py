import unittest

import numpy as np
import porepy as pp

import pygeon as pg


class PwLinearsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()
        assert discr.ndof(sd) == 24

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.PwLinears()
        M = discr.assemble_mass_matrix(sd)

        M_known = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 24

        self.assertTrue(np.allclose(M.todense(), M_known))

    def test_assemble_lumped_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        M = discr.assemble_lumped_matrix(sd)

        # fmt: off
        M_known_data = np.full(discr.ndof(sd), 1/24)

        M_known_indices = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23]
        )

        M_known_indptr = np.array(
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24]
        )
        # fmt: on

        self.assertTrue(np.allclose(M.data, M_known_data))
        self.assertTrue(np.allclose(M.indptr, M_known_indptr))
        self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        B = discr.assemble_diff_matrix(sd)
        self.assertTrue(B.nnz == 0)

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        stiff = discr.assemble_stiff_matrix(sd)
        self.assertTrue(stiff.nnz == 0)

    def test_interpolate(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        interp = discr.interpolate(sd, lambda x: x[0])
        P = discr.eval_at_cell_centers(sd)
        known = sd.cell_centers[0]

        self.assertTrue(np.allclose(P @ interp, known))

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        known_func = np.ones(discr.ndof(sd))

        P = discr.eval_at_cell_centers(sd)

        self.assertTrue(np.allclose(P @ known_func, np.ones(sd.num_cells)))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        func = lambda x: np.sin(x[0])
        b_faces = np.array([0, 1, 3])

        b = discr.assemble_nat_bc(sd, func, b_faces)
        self.assertTrue(np.all(b == 0))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwLinears()

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_proj_to_quadratics(self):
        sd = pg.unit_grid(2, 1.0, as_mdg=False)
        sd.compute_geometry()

        P1 = pg.PwLinears()
        Proj = P1.proj_to_pwQuadratics(sd)
        M_1 = P1.assemble_mass_matrix(sd)

        test_func = np.arange(P1.ndof(sd))
        norm_test_func = test_func @ M_1 @ test_func

        P2 = pg.PwQuadratics()
        M_2 = P2.assemble_mass_matrix(sd)

        quad_func = Proj @ test_func
        norm_quad_func = quad_func @ M_2 @ quad_func

        self.assertTrue(np.isclose(norm_test_func, norm_quad_func))

    def test_proj_to_pwconstants(self):
        P1 = pg.PwLinears()
        P0 = pg.PwConstants()

        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            Proj = P1.proj_to_pwConstants(sd)
            fun_P1 = P1.interpolate(sd, lambda x: np.sum(x))
            fun_P0 = P0.interpolate(sd, lambda x: np.sum(x))

            self.assertTrue(np.allclose(Proj @ fun_P1 - fun_P0, 0.0))


if __name__ == "__main__":
    unittest.main()
