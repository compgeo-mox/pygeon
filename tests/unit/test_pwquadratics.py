import unittest
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp


class PwQuadraticsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        assert discr.ndof(sd) == sd.num_cells * 6

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        M = discr.assemble_mass_matrix(sd)

        discr_l2 = pg.Lagrange2()
        M_l2 = discr_l2.assemble_mass_matrix(sd)

        self.assertTrue(np.allclose((M - M_l2).data, 0))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        D = discr.assemble_diff_matrix(sd).todense()
        D_known = sps.csc_array((0, discr.ndof(sd))).todense()

        self.assertTrue(np.allclose(D, D_known))

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        D = discr.assemble_stiff_matrix(sd).todense()
        D_known = sps.csc_array((discr.ndof(sd), discr.ndof(sd))).todense()

        self.assertTrue(np.allclose(D, D_known))

    def test_interpolate(self):
        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 1, as_mdg=False, structured=True)
            sd.compute_geometry()

            discr = pg.PwQuadratics()

            func = lambda x: x[0] ** 2  # Example function
            vals = discr.interpolate(sd, func)

            M = discr.assemble_mass_matrix(sd)

            self.assertTrue(np.isclose(vals @ M @ vals, 1 / 5))

    def test_eval_at_cell_centers(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwQuadratics("P0")
        P = discr.eval_at_cell_centers(sd)

        # fmt: off
        P_known_data = np.array(
        [8., 8., 8., 8., 8., 8., 8., 8.]
        )

        P_known_indices = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7]
        )

        P_known_indptr = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
        # fmt: on

        self.assertTrue(np.allclose(P.data, P_known_data))
        self.assertTrue(np.allclose(P.indptr, P_known_indptr))
        self.assertTrue(np.allclose(P.indices, P_known_indices))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwQuadratics("P0")

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces
        vals = discr.assemble_nat_bc(sd, func, b_faces)

        vals_known = np.zeros(sd.num_cells)

        self.assertTrue(np.allclose(vals, vals_known))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwQuadratics("P0")

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_error_l2(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwQuadratics("P0")
        num_sol = np.array([0.5, 0.3, 0.7, 0.9, 0.1, 0, 0.2, 1])

        ana_sol = lambda x: np.sin(x[0])

        err = discr.error_l2(sd, num_sol, ana_sol)
        err_known = 7.798438721533104

        self.assertTrue(np.allclose(err, err_known))

    def test_source(self):
        dim = 2
        sd = pp.StructuredTriangleGrid([2] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        discr = pg.PwQuadratics("P0")

        func = lambda _: 2
        source = discr.source_term(sd, func)

        self.assertTrue(np.allclose(source, 2))

    def test_proj_to_pwlinears(self):

        for dim in [1, 2, 3]:
            sd = pg.unit_grid(dim, 0.5, as_mdg=False)
            sd.compute_geometry()

            p0 = pg.PwQuadratics()
            proj_p0 = p0.proj_to_pwLinears(sd)
            mass_p0 = p0.assemble_mass_matrix(sd)

            p1 = pg.PwLinears()
            mass_p1 = p1.assemble_mass_matrix(sd)

            diff = proj_p0.T @ mass_p1 @ proj_p0 - mass_p0

            self.assertTrue(np.allclose(diff.data, 0.0))


if __name__ == "__main__":
    PwQuadraticsTest().test_interpolate()
    # unittest.main()
