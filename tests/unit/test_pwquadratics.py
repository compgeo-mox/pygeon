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
        sd = pg.unit_grid(dim, 1, as_mdg=False)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        M = discr.assemble_mass_matrix(sd)

        discr_l2 = pg.Lagrange2()
        proj = discr_l2.proj_to_pwQuadratics(sd)
        M_l2 = discr_l2.assemble_mass_matrix(sd)

        self.assertTrue(np.allclose((proj.T @ M @ proj - M_l2).data, 0))

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
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.PwQuadratics()
        P = discr.eval_at_cell_centers(sd)

        func = lambda x: (x[0] + x[1]) ** 2  # Example function
        true_vals = [func(x) for x in sd.cell_centers.T]

        interp = discr.interpolate(sd, func)

        self.assertTrue(np.allclose(P @ interp, true_vals))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.PwQuadratics()

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces
        vals = discr.assemble_nat_bc(sd, func, b_faces)

        self.assertTrue(np.allclose(vals, 0))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.PwQuadratics()

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_source(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.PwQuadratics()

        func = lambda _: 2
        source = discr.source_term(sd, func)

        source_known = np.zeros(discr.ndof(sd))
        source_known[: sd.num_nodes] = 1 / 12

        self.assertTrue(np.allclose(source, source_known))


if __name__ == "__main__":
    PwQuadraticsTest().test_assemble_mass_matrix()
    # unittest.main()
