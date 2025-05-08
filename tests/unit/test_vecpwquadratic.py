import unittest
import numpy as np
import scipy.linalg as spla

import pygeon as pg
import porepy as pp


class VecPwQuadraticsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()
        self.assertTrue(
            discr.ndof(sd) == sd.num_cells * ((dim * (dim + 1)) // 2 + dim + 1) * dim
        )

    def test_assemble_mass_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()
        M = discr.assemble_mass_matrix(sd)

        quad = pg.PwQuadratics()
        M_base = quad.assemble_mass_matrix(sd).todense()

        M_known = spla.block_diag(M_base, M_base)

        self.assertTrue(np.allclose(M.todense(), M_known))

    # def test_assemble_lumped_matrix(self):
    #     dim = 2
    #     sd = pg.unit_grid(dim, 0.5, as_mdg = False, structured=True)

    #     sd.compute_geometry()

    #     discr = pg.VecPwQuadratics()

    #     M = discr.assemble_lumped_matrix(sd)

    #     # fmt: off
    #     M_known_data = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 6

    #     M_known_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    #     M_known_indptr = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
    #     # fmt: on

    #     self.assertTrue(np.allclose(M.data, M_known_data))
    #     self.assertTrue(np.allclose(M.indptr, M_known_indptr))
    #     self.assertTrue(np.allclose(M.indices, M_known_indices))

    def test_assemble_diff_matrix(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()

        B = discr.assemble_diff_matrix(sd)

        self.assertTrue(B.nnz == 0)

    def test_assemble_stiff_matrix(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()

        self.assertRaises(NotImplementedError, discr.assemble_stiff_matrix, sd)

    def test_interpolate(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)

        sd.compute_geometry()

        discr = pg.VecPwQuadratics()

        interp = discr.interpolate(sd, lambda x: x**2)
        P = discr.eval_at_cell_centers(sd)
        known = (sd.cell_centers[:dim] ** 2).ravel()

        self.assertTrue(np.allclose(P @ interp, known))

    def test_assemble_nat_bc(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()

        func = lambda x: np.sin(x[0])  # Example function

        b_faces = np.array([0, 1, 3])  # Example boundary faces
        vals = discr.assemble_nat_bc(sd, func, b_faces)

        vals_known = np.zeros(discr.ndof(sd))

        self.assertTrue(np.allclose(vals, vals_known))

    def test_get_range_discr_class(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecPwQuadratics()

        self.assertRaises(
            NotImplementedError,
            discr.get_range_discr_class,
            dim,
        )

    def test_error_l2(self):
        dim = 2
        sd = pg.unit_grid(dim, 0.5, as_mdg=False, structured=True)

        sd.compute_geometry()

        discr = pg.VecPwQuadratics()
        ana_sol = lambda x: [x[0], x[0] * x[1]]
        num_sol = discr.interpolate(sd, ana_sol)

        error = discr.error_l2(sd, num_sol, ana_sol)
        self.assertTrue(np.isclose(error, 0))


if __name__ == "__main__":
    unittest.main()
