import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class VecRT1Test(unittest.TestCase):
    def test_ndof_2d(self):
        sd = pg.reference_element(2)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        self.assertEqual(vec_rt1.ndof(sd), 16)

    def test_ndof_3d(self):
        sd = pg.reference_element(3)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        self.assertEqual(vec_rt1.ndof(sd), 45)

    def test_trace_2d(self):
        sd = pg.unit_grid(2, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        B = vec_rt1.assemble_trace_matrix(sd)

        fun = lambda _: np.array([[1, 2, 0], [3, 4, 0]])
        u = vec_rt1.interpolate(sd, fun)

        trace = B @ u

        self.assertTrue(np.allclose(trace, 5))

    def test_trace_3d(self):
        sd = pg.unit_grid(3, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        B = vec_rt1.assemble_trace_matrix(sd)

        fun = lambda _: np.array([[1, 2, 3], [4, 5, 6], [0, 0, 7]])
        u = vec_rt1.interpolate(sd, fun)

        trace = B @ u

        self.assertTrue(np.allclose(trace, 13))

    def test_eval_at_cell_centers_2d(self):
        N = 1
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        def linear(x):
            return np.array([x, 2 * x])

        interp = vec_rt1.interpolate(sd, linear)
        eval = vec_rt1.eval_at_cell_centers(sd) @ interp
        eval = np.reshape(eval, (6, sd.num_cells))

        known = np.array([linear(x).ravel() for x in sd.cell_centers.T]).T

        self.assertAlmostEqual(np.linalg.norm(eval - known), 0)

    def test_eval_at_cell_centers_3d(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()

        def linear(x):
            return np.array([x, 2 * x, -x])

        interp = vec_rt1.interpolate(sd, linear)
        eval = vec_rt1.eval_at_cell_centers(sd) @ interp
        eval = np.reshape(eval, (9, sd.num_cells))

        known = np.array([linear(x).ravel() for x in sd.cell_centers.T]).T
        self.assertAlmostEqual(np.linalg.norm(eval - known), 0)

    def test_range(self):
        vec_rt1 = pg.VecRT1()
        self.assertTrue(vec_rt1.get_range_discr_class(2) is pg.VecPwLinears)

    def test_assemble_mass_matrix_2d(self):
        N = 10
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()
        key = vec_rt1.keyword

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        M = vec_rt1.assemble_mass_matrix(sd, data)

        fun = lambda _: np.array([[1, 2, 0], [4, 3, 0]])
        u = vec_rt1.interpolate(sd, fun)

        self.assertAlmostEqual(u.T @ M @ u, 26)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
        M = vec_rt1.assemble_mass_matrix(sd, data)
        self.assertAlmostEqual(u.T @ M @ u, 30)

    def test_assemble_mass_matrix_3d(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_rt1 = pg.VecRT1()
        key = vec_rt1.keyword

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        M = vec_rt1.assemble_mass_matrix(sd, data)

        fun = lambda _: np.array([[1, 2, 0], [4, 3, 0], [0, 1, 1]])
        u = vec_rt1.interpolate(sd, fun)

        self.assertAlmostEqual(u.T @ M @ u, 27)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
        M = vec_rt1.assemble_mass_matrix(sd, data)
        self.assertAlmostEqual(u.T @ M @ u, 32)

    # def test_assemble_asym_matrix_2d(self):
    #     N = 1
    #     sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     vec_rt1 = pg.VecRT1()

    #     fun = lambda _: np.array([[1, 2, 0], [4, 3, 0]])
    #     u = vec_rt1.interpolate(sd, fun)
    #     asym = vec_rt1.assemble_asym_matrix(sd)

    #     p0 = pg.PwConstants("p0")
    #     cell_asym_u = p0.eval_at_cell_centers(sd) @ (asym @ u)

    #     self.assertTrue(np.allclose(cell_asym_u, 2))

    # def test_assemble_asym_matrix_3d(self):
    #     N = 1
    #     sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     vec_rt1 = pg.VecRT1()

    #     fun = lambda _: np.array([[1, 2, -1], [4, 3, 2], [1, 1, 1]])
    #     u = vec_rt1.interpolate(sd, fun)
    #     asym = vec_rt1.assemble_asym_matrix(sd)

    #     p0 = pg.VecPwConstants("p0")
    #     cell_asym_u = p0.eval_at_cell_centers(sd) @ (asym @ u)
    #     cell_asym_u = cell_asym_u.reshape((3, -1))

    #     self.assertTrue(np.allclose(cell_asym_u[0], -1))
    #     self.assertTrue(np.allclose(cell_asym_u[1], -2))
    #     self.assertTrue(np.allclose(cell_asym_u[2], 2))

    # def test_assemble_mass_matrix_cosserat_2d(self):
    #     N = 10
    #     sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     vec_rt1 = pg.VecRT1()

    #     data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5, "mu_c": 0.25}}}
    #     M = vec_rt1.assemble_mass_matrix_cosserat(sd, data)

    #     fun = lambda _: np.array([[1, 2, 0], [4, 3, 0]])
    #     u = vec_rt1.interpolate(sd, fun)

    #     self.assertAlmostEqual(u.T @ M @ u, 28)

    # def test_assemble_mass_matrix_cosserat_3d(self):
    #     N = 1
    #     sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
    #     pg.convert_from_pp(sd)
    #     sd.compute_geometry()

    #     vec_rt1 = pg.VecRT1()

    #     data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5, "mu_c": 0.25}}}
    #     M = vec_rt1.assemble_mass_matrix_cosserat(sd, data)

    #     fun = lambda _: np.array([[1, 2, 0], [4, 3, 0], [0, 1, 1]])
    #     u = vec_rt1.interpolate(sd, fun)

    #     self.assertAlmostEqual(u.T @ M @ u, 29.5)


if __name__ == "__main__":
    # VecRT1Test().test_trace_2d()
    unittest.main()
