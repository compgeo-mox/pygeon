import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class VecBDM1Test(unittest.TestCase):
    def test_trace_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_bdm1 = pg.VecBDM1("vecbdm1")

        B = vec_bdm1.assemble_trace_matrix(sd)

        fun = lambda x: np.array([[x[0] + x[1], x[0], 0], [x[1], -x[0] - x[1], 0]])
        u = vec_bdm1.interpolate(sd, fun)

        trace = B @ u

        self.assertTrue(np.allclose(trace, 0))

    def test_ndof_2d(self):
        sd = pp.StructuredTriangleGrid([1] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_bdm1 = pg.VecBDM1("vecbdm1")

        self.assertEqual(vec_bdm1.ndof(sd), 20)

    def test_assemble_mass_matrix_2d(self):
        N = 10
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vecbdm1"
        vec_bdm1 = pg.VecBDM1(key)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        M = vec_bdm1.assemble_mass_matrix(sd, data)

        fun = lambda _: np.array([[1, 2, 0], [4, 3, 0]])
        u = vec_bdm1.interpolate(sd, fun)

        self.assertAlmostEqual(u.T @ M @ u, 26)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
        M = vec_bdm1.assemble_mass_matrix(sd, data)
        self.assertAlmostEqual(u.T @ M @ u, 30)

    def test_eval_at_cell_centers_2d(self):
        N = 1
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vecbdm1"
        vec_bdm1 = pg.VecBDM1(key)

        def linear(x):
            return np.array([x, 2 * x])

        interp = vec_bdm1.interpolate(sd, linear)
        eval = vec_bdm1.eval_at_cell_centers(sd) @ interp
        eval = np.reshape(eval, (6, -1))

        known = np.array([linear(x).ravel() for x in sd.cell_centers.T]).T

        self.assertAlmostEqual(np.linalg.norm(eval - known), 0)

    def test_range(self):
        key = "vecbdm1"
        vec_bdm1 = pg.VecBDM1(key)
        self.assertTrue(vec_bdm1.get_range_discr_class(2) is pg.VecPwConstants)

    def test_assemble_asym_matrix_2d(self):
        N = 1
        sd = pp.StructuredTriangleGrid([N] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vecbdm1"
        vec_bdm1 = pg.VecBDM1(key)

        fun = lambda _: np.array([[1, 2, 0], [4, 3, 0]])
        u = vec_bdm1.interpolate(sd, fun)
        asym = vec_bdm1.assemble_asym_matrix(sd)

        p0 = pg.PwConstants("p0")
        cell_asym_u = p0.eval_at_cell_centers(sd) @ (asym @ u)

        self.assertTrue(np.allclose(cell_asym_u, -2))

    def test_trace_3d(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_bdm1 = pg.VecBDM1("vec_bdm1")

        B = vec_bdm1.assemble_trace_matrix(sd)

        fun = lambda x: np.array(
            [[x[0], x[1], x[2]], [x[0], x[1], x[2]], [0, 0, -x[0] - x[1]]]
        )
        u = vec_bdm1.interpolate(sd, fun)

        trace = B @ u

        self.assertTrue(np.allclose(trace, 0))

    def test_ndof_3d(self):
        sd = pp.StructuredTetrahedralGrid([1] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        vec_bdm1 = pg.VecBDM1("vec_bdm1")

        self.assertEqual(vec_bdm1.ndof(sd), 162)

    def test_assemble_mass_matrix_3d(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vec_bdm1"
        vec_bdm1 = pg.VecBDM1(key)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
        M = vec_bdm1.assemble_mass_matrix(sd, data)

        fun = lambda _: np.array([[1, 2, 0], [4, 3, 0], [0, 1, 1]])
        u = vec_bdm1.interpolate(sd, fun)

        self.assertAlmostEqual(u.T @ M @ u, 27)

        data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
        M = vec_bdm1.assemble_mass_matrix(sd, data)
        self.assertAlmostEqual(u.T @ M @ u, 32)

    def test_eval_at_cell_centers_3d(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vec_bdm1"
        vec_bdm1 = pg.VecBDM1(key)

        def linear(x):
            return np.array([x, 2 * x, -x])

        interp = vec_bdm1.interpolate(sd, linear)
        eval = vec_bdm1.eval_at_cell_centers(sd) @ interp
        eval = np.reshape(eval, (9, -1))

        known = np.array([linear(x).ravel() for x in sd.cell_centers.T]).T
        self.assertAlmostEqual(np.linalg.norm(eval - known), 0)

    def test_assemble_asym_matrix_3d(self):
        N = 1
        sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        key = "vec_bdm1"
        vec_bdm1 = pg.VecBDM1(key)

        fun = lambda _: np.array([[1, 2, -1], [4, 3, 2], [1, 1, 1]])
        u = vec_bdm1.interpolate(sd, fun)
        asym = vec_bdm1.assemble_asym_matrix(sd)

        p0 = pg.VecPwConstants("p0")
        cell_asym_u = p0.eval_at_cell_centers(sd) @ (asym @ u)
        cell_asym_u = cell_asym_u.reshape((3, -1))

        self.assertTrue(np.allclose(cell_asym_u[0], -1))
        self.assertTrue(np.allclose(cell_asym_u[1], -2))
        self.assertTrue(np.allclose(cell_asym_u[2], 2))


if __name__ == "__main__":
    unittest.main()
