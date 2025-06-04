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

    def test_trace_with_proj(self):
        for dim in [2, 3]:
            sd = pg.reference_element(dim)
            sd.compute_geometry()

            rt1 = pg.VecRT1()
            proj = rt1.proj_to_MatPwPolynomials(sd)
            trace_bdm = rt1.assemble_trace_matrix(sd)

            discr = pg.MatPwQuadratics()
            trace = discr.assemble_trace_matrix(sd)

            check = trace_bdm - trace @ proj
            self.assertTrue(np.allclose(check.data, 0))

    def test_proj_topwquadratics(self):
        for dim in [2, 3]:
            sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
            sd.compute_geometry()

            key = "test"
            disc = pg.VecRT1(key)
            data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
            M_RT = disc.assemble_mass_matrix(sd, data)
            P = disc.proj_to_MatPwPolynomials(sd)

            quadratics = pg.MatPwQuadratics()
            M_quad = quadratics.assemble_mass_matrix(sd)

            check = M_RT - P.T @ M_quad @ P

            self.assertTrue(np.allclose(check.data, 0))

    def test_asym_2d(self):
        dim = 2
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecRT1()
        asym = discr.assemble_asym_matrix(sd)

        func = lambda x: np.array([[x[0], x[1], 0], [x[0], x[1], 0]])
        func_asym = lambda x: x[0] - x[1]

        func_interp = discr.interpolate(sd, func)
        asym_interp = pg.PwQuadratics().interpolate(sd, func_asym)

        self.assertTrue(np.allclose(asym @ func_interp, asym_interp))

    def test_asym_3d(self):
        dim = 3
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.VecRT1()
        asym = discr.assemble_asym_matrix(sd)

        func = lambda x: np.array(
            [
                [x[0], x[1], x[2]],
                [x[0], x[1], x[2]],
                [x[0], x[1], x[2]],
            ]
        )
        func_asym = lambda x: np.array(
            [
                x[1] - x[2],
                x[2] - x[0],
                x[0] - x[1],
            ]
        )

        func_interp = discr.interpolate(sd, func)
        asym_interp = pg.VecPwQuadratics().interpolate(sd, func_asym)

        self.assertTrue(np.allclose(asym @ func_interp, asym_interp))

    def test_proj_with_interp(self):
        for dim in [2, 3]:
            sd = pg.reference_element(dim)
            sd.compute_geometry()

            discr = pg.VecRT1()

            M_lumped = discr.assemble_lumped_matrix(sd)

            func = lambda x: np.array([x, x, x])
            func_interp = discr.interpolate(sd, func)

            norm_L = func_interp @ M_lumped @ func_interp

            quads = pg.MatPwQuadratics()
            P = discr.proj_to_MatPwPolynomials(sd)
            M_q = quads.assemble_lumped_matrix(sd)
            M_lumped2 = P.T @ M_q @ P

            norm_L2 = func_interp @ M_lumped2 @ func_interp

            self.assertTrue(np.isclose(norm_L, norm_L2))

            self.assertTrue(np.allclose((M_lumped2 - M_lumped).data, 0))

    def test_lumped_inv(self):
        max_nnz = [0, 0, 52, 333]
        for dim in [2, 3]:
            sd = pg.reference_element(dim)
            sd.compute_geometry()

            key = "test"
            data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 1.0, "mu_c": 1.0}}}
            discr = pg.VecRT1(key)

            # check for data and without data, so we use default parameters
            for d in [data, None]:
                L = discr.assemble_lumped_matrix_cosserat(sd, d)
                L_inv = pg.assemble_inverse(L)

                L_inv.data[np.abs(L_inv.data) < 1e-10] = 0
                L_inv.eliminate_zeros()

                self.assertTrue(L_inv.nnz <= max_nnz[dim])


if __name__ == "__main__":
    unittest.main()
