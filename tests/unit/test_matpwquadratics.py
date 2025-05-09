import unittest
import numpy as np

import pygeon as pg


class MatPwQuadraticsTest(unittest.TestCase):
    def test_ndof(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.MatPwQuadratics()
        self.assertTrue(discr.ndof(sd) == 24)

    def test_trace_2d(self):
        dim = 2
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.MatPwQuadratics()
        trace = discr.assemble_trace_matrix(sd)

        func = lambda x: np.array([[x[0], x[1]], [x[1], x[0] * x[1]]])
        func_trace = lambda x: x[0] + x[0] * x[1]

        func_interp = discr.interpolate(sd, func)
        trace_interp = pg.PwQuadratics().interpolate(sd, func_trace)

        self.assertTrue(np.allclose(trace @ func_interp, trace_interp))

    def test_asym_2d(self):
        dim = 2
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.MatPwQuadratics()
        asym = discr.assemble_asym_matrix(sd)

        func = lambda x: np.array([[x[0], x[1]], [x[0] * x[1], x[1]]])
        func_asym = lambda x: x[0] * x[1] - x[1]

        func_interp = discr.interpolate(sd, func)
        asym_interp = pg.PwQuadratics().interpolate(sd, func_asym)

        self.assertTrue(np.allclose(asym @ func_interp, asym_interp))

    def test_trace_3d(self):
        dim = 3
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.MatPwQuadratics()
        trace = discr.assemble_trace_matrix(sd)

        func = lambda x: np.array(
            [
                [x[2] * x[0], x[1], x[2]],
                [x[0] * x[1], x[1], x[0]],
                [x[0] * x[2], x[1], x[0]],
            ]
        )
        func_trace = lambda x: x[2] * x[0] + x[1] + x[0]

        func_interp = discr.interpolate(sd, func)
        trace_interp = pg.PwQuadratics().interpolate(sd, func_trace)

        self.assertTrue(np.allclose(trace @ func_interp, trace_interp))

    def test_asym_3d(self):
        dim = 3
        sd = pg.unit_grid(dim, 1.0, as_mdg=False, structured=True)
        sd.compute_geometry()

        discr = pg.MatPwQuadratics()
        asym = discr.assemble_asym_matrix(sd)

        func = lambda x: np.array(
            [
                [x[2] * x[0], x[1], x[2]],
                [x[0] * x[1], x[1], x[0]],
                [x[0] * x[2], x[1], x[0]],
            ]
        )
        func_asym = lambda x: np.array(
            [
                x[1] - x[0],
                x[2] - x[0] * x[2],
                x[0] * x[1] - x[1],
            ]
        )

        func_interp = discr.interpolate(sd, func)
        asym_interp = pg.VecPwQuadratics().interpolate(sd, func_asym)

        self.assertTrue(np.allclose(asym @ func_interp, asym_interp))


if __name__ == "__main__":
    unittest.main()
