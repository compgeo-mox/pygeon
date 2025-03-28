"""Module contains a unit test for the error computation."""

import unittest

import numpy as np
import porepy as pp

import pygeon as pg  # type: ignore[import-untyped]


class ErrorTest(unittest.TestCase):
    def test_0(self):
        sd = pp.CartGrid(2 * [3])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        def fun(pt):
            return pt[0] + 2 * pt[1]

        int_sol = np.array([1.5, 2.5, 3.5, 3.5, 4.5, 5.5, 5.5, 6.5, 7.5])

        discr = pg.PwConstants()

        err = discr.error_l2(sd, np.zeros_like(int_sol), fun)
        self.assertTrue(np.isclose(err, 1))

        err = discr.error_l2(sd, int_sol, fun, etype="standard")
        self.assertTrue(np.isclose(err, 0))

        err = discr.error_l2(sd, int_sol, fun)
        self.assertTrue(np.isclose(err, 0.22435590134827893))

    def test_1(self):
        sd = pp.StructuredTriangleGrid(2 * [3])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        def fun(pt):
            return np.array([pt[0] + 2 * pt[1], 2 * pt[0] + pt[1], 0])

        int_sol = np.array(
            [
                -1.0,
                1.0,
                0.0,
                -3.0,
                2.0,
                -1.0,
                -5.0,
                3.0,
                -2.0,
                4.0,
                -2.0,
                3.0,
                1.0,
                -4.0,
                4.0,
                0.0,
                -6.0,
                5.0,
                -1.0,
                6.0,
                -3.0,
                5.0,
                2.0,
                -5.0,
                6.0,
                1.0,
                -7.0,
                7.0,
                0.0,
                8.0,
                -4.0,
                -6.0,
                -8.0,
            ]
        )

        discr = pg.RT0()

        err = discr.error_l2(sd, np.zeros_like(int_sol), fun)
        self.assertTrue(np.isclose(err, 1))

        err = discr.error_l2(sd, int_sol, fun, etype="standard")
        self.assertTrue(np.isclose(err, 0))

        err = discr.error_l2(sd, int_sol, fun)
        self.assertTrue(np.isclose(err, 0.06859943405700351))

    def test_2(self):
        sd = pp.StructuredTetrahedralGrid(3 * [3])
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        def fun(pt):
            return np.array([pt[0] + 2 * pt[1] - pt[2], 2 * pt[0] - pt[1], 6 * pt[2]])

        # fmt: off
        int_sol = \
        np.array([ 0.5, -0.5,  3. ,  1.5, -1. ,  1.5,  3. ,  3. ,  2.5,  2.5,  0. ,
                   3.5,  2. ,  3. ,  3.5,  1. ,  5.5,  1. ,  3. ,  4.5,  2.5, -1.5,
                   3.5,  3. ,  3.5, -4. ,  0.5,  1.5,  1. ,  3. , -0.5,  4.5, -3. ,
                   2.5, -0.5,  0. ,  3. ,  0.5, -2. ,  4.5, -2.5, -1. ,  3. ,  1.5,
                   4.5, -2.5,  4.5,  3. ,  5.5, -7. , -0.5,  2.5, -1. ,  3. , -3.5,
                   6.5, -6. ,  1.5,  0.5, -2. ,  3. , -2.5, -5. ,  3.5, -1.5, -3. ,
                   3. , -1.5,  6.5,  5.5,  3. ,  7.5,  3.5, -3. ,  3. ,  8.5,  1.5,
                  -4. ,  3. , -0.5, -5. ,  3. , -0.5, -0.5,  9. ,  0.5,  0. ,  1.5,
                  10. ,  9. ,  9.5,  1.5,  1. ,  3.5,  9. ,  9. , 10.5,  2. ,  5.5,
                   8. ,  9. , 11.5,  1.5, -1.5,  9.5,  9. ,  2.5, -3. ,  0.5,  7.5,
                   8. ,  9. ,  6.5,  3.5, -2. ,  2.5,  5.5,  7. ,  9. ,  7.5, -1. ,
                   4.5,  3.5,  6. ,  9. ,  8.5,  3.5, -2.5, 10.5,  9. ,  4.5, -6. ,
                  -0.5,  8.5,  6. ,  9. ,  3.5,  5.5, -5. ,  1.5,  6.5,  5. ,  9. ,
                   4.5, -4. ,  3.5,  4.5,  4. ,  9. ,  5.5,  5.5, 11.5,  9. ,  6.5,
                   9.5,  4. ,  9. ,  7.5,  7.5,  3. ,  9. ,  5.5,  2. ,  9. , -1.5,
                  -0.5, 15. , -0.5,  1. ,  1.5, 17. , 15. , 16.5,  0.5,  2. ,  3.5,
                  16. , 15. , 17.5,  3. ,  5.5, 15. , 15. , 18.5,  0.5, -1.5, 15.5,
                  15. ,  1.5, -2. ,  0.5, 13.5, 15. , 15. , 13.5,  2.5, -1. ,  2.5,
                  11.5, 14. , 15. , 14.5,  0. ,  4.5,  9.5, 13. , 15. , 15.5,  2.5,
                  -2.5, 16.5, 15. ,  3.5, -5. , -0.5, 14.5, 13. , 15. , 10.5,  4.5,
                  -4. ,  1.5, 12.5, 12. , 15. , 11.5, -3. ,  3.5, 10.5, 11. , 15. ,
                  12.5,  4.5, 17.5, 15. ,  5.5, 15.5, 11. , 15. ,  6.5, 13.5, 10. ,
                  15. , 11.5,  9. , 15. , -2.5, -0.5, -1.5,  2. ,  1.5, -0.5,  3. ,
                   3.5,  4. ,  5.5, -0.5, -1.5,  0.5, -1. ,  0.5,  1.5,  0. ,  2.5,
                   1. ,  4.5,  1.5, -2.5,  2.5, -4. , -0.5,  3.5, -3. ,  1.5, -2. ,
                   3.5,  3.5,  4.5,  5.5])
        # fmt: on

        discr = pg.Nedelec0()

        err = discr.error_l2(sd, np.zeros_like(int_sol), fun)
        self.assertTrue(np.isclose(err, 1))

        err = discr.error_l2(sd, int_sol, fun)
        self.assertTrue(np.isclose(err, 0))


if __name__ == "__main__":
    unittest.main()
