""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import porepy as pp
import pygeon as pg


class SweeperUnitTest(unittest.TestCase):
    def check(self, sd):
        swp = pg.Sweeper(sd)
        f = np.arange(sd.num_cells)
        q_f = swp.sweep(f)

        self.assertTrue(np.allclose(pg.div(sd) @ q_f, f))

    def check_mdg(self, mdg):
        swp = pg.Sweeper(mdg)
        f = np.arange(mdg.num_subdomain_cells())
        q_f = swp.sweep(f)

        self.assertTrue(np.allclose(pg.div(mdg) @ q_f, f))

    def test_cart_grid(self):
        N = 3
        for dim in np.arange(1, 4):
            sd = pp.CartGrid([N] * dim, [1] * dim)
            pg.convert_from_pp(sd)
            self.check(sd)

    def test_structured_triangle(self):
        N, dim = 3, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        self.check(sd)

    def test_structured_tetra(self):
        N, dim = 3, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        pg.convert_from_pp(sd)
        self.check(sd)

    def test_2d_mdg(self):
        grids = [pp.md_grids_2d.single_horizontal,
                 pp.md_grids_2d.single_vertical, pp.md_grids_2d.two_intersecting]

        for g in grids:
            mdg, _ = g()
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()
            self.check_mdg(mdg)

    def test_3d_mdg(self):
        mdg, _ = pp.md_grids_3d.single_horizontal()
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()
        self.check_mdg(mdg)


if __name__ == "__main__":
    unittest.main()
