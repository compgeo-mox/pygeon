""" Module contains a dummy unit test that always passes.
"""
import unittest
import numpy as np
import porepy as pp
import scipy.sparse as sps

import pygeon as pg


class SweeperUnitTest(unittest.TestCase):
    def check_flux(self, mdg):
        """
        Check whether the constructed flux balances the given mass-source
        """
        swp = pg.Sweeper(mdg)
        f = np.arange(swp.expand.shape[1])
        q_f = swp.sweep(f)

        self.assertTrue(np.allclose(pg.cell_mass(mdg) @ pg.div(mdg) @ q_f, f))

    def check_pressure(self, mdg):
        """
        Check whether the post-processing of the pressure is correct
        """
        div = pg.cell_mass(mdg) @ pg.div(mdg)
        face_mass = pg.face_mass(mdg)
        system = sps.bmat([[face_mass, -div.T], [div, None]], "csc")

        f = np.ones(div.shape[0])
        rhs = np.hstack([np.zeros(div.shape[1]), f])

        x = sps.linalg.spsolve(system, rhs)
        q = x[: div.shape[1]]
        p = x[div.shape[1] :]

        swp = pg.Sweeper(mdg)
        p_swp = swp.sweep_transpose(face_mass @ q)

        self.assertTrue(np.allclose(p, p_swp))

    def test_cart_grid(self):
        N = 3
        for dim in np.arange(1, 4):
            sd = pp.CartGrid([N] * dim, [1] * dim)
            mdg = pg.as_mdg(sd)
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()
            self.check_flux(mdg)

    def test_structured_triangle(self):
        N, dim = 3, 2
        sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()
        self.check_flux(mdg)
        self.check_pressure(mdg)

    def test_unstructured_triangle(self):
        sd = pg.unit_grid(2, 0.25, as_mdg=False)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()
        self.check_flux(mdg)
        self.check_pressure(mdg)

    def test_structured_tetra(self):
        N, dim = 3, 3
        sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
        mdg = pg.as_mdg(sd)
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()
        self.check_flux(mdg)
        self.check_pressure(mdg)

    def test_2d_mdg(self):
        grids = [
            # pp.md_grids_2d.single_horizontal,
            pp.md_grids_2d.single_vertical,
            # pp.md_grids_2d.two_intersecting,
        ]

        for g in grids:
            mdg, _ = g()
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()
            self.check_flux(mdg)
            self.check_pressure(mdg)

    def test_3d_mdg(self):
        mdg, _ = pp.md_grids_3d.single_horizontal()
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()
        self.check_flux(mdg)
        self.check_pressure(mdg)


if __name__ == "__main__":
    unittest.main()
