import unittest

import numpy as np
import porepy as pp

import pygeon as pg

""" 
Module contains a unit tests to validate the inner products.
"""


class InnerProductsTest(unittest.TestCase):
    def test_cell_mass_cart(self):
        N = 3
        grids = [pp.CartGrid([N] * n, [1] * n) for n in [1, 2, 3]]

        for sd in grids:
            mdg = pp.meshing.subdomains_to_mdg([[sd]])
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()

            cell_mass = pg.cell_mass(mdg)
            self.assertTrue(np.allclose(cell_mass.data, float(N) ** sd.dim))

    def test_cell_mass_simplices(self):
        N = 3
        grids = [
            pp.StructuredTriangleGrid([N] * 2, [1] * 2),
            pp.StructuredTetrahedralGrid([N] * 3, [1] * 3),
        ]

        for sd in grids:
            mdg = pp.meshing.subdomains_to_mdg([[sd]])
            pg.convert_from_pp(mdg)
            mdg.compute_geometry()

            cell_mass = pg.cell_mass(mdg)
            self.assertTrue(
                np.allclose(
                    cell_mass.data, (float(N) ** sd.dim) * sd.dim * (sd.dim - 1)
                )
            )

    def test_symmetry(self):
        sd = pp.StructuredTetrahedralGrid([3] * 3, [1] * 3)
        mdg = pp.meshing.subdomains_to_mdg([[sd]])
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for n_minus_k in range(3):
            mass = pg.numerics.innerproducts.mass_matrix(mdg, n_minus_k, None)
            self.assertTrue(np.allclose((mass - mass.T).data, 0))

    def test_dimensions(self):
        sd = pp.StructuredTetrahedralGrid([3] * 3, [1] * 3)
        mdg = pp.meshing.subdomains_to_mdg([[sd]])
        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        for n_minus_k in range(1, 3):
            mass = pg.numerics.innerproducts.mass_matrix(mdg, n_minus_k, None)
            stiff = pg.numerics.stiffness.stiff_matrix(mdg, n_minus_k, None)
            self.assertTrue(np.allclose(mass.shape, stiff.shape))


if __name__ == "__main__":
    unittest.main()
