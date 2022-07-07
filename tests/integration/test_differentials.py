import unittest
import numpy as np
import porepy as pp
import pygeon as pg

""" 
Module contains a unit tests to validate the differential operators.
"""


class DifferentialsTest(unittest.TestCase):
    def test_cochain_CartGrids(self):
        N = 3
        grids = [pp.CartGrid([N] * n, [1] * n) for n in [1, 2, 3]]

        for grid in grids:
            self.run_grid_test(grid)

    def test_cochain_SimplicialGrids(self):
        N = 3
        grids = [
            pp.StructuredTriangleGrid([N] * 2, [1] * 2),
            pp.StructuredTetrahedralGrid([N] * 3, [1] * 3),
        ]

        for grid in grids:
            self.run_grid_test(grid)

    def test_cochain_MD_Grid_2d(self):
        p = np.array([[0.0, 1.0, 0.5, 0.5], [0.5, 0.5, 0.0, 1.0]])
        e = np.array([[0, 2], [1, 3]])

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(p, e, domain)
        mesh_kwargs = {"mesh_size_frac": 1, "mesh_size_min": 1}

        mdg = network.mesh(mesh_kwargs)

        self.run_grid_test(mdg)

    def run_grid_test(self, grid):
        pg.convert_from_pp(grid)
        grid.compute_geometry()

        for n_minus_k in [1, 2]:
            diff1 = pg.numerics.differentials.exterior_derivative(grid, n_minus_k)
            diff2 = pg.numerics.differentials.exterior_derivative(grid, n_minus_k + 1)

            product = diff1 * diff2
            self.assertTrue(product.nnz == 0)


if __name__ == "__main__":
    unittest.main()
