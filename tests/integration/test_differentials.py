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

        frac1 = pp.LineFracture(p[:, e[0]])
        frac2 = pp.LineFracture(p[:, e[1]])

        fracs = [frac1, frac2]

        bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        domain = pp.Domain(bounding_box=bbox)
        network = pp.FractureNetwork2d(fracs, domain)
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

    def test_stiffness_P1_3D(self):
        """
        Test whether the stiffness matrix of P1 corresponds to
        grad.T M grad where M is the mass matrix of Ne0.
        """

        la = pg.Lagrange1("flow")
        ne = pg.Nedelec0("flow")

        grid = pp.StructuredTetrahedralGrid([4] * 3, [1] * 3)
        pg.convert_from_pp(grid)
        grid.compute_geometry()

        data = {pp.PARAMETERS: {"flow": {}}}
        k = pp.SecondOrderTensor(np.ones(grid.num_cells))
        data[pp.PARAMETERS]["flow"]["second_order_tensor"] = k

        A_la = la.assemble_stiffness_matrix(grid, data)
        M_ne = ne.assemble_mass_matrix(grid, data)
        grad = pg.grad(grid)

        diff = A_la - grad.T * M_ne * grad
        self.assertAlmostEqual(np.linalg.norm(diff.data), 0)

    def test_stiffness_P1_2D(self):
        """
        Test whether the stiffness matrix of P1 in 2D corresponds to
        curl.T M curl where M is the mass matrix of RT0.
        """

        la = pg.Lagrange1("flow")
        rt = pp.RT0("flow")

        grid = pp.StructuredTriangleGrid([4] * 2, [1] * 2)
        pg.convert_from_pp(grid)
        grid.compute_geometry()

        data = {pp.PARAMETERS: {"flow": {}}, pp.DISCRETIZATION_MATRICES: {"flow": {}}}
        k = pp.SecondOrderTensor(np.ones(grid.num_cells))
        data[pp.PARAMETERS]["flow"]["second_order_tensor"] = k

        rt.discretize(grid, data)

        A_la = la.assemble_stiffness_matrix(grid, data)
        M_rt = data[pp.DISCRETIZATION_MATRICES]["flow"]["mass"]
        curl = pg.curl(grid)

        diff = A_la - curl.T * M_rt * curl
        self.assertAlmostEqual(np.linalg.norm(diff.data), 0)


if __name__ == "__main__":
    unittest.main()
