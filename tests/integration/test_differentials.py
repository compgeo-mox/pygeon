import unittest

import numpy as np
import porepy as pp

import pygeon as pg  # type: ignore[import-untyped]

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
        network = pp.create_fracture_network(fracs, domain)
        mesh_kwargs = {"mesh_size_frac": 1, "mesh_size_min": 1}

        mdg = network.mesh(mesh_kwargs)

        self.run_grid_test(mdg)

    def run_grid_test(self, grid):
        pg.convert_from_pp(grid)
        grid.compute_geometry()

        for n_minus_k in [1, 2]:
            diff1 = pg.numerics.differentials.exterior_derivative(grid, n_minus_k)
            diff2 = pg.numerics.differentials.exterior_derivative(grid, n_minus_k + 1)

            product = diff1 @ diff2
            self.assertTrue(product.nnz == 0)

    def test_stiffness_P1_3D(self):
        """
        Test whether the stiffness matrix of P1 corresponds to
        grad.T M grad where M is the mass matrix of Ne0.
        """
        sd = pp.StructuredTetrahedralGrid([4] * 3, [1] * 3)
        pg.convert_from_pp(sd)
        disc = pg.Lagrange1()
        self.check_stiffness_consistency(sd, disc)

    def test_stiffness_P1_2D(self):
        """
        Test whether the stiffness matrix of P1 in 2D corresponds to
        curl.T M curl where M is the mass matrix of RT0.
        """

        sd = pp.StructuredTriangleGrid([4] * 2, [1] * 2)
        pg.convert_from_pp(sd)
        disc = pg.Lagrange1()
        self.check_stiffness_consistency(sd, disc)

    def test_stiffness_P2_1D(self):
        sd = pp.CartGrid([3], 1)
        pg.convert_from_pp(sd)
        disc = pg.Lagrange2()
        self.check_stiffness_consistency(sd, disc)

    def test_stiffness_P2_2D_structured(self):
        sd = pp.StructuredTriangleGrid([2, 2])
        pg.convert_from_pp(sd)
        disc = pg.Lagrange2()
        self.check_stiffness_consistency(sd, disc)

    def test_stiffness_P2_2D_unstructured(self):
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        disc = pg.Lagrange2()
        self.check_stiffness_consistency(sd, disc)

    def check_stiffness_consistency(self, sd, disc):
        """Compare the implemented stiffness matrix
        to the one obtained by mapping to the range discretization"""
        sd.compute_geometry()

        Stiff_1 = disc.assemble_stiff_matrix(sd, None)
        Stiff_2 = pg.Discretization.assemble_stiff_matrix(disc, sd)

        diff = Stiff_1 - Stiff_2
        self.assertTrue(np.allclose(diff.data, 0))

    def test_cochain_property_P1_2D(self):
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        disc = pg.Lagrange1()
        self.check_cochain_property(sd, disc)

    def test_cochain_property_P2_2D(self):
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        disc = pg.Lagrange2()
        self.check_cochain_property(sd, disc)

    def test_cochain_property_P1_3D(self):
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        disc = pg.Lagrange1()
        self.check_cochain_property(sd, disc)

    def test_cochain_property_P2_3D(self):
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        disc = pg.Lagrange2()
        self.check_cochain_property(sd, disc)

    def test_cochain_property_N0_3D(self):
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        disc = pg.Nedelec0()
        self.check_cochain_property(sd, disc)

    def test_cochain_property_N1_3D(self):
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        disc = pg.Nedelec1()
        self.check_cochain_property(sd, disc)

    def check_cochain_property(self, sd, disc):
        sd.compute_geometry()

        Diff = disc.assemble_diff_matrix(sd)
        range_discr = disc.get_range_discr_class(sd.dim)()
        range_Diff = range_discr.assemble_diff_matrix(sd)

        prod = range_Diff @ Diff
        self.assertTrue(np.allclose(prod.data, 0))


if __name__ == "__main__":
    unittest.main()
