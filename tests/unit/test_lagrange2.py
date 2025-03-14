"""Module contains a unit test for the Lagrangean P2 discretization."""

import unittest
import numpy as np

import porepy as pp
import pygeon as pg


class Lagrange2Test(unittest.TestCase):
    def test_eval_1d(self):
        """Test the interpolation and evaluation of a quadratic in 1D"""
        sd = pp.CartGrid([10])
        pg.convert_from_pp(sd)
        func = lambda x: x[0] * (1 - x[0])
        self.interpolate_and_evaluate(sd, func)

    def test_eval_2d(self):
        """Test the interpolation and evaluation of a quadratic in 2D"""
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        func = lambda x: x[1] * (1 + x[0] + x[1])
        self.interpolate_and_evaluate(sd, func)

    def test_eval_3d(self):
        """Test the interpolation and evaluation of a quadratic in 3D"""
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        func = lambda x: np.sum(x * (1 - x))
        self.interpolate_and_evaluate(sd, func)

    def interpolate_and_evaluate(self, sd, func):
        """For a given polynomial in the space, interpolate onto the discrete space,
        and evaluate at cell centers"""

        sd.compute_geometry()
        discr = pg.Lagrange2()

        interp_func = discr.interpolate(sd, func)
        P = discr.eval_at_cell_centers(sd)

        evaluated = P @ interp_func
        known_func = np.array([func(x) for x in sd.cell_centers.T])

        self.assertTrue(np.allclose(evaluated, known_func))

    def test_laplacian_1d(self):
        """Solve a Laplace problem with known, quadratic solution in 1D"""
        sd = pp.CartGrid([10], 1)
        pg.convert_from_pp(sd)
        self.solve_laplacian(sd)

    def test_laplacian_2d(self):
        """Solve a Laplace problem with known, quadratic solution in 2D"""
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        self.solve_laplacian(sd)

    def test_laplacian_3d(self):
        """Solve a Laplace problem with known, quadratic solution in 3D"""
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        self.solve_laplacian(sd)

    def solve_laplacian(self, sd):
        sd.compute_geometry()
        discr = pg.Lagrange2()
        A = discr.assemble_stiff_matrix(sd, None)

        source_func = lambda _: 1.0
        sol_func = lambda x: np.sum(x * (1 - x)) / (2 * sd.dim)

        true_sol = discr.interpolate(sd, sol_func)
        f = discr.source_term(sd, source_func)

        if sd.dim == 1:
            bdry_edges = np.zeros(sd.num_cells, dtype=bool)
        elif sd.dim == 2:
            bdry_edges = sd.tags["domain_boundary_faces"]
        elif sd.dim == 3:
            bdry_edges = sd.tags["domain_boundary_ridges"]
        ess_bc = np.hstack((sd.tags["domain_boundary_nodes"], bdry_edges), dtype=bool)

        ess_vals = np.zeros_like(ess_bc, dtype=float)
        ess_vals[ess_bc] = true_sol[ess_bc]

        LS = pg.LinearSystem(A, f)
        LS.flag_ess_bc(ess_bc, ess_vals)

        u = LS.solve()

        self.assertTrue(np.allclose(u, true_sol))

    def check_natural_bc(self, sd):
        pg.convert_from_pp(sd)
        sd.compute_geometry()

        func = lambda x: x[0]
        disc = pg.Lagrange2()

        b_faces = sd.face_centers[sd.dim - 1] <= 1e-5

        return disc.assemble_nat_bc(sd, func, b_faces)

    def test_natural_bc_2D(self):
        sd = pp.StructuredTriangleGrid([1, 1])
        b = self.check_natural_bc(sd)

        known_b = np.zeros_like(b)
        known_b[[1, 4]] = np.array([1, 2]) / 6

        self.assertTrue(np.allclose(b, known_b))

    def test_natural_bc_3D(self):
        sd = pp.StructuredTetrahedralGrid([1, 1, 1])
        b = self.check_natural_bc(sd)

        known_b = np.zeros_like(b)
        known_b[[0, 1, 2, 3, 8, 9, 11, 12, 16]] = [-1, 3, -3, 1, 8, 4, 20, 16, 12]
        known_b /= 120

        self.assertTrue(np.allclose(b, known_b))

    def test_mixed_bcs_1d(self):
        """Solve a Laplace problem with (partial) Neumann bcs in 2D"""
        sd = pp.CartGrid([10], 1)
        pg.convert_from_pp(sd)
        self.solve_mixed_bcs(sd)

    def test_mixed_bcs_2d(self):
        """Solve a Laplace problem with (partial) Neumann bcs in 2D"""
        sd = pg.unit_grid(2, 0.5, as_mdg=False)
        self.solve_mixed_bcs(sd)

    def test_mixed_bcs_3d(self):
        """Solve a Laplace problem with (partial) Neumann bcs in 3D"""
        sd = pg.unit_grid(3, 0.5, as_mdg=False)
        self.solve_mixed_bcs(sd)

    def solve_mixed_bcs(self, sd):
        sd.compute_geometry()
        discr = pg.Lagrange2()
        A = discr.assemble_stiff_matrix(sd, None)

        source_func = lambda _: 1.0
        sol_func = lambda x: np.sum(x * (1 - x)) / (2 * sd.dim)
        flux_func = lambda _: -1 / (2 * sd.dim)

        true_sol = discr.interpolate(sd, sol_func)
        f = discr.source_term(sd, source_func)

        bdry_nodes = sd.nodes[0, :] <= 1e-6
        if sd.dim == 1:
            bdry_edges = np.zeros(sd.num_cells, dtype=bool)
        elif sd.dim == 2:
            bdry_edges = bdry_nodes @ np.abs(sd.face_ridges) > 1
        elif sd.dim == 3:
            bdry_edges = bdry_nodes @ np.abs(sd.ridge_peaks) > 1
        ess_bc = np.hstack((bdry_nodes, bdry_edges), dtype=bool)

        ess_vals = np.zeros_like(ess_bc, dtype=float)
        ess_vals[ess_bc] = true_sol[ess_bc]

        ess_bdry_faces = sd.face_centers[0, :] <= 1e-6
        b_faces = np.logical_xor(sd.tags["domain_boundary_faces"], ess_bdry_faces)
        b = discr.assemble_nat_bc(sd, flux_func, b_faces)

        LS = pg.LinearSystem(A, f + b)
        LS.flag_ess_bc(ess_bc, ess_vals)

        u = LS.solve()

        self.assertTrue(np.allclose(u, true_sol))


if __name__ == "__main__":
    unittest.main()
