""" Module contains a unit test for the Lagrangean P2 discretization.
"""

import unittest
import numpy as np
import scipy.sparse as sps

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
        sd = pp.CartGrid([10])
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


if __name__ == "__main__":
    unittest.main()
