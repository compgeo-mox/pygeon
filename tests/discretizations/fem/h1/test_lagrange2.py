"""Module contains a unit test for the Lagrangean P2 discretization."""

import numpy as np
import porepy as pp
import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.Lagrange2("test")


def test_ndof(discr, unit_sd):
    match unit_sd.dim:
        case 1:
            num_edges = unit_sd.num_cells
        case 2:
            num_edges = unit_sd.num_faces
        case 3:
            num_edges = unit_sd.num_ridges

    assert discr.ndof(unit_sd) == unit_sd.num_nodes + num_edges


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [4, -1, 2],
                        [-1, 4, 2],
                        [2, 2, 16],
                    ]
                )
                / 30
            )

        case 2:
            M_known = (
                np.array(
                    [
                        [6, -1, -1, 0, 0, -4],
                        [-1, 6, -1, 0, -4, 0],
                        [-1, -1, 6, -4, 0, 0],
                        [0, 0, -4, 32, 16, 16],
                        [0, -4, 0, 16, 32, 16],
                        [-4, 0, 0, 16, 16, 32],
                    ]
                )
                / 360
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [3.0, 0.5, 0.5, 0.5, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0],
                        [0.5, 3.0, 0.5, 0.5, -2.0, -3.0, -3.0, -2.0, -2.0, -3.0],
                        [0.5, 0.5, 3.0, 0.5, -3.0, -2.0, -3.0, -2.0, -3.0, -2.0],
                        [0.5, 0.5, 0.5, 3.0, -3.0, -3.0, -2.0, -3.0, -2.0, -2.0],
                        [-2.0, -2.0, -3.0, -3.0, 16.0, 8.0, 8.0, 8.0, 8.0, 4.0],
                        [-2.0, -3.0, -2.0, -3.0, 8.0, 16.0, 8.0, 8.0, 4.0, 8.0],
                        [-2.0, -3.0, -3.0, -2.0, 8.0, 8.0, 16.0, 4.0, 8.0, 8.0],
                        [-3.0, -2.0, -2.0, -3.0, 8.0, 8.0, 4.0, 16.0, 8.0, 8.0],
                        [-3.0, -2.0, -3.0, -2.0, 8.0, 4.0, 8.0, 8.0, 16.0, 8.0],
                        [-3.0, -3.0, -2.0, -2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 16.0],
                    ]
                )
                / 1260
            )

    assert np.allclose(M.todense(), M_known)


def test_interpolate_and_evaluate(discr: pg.Discretization, unit_sd: pg.Grid):
    func = lambda x: x[0] ** 2
    known_vals = func(unit_sd.cell_centers)

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals)

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
            bdry_edges = bdry_nodes @ abs(sd.face_ridges) > 1
        elif sd.dim == 3:
            bdry_edges = bdry_nodes @ abs(sd.ridge_peaks) > 1
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

    def test_diff_matrix(self):
        dim = 2
        sd = pg.reference_element(dim)
        sd.compute_geometry()

        discr = pg.Lagrange2()
        D = discr.assemble_diff_matrix(sd)

        D_known = np.array(
            [
                [0.0, -3.0, -1.0, 4.0, 0.0, 0.0],
                [-3.0, 0.0, -1.0, 0.0, 4.0, 0.0],
                [-3.0, -1.0, 0.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 3.0, -4.0, 0.0, 0.0],
                [1.0, 0.0, 3.0, 0.0, -4.0, 0.0],
                [1.0, 3.0, 0.0, 0.0, 0.0, -4.0],
            ]
        )

        self.assertTrue(np.allclose(D.todense(), D_known))
