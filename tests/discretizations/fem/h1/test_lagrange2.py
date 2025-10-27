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
                        [6, -1, -1, -4, 0, 0],
                        [-1, 6, -1, 0, -4, 0],
                        [-1, -1, 6, 0, 0, -4],
                        [-4, 0, 0, 32, 16, 16],
                        [0, -4, 0, 16, 32, 16],
                        [0, 0, -4, 16, 16, 32],
                    ]
                )
                / 360
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [6.0, 1.0, 1.0, 1.0, -6.0, -6.0, -4.0, -6.0, -4.0, -4.0],
                        [1.0, 6.0, 1.0, 1.0, -6.0, -4.0, -6.0, -4.0, -6.0, -4.0],
                        [1.0, 1.0, 6.0, 1.0, -4.0, -6.0, -6.0, -4.0, -4.0, -6.0],
                        [1.0, 1.0, 1.0, 6.0, -4.0, -4.0, -4.0, -6.0, -6.0, -6.0],
                        [-6.0, -6.0, -4.0, -4.0, 32.0, 16.0, 16.0, 16.0, 16.0, 8.0],
                        [-6.0, -4.0, -6.0, -4.0, 16.0, 32.0, 16.0, 16.0, 8.0, 16.0],
                        [-4.0, -6.0, -6.0, -4.0, 16.0, 16.0, 32.0, 8.0, 16.0, 16.0],
                        [-6.0, -4.0, -4.0, -6.0, 16.0, 16.0, 8.0, 32.0, 16.0, 16.0],
                        [-4.0, -6.0, -4.0, -6.0, 16.0, 8.0, 16.0, 16.0, 32.0, 16.0],
                        [-4.0, -4.0, -6.0, -6.0, 8.0, 16.0, 16.0, 16.0, 16.0, 32.0],
                    ]
                )
                / 2520
            )

    assert np.allclose(M.todense(), M_known)


def test_assemble_diff_matrix(discr, ref_sd):
    D = discr.assemble_diff_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            D_known = np.array(
                [
                    [-3.0, -1.0, 4.0],
                    [1.0, 3.0, -4.0],
                ]
            )

        case 2:
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
        case 3:
            D_known = np.array(
                [
                    [-3.0, -1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.0, 0.0, -1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, -3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, -3.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, -3.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                    [-1.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, -1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                ]
            )

    assert np.allclose(D.todense(), D_known)


def test_assemble_stiff_matrix(discr, ref_sd):
    M = discr.assemble_stiff_matrix(ref_sd)

    match ref_sd.dim:
        case 1:
            M_known = (
                np.array(
                    [
                        [14.0, 2.0, -16.0],
                        [2.0, 14.0, -16.0],
                        [-16.0, -16.0, 32.0],
                    ]
                )
                / 6
            )

        case 2:
            M_known = (
                np.array(
                    [
                        [6.0, 1.0, 1.0, 0.0, -4.0, -4.0],
                        [1.0, 3.0, 0.0, 0.0, 0.0, -4.0],
                        [1.0, 0.0, 3.0, 0.0, -4.0, 0.0],
                        [0.0, 0.0, 0.0, 16.0, -8.0, -8.0],
                        [-4.0, 0.0, -4.0, -8.0, 16.0, 0.0],
                        [-4.0, -4.0, 0.0, -8.0, 0.0, 16.0],
                    ]
                )
                / 6
            )
        case 3:
            M_known = (
                np.array(
                    [
                        [9.0, 1.0, 1.0, 1.0, 2.0, 2.0, -6.0, 2.0, -6.0, -6.0],
                        [1.0, 3.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, -4.0],
                        [1.0, 0.0, 3.0, 0.0, -1.0, 0.0, 1.0, -1.0, -4.0, 1.0],
                        [1.0, 0.0, 0.0, 3.0, -1.0, -1.0, -4.0, 0.0, 1.0, 1.0],
                        [2.0, 0.0, -1.0, -1.0, 16.0, 4.0, -8.0, 4.0, -8.0, -8.0],
                        [2.0, -1.0, 0.0, -1.0, 4.0, 16.0, -8.0, 4.0, -8.0, -8.0],
                        [-6.0, 1.0, 1.0, -4.0, -8.0, -8.0, 24.0, -8.0, 4.0, 4.0],
                        [2.0, -1.0, -1.0, 0.0, 4.0, 4.0, -8.0, 16.0, -8.0, -8.0],
                        [-6.0, 1.0, -4.0, 1.0, -8.0, -8.0, 4.0, -8.0, 24.0, 4.0],
                        [-6.0, -4.0, 1.0, 1.0, -8.0, -8.0, 4.0, -8.0, 4.0, 24.0],
                    ]
                )
                / 30
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
