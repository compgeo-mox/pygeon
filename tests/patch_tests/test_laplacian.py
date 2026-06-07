"""Module contains tests to validate the Laplacian computation."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.Lagrange1,
        pg.Lagrange2,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_laplacian_constant(discr, unit_sd):
    A = discr.assemble_stiff_matrix(unit_sd)
    sol_func = lambda _: 1.0

    true_sol = discr.interpolate(unit_sd, sol_func)
    res = A @ true_sol

    assert np.allclose(res, 0)


def test_laplacian_dirichlet_bcs(unit_sd):
    discr = pg.Lagrange2("test")
    A = discr.assemble_stiff_matrix(unit_sd, None)

    source_func = lambda _: 1.0
    sol_func = lambda x: np.sum(x * (1 - x)) / (2 * unit_sd.dim)

    true_sol = discr.interpolate(unit_sd, sol_func)
    f = discr.source_term(unit_sd, source_func)

    if unit_sd.dim == 1:
        bdry_edges = np.zeros(unit_sd.num_cells, dtype=bool)
    elif unit_sd.dim == 2:
        bdry_edges = unit_sd.tags["domain_boundary_faces"]
    elif unit_sd.dim == 3:
        bdry_edges = unit_sd.tags["domain_boundary_ridges"]
    ess_bc = np.hstack((unit_sd.tags["domain_boundary_nodes"], bdry_edges), dtype=bool)

    ess_vals = np.zeros_like(ess_bc, dtype=float)
    ess_vals[ess_bc] = true_sol[ess_bc]

    LS = pg.LinearSystem(A, f)
    LS.flag_ess_bc(ess_bc, ess_vals)

    u = LS.solve()

    assert np.allclose(u, true_sol)


def test_laplacian_mixed_bcs(unit_sd):
    discr = pg.Lagrange2()
    A = discr.assemble_stiff_matrix(unit_sd, None)

    source_func = lambda _: 1.0
    sol_func = lambda x: np.sum(x * (1 - x)) / (2 * unit_sd.dim)
    flux_func = lambda _: -1 / (2 * unit_sd.dim)

    true_sol = discr.interpolate(unit_sd, sol_func)
    f = discr.source_term(unit_sd, source_func)

    bdry_nodes = unit_sd.nodes[0, :] <= 1e-6
    if unit_sd.dim == 1:
        bdry_edges = np.zeros(unit_sd.num_cells, dtype=bool)
    elif unit_sd.dim == 2:
        bdry_edges = bdry_nodes @ abs(unit_sd.face_ridges) > 1
    elif unit_sd.dim == 3:
        bdry_edges = bdry_nodes @ abs(unit_sd.ridge_peaks) > 1
    ess_bc = np.hstack((bdry_nodes, bdry_edges), dtype=bool)

    ess_vals = np.zeros_like(ess_bc, dtype=float)
    ess_vals[ess_bc] = true_sol[ess_bc]

    ess_bdry_faces = unit_sd.face_centers[0, :] <= 1e-6
    b_faces = np.logical_xor(unit_sd.tags["domain_boundary_faces"], ess_bdry_faces)
    b = discr.assemble_nat_bc(unit_sd, flux_func, b_faces)

    LS = pg.LinearSystem(A, f + b)
    LS.flag_ess_bc(ess_bc, ess_vals)

    u = LS.solve()

    assert np.allclose(u, true_sol)
