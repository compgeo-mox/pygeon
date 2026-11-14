"""Module contains general tests for all H(curl) discretizations."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture(
    params=[
        pg.NedelecR0,
        pg.NedelecF1,
    ]
)
def discr(request: pytest.FixtureRequest) -> pg.Discretization:
    return request.param("test")


def test_assemble_nat_bc(discr, ref_sd_3d):
    with pytest.raises(NotImplementedError):
        discr.assemble_nat_bc(ref_sd_3d, lambda _: np.zeros(1), np.zeros(1))


def test_range_disc(discr):
    assert discr.get_range_discr_class(3) is pg.RT0


def test_interp_eval_constants(discr, unit_sd_3d):
    f = lambda _: np.array([2, 3, -1])
    f_known = np.vstack([f(x) for x in unit_sd_3d.cell_centers.T]).T
    f_known[unit_sd_3d.dim :, :] = 0
    f_known = f_known.ravel()

    P = discr.eval_at_cell_centers(unit_sd_3d)
    f_interp = P @ discr.interpolate(unit_sd_3d, f)

    assert np.allclose(f_interp, f_known)


def test_broken_curl(discr, unit_sd_3d):
    range_disc = discr.get_range_discr_class(unit_sd_3d.dim)()
    proj_0 = pg.proj_to_PwPolynomials(range_disc, unit_sd_3d, discr.poly_order - 1)
    curl = discr.assemble_diff_matrix(unit_sd_3d)

    proj_1 = discr.proj_to_PwPolynomials(unit_sd_3d)
    pwp = pg.get_PwPolynomials(discr.poly_order, pg.VECTOR)()
    broken_curl = pwp.assemble_broken_curl_matrix(unit_sd_3d)

    assert np.allclose((proj_0 @ curl - broken_curl @ proj_1).data, 0)


def test_interp_eval_2d(discr, unit_sd_2d, mdg):
    # In 2D the spaces are rotated RT0 spaces. On the 2D grids of the mixed-dimensional
    # grids, the order of the nodes of an edge can be opposite to its orientation:
    # with P the projection on the plane of the grid, a field f is represented as P f
    a = np.array([0.3, -0.7, 1.1])
    A = np.array([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [3.0, 1.0, 1.0]])
    if isinstance(discr, pg.NedelecR0):
        f = lambda x: np.tile(a.reshape((-1, 1)), x.shape[1])
    else:
        f = lambda x: a.reshape((-1, 1)) + A @ x

    for sd in [unit_sd_2d] + mdg.subdomains(dim=2):
        P = sd.rotation_matrix.T @ sd.rotation_matrix
        u = discr.interpolate(sd, f)
        f_interp = discr.eval_at_cell_centers(sd) @ u
        assert np.allclose(f_interp, (P @ f(sd.cell_centers)).ravel())


def test_proj_between_nedelec(mdg):
    # The constants interpolated in NedelecF1 are the ones of NedelecR0, also when the
    # order of the nodes of an edge is opposite to its orientation
    ne0, ne1 = pg.NedelecR0(), pg.NedelecF1()
    f = lambda _: np.array([0.3, -0.7, 1.1])
    for sd in mdg.subdomains():
        if sd.dim < 2:
            continue
        u0, u1 = ne0.interpolate(sd, f), ne1.interpolate(sd, f)
        assert np.allclose(ne1.proj_to_Ne0(sd) @ u1, u0)
        assert np.allclose(ne0.proj_to_Ne1(sd) @ u0, u1)
