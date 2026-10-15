"""Module contains general tests for all H1 discretizations."""

import numpy as np
import porepy as pp
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


@pytest.fixture(params=[0.0, np.pi / 4])
def rotation(request: pytest.FixtureRequest) -> np.ndarray:
    """Rotation of the given angle around the y axis, which tilts the 1D grid along x
    and the 2D grid in the xy-plane out of their coordinate planes."""
    c, s = np.cos(request.param), np.sin(request.param)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_interpolate_and_evaluate(discr, unit_sd):
    func = lambda x: x[0] ** discr.poly_order
    known_vals = func(unit_sd.cell_centers)

    interp = discr.interpolate(unit_sd, func)
    proj = discr.eval_at_cell_centers(unit_sd)

    assert np.allclose(proj @ interp, known_vals)


def test_lumped_consistency(discr, unit_sd):
    M_lumped = discr.assemble_lumped_matrix(unit_sd)
    M_full = discr.assemble_mass_matrix(unit_sd)

    one_interp = discr.interpolate(unit_sd, lambda _: 1)

    integral_L = M_lumped @ one_interp
    integral_M = M_full @ one_interp

    assert np.allclose(integral_L, integral_M)


def test_stiffness_consistency(discr, unit_sd):
    """Compare the implemented stiffness matrix
    to the one obtained by mapping to the range discretization"""

    Stiff_1 = discr.assemble_stiff_matrix(unit_sd)
    Stiff_2 = discr.assemble_grad_grad_matrix(unit_sd)

    diff = Stiff_1 - Stiff_2
    assert np.allclose(diff.data, 0)


def test_differential_consistency(discr, unit_sd):
    pi_grad = discr.assemble_diff_matrix(unit_sd)
    range_dis = discr.get_range_discr_class(unit_sd.dim)()
    pi_grad = range_dis.proj_to_PwPolynomials(unit_sd) @ pi_grad

    grad_pi = discr.assemble_broken_grad_matrix(unit_sd)

    if isinstance(discr, pg.Lagrange1) and unit_sd.dim > 1:
        proj = pg.VecPwConstants().proj_to_higher_PwPolynomials(unit_sd)
        grad_pi = proj @ grad_pi

    if unit_sd.dim == 2:
        # In 2D, the differential is a rotated gradient. So the matrices don't
        # coincide, but their squares do.
        diff = pi_grad.T @ pi_grad - grad_pi.T @ grad_pi
    else:
        diff = pi_grad - grad_pi

    assert np.allclose(diff.data, 0)


def test_stiffness_of_linear(discr, ref_sd):
    stiff = discr.assemble_stiff_matrix(ref_sd)

    sol_func = lambda x: x[0]
    interp = discr.interpolate(ref_sd, sol_func)

    assert np.isclose(interp @ stiff @ interp, ref_sd.cell_volumes[0])


def test_point_grid(discr, ref_sd_0d):
    """Tests with a point grid"""

    assert discr.ndof(ref_sd_0d) == 0
    assert discr.assemble_diff_matrix(ref_sd_0d).nnz == 0
    assert discr.eval_at_cell_centers(ref_sd_0d).nnz == 0
    assert discr.interpolate(ref_sd_0d, lambda x: x).size == 0

    with pytest.raises(NotImplementedError):
        discr.get_range_discr_class(ref_sd_0d.dim)

    with pytest.raises(ValueError):
        sd_copy = ref_sd_0d.copy()
        sd_copy.dim = -1
        discr.assemble_diff_matrix(sd_copy)


def test_postprocess_darcy_flux(discr, unit_sd, rotation):
    # The problem is posed on the grid rotated by the given rotation, with the tensor
    # and the linear function rotated accordingly, so the flux is the rotated flux
    # of the unrotated problem.
    sd = unit_sd.copy(keep_history=False)
    sd.nodes = rotation @ sd.nodes
    sd.compute_geometry()

    K = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    K_rot = rotation @ K @ rotation.T
    ones = np.ones(sd.num_cells)
    perm = pp.SecondOrderTensor(
        kxx=K_rot[0, 0] * ones,
        kyy=K_rot[1, 1] * ones,
        kzz=K_rot[2, 2] * ones,
        kxy=K_rot[0, 1] * ones,
        kxz=K_rot[0, 2] * ones,
        kyz=K_rot[1, 2] * ones,
    )
    data = pp.initialize_data({}, "test", {pg.SECOND_ORDER_TENSOR: perm})

    slope = np.array([1.0, 2.0, 3.0])
    p = discr.interpolate(sd, lambda x: (rotation @ slope) @ x)
    q = discr.postprocess_darcy_flux(p, sd, data)

    # Only the part of the gradient and of the flux along the grid is seen
    P = np.diag([1.0] * sd.dim + [0.0] * (3 - sd.dim))
    q_known = rotation @ (-P @ K @ P @ slope)

    q_space = pg.get_PwPolynomials(discr.poly_order - 1, pg.VECTOR)("test")

    assert np.allclose(q, q_space.interpolate(sd, lambda _: q_known))
