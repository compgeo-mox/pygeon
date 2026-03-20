"""Module contains specific tests for the vector P2 discretization."""

import pytest
import scipy.linalg as spla

import pygeon as pg
from tests.helpers import matrix_equals


@pytest.fixture
def discr() -> pg.VecPwQuadratics:
    return pg.VecPwQuadratics("test")


def test_ndof(discr, unit_sd):
    assert (
        discr.ndof(unit_sd)
        == unit_sd.num_cells
        * ((unit_sd.dim * (unit_sd.dim + 1)) // 2 + unit_sd.dim + 1)
        * unit_sd.dim
    )


def test_assemble_mass_matrix(discr, ref_sd):
    M = discr.assemble_mass_matrix(ref_sd)

    quad = pg.PwQuadratics()
    M_base = quad.assemble_mass_matrix(ref_sd).todense()
    factor = discr.ndof(ref_sd) // quad.ndof(ref_sd)

    M_known = spla.block_diag(*([M_base] * factor))

    assert matrix_equals(M.todense(), M_known)
