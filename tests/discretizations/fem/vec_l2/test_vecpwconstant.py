"""Module contains specific tests for the vector P0 discretization."""

import numpy as np
import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.VecPwConstants:
    return pg.VecPwConstants("test")


def test_ndof(discr, unit_sd):
    assert discr.ndof(unit_sd) == unit_sd.num_cells * unit_sd.dim


def test_assemble_mass_matrix(discr, ref_sd):
    from math import factorial

    M = discr.assemble_mass_matrix(ref_sd)
    L = discr.assemble_lumped_matrix(ref_sd)
    P = discr.eval_at_cell_centers(ref_sd)

    assert np.allclose(M.data, factorial(ref_sd.dim))
    assert np.allclose(M.indptr[:-1], M.indices)
    assert np.allclose((M - L).data, 0)
    assert np.allclose((M - P).data, 0)
