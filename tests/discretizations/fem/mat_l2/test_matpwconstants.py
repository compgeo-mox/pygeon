"""Module contains specific tests for the matrix P0 discretization."""

import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.MatPwConstants("test")


def test_ndof(discr, ref_sd):
    assert discr.ndof(ref_sd) == ref_sd.dim**2


def test_assemble_symmetrizing_matrix(discr, ref_sd_0d):
    with pytest.raises(ValueError):
        discr.assemble_symmetrizing_matrix(ref_sd_0d)
