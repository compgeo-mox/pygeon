"""Module contains specific tests for the vector RT1 discretization."""

import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecRT1("test")


def test_ndof(discr, ref_sd):
    known = [0, 3, 16, 45]
    assert discr.ndof(ref_sd) == known[ref_sd.dim]
