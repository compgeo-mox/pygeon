"""Module contains specific tests for the vector RT0 discretization."""

import pytest

import pygeon as pg


@pytest.fixture
def discr():
    return pg.VecRT0("test")


def test_ndof(discr, ref_sd):
    known = [0, 2, 6, 12]
    assert discr.ndof(ref_sd) == known[ref_sd.dim]
