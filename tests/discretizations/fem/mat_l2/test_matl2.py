"""Module contains specific tests for the polynomial matrix discretization."""

import pytest

import pygeon as pg


@pytest.fixture
def discr() -> pg.MatPwPolynomials:
    return pg.MatPwPolynomials()


def test_trace(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.assemble_trace_matrix(ref_square)


def test_asym(discr, ref_square):
    with pytest.raises(NotImplementedError):
        discr.assemble_asym_matrix(ref_square)
